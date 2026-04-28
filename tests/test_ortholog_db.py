"""Tests for ortholog mapping module."""

import json
import tempfile
from pathlib import Path

import pytest

from rosetta.data.ortholog_db import (
    OrthologMapping,
    _build_biomart_xml,
    _cache_path,
    _load_cache,
    _save_cache,
    build_ortholog_mapping_from_gene_names,
    load_ortholog_mapping,
)


class TestOrthologMapping:
    """Test OrthologMapping dataclass."""

    def test_empty_mapping(self):
        m = OrthologMapping(source_species="human", target_species="mouse")
        assert m.n_orthologs == 0
        assert m.map_gene("TP53") is None

    def test_forward_mapping(self):
        m = OrthologMapping(
            source_species="human",
            target_species="mouse",
            forward={"TP53": "Trp53", "BRCA1": "Brca1"},
            reverse={"Trp53": "TP53", "Brca1": "BRCA1"},
        )
        assert m.n_orthologs == 2
        assert m.map_gene("TP53", "forward") == "Trp53"
        assert m.map_gene("BRCA1", "forward") == "Brca1"

    def test_reverse_mapping(self):
        m = OrthologMapping(
            source_species="human",
            target_species="mouse",
            forward={"TP53": "Trp53"},
            reverse={"Trp53": "TP53"},
        )
        assert m.map_gene("Trp53", "reverse") == "TP53"

    def test_missing_gene_returns_none(self):
        m = OrthologMapping(
            source_species="human",
            target_species="mouse",
            forward={"TP53": "Trp53"},
            reverse={"Trp53": "TP53"},
        )
        assert m.map_gene("NONEXISTENT") is None


class TestGeneNameFallback:
    """Test gene name matching fallback."""

    def test_basic_matching(self):
        source_genes = ["TP53", "BRCA1", "EGFR", "UNIQUE_HUMAN"]
        target_genes = ["Tp53", "Brca1", "Egfr", "Unique_mouse"]

        mapping = build_ortholog_mapping_from_gene_names(
            source_genes, target_genes, "human", "mouse"
        )
        # TP53 -> Tp53 (both uppercase to TP53), BRCA1 -> Brca1, EGFR -> Egfr
        assert mapping.n_orthologs == 3
        assert mapping.map_gene("TP53") == "Tp53"
        assert mapping.map_gene("BRCA1") == "Brca1"
        assert mapping.map_gene("EGFR") == "Egfr"

    def test_no_matches(self):
        source_genes = ["GENE_A", "GENE_B"]
        target_genes = ["gene_c", "gene_d"]

        mapping = build_ortholog_mapping_from_gene_names(
            source_genes, target_genes, "human", "mouse"
        )
        assert mapping.n_orthologs == 0

    def test_exact_case_match(self):
        source_genes = ["ACTB", "GAPDH"]
        target_genes = ["ACTB", "GAPDH"]  # Same case

        mapping = build_ortholog_mapping_from_gene_names(
            source_genes, target_genes, "human", "human"
        )
        assert mapping.n_orthologs == 2

    def test_reverse_direction(self):
        source_genes = ["TP53"]
        target_genes = ["Tp53"]

        mapping = build_ortholog_mapping_from_gene_names(
            source_genes, target_genes, "human", "mouse"
        )
        assert mapping.map_gene("Tp53", "reverse") == "TP53"


class TestCaching:
    """Test JSON caching of ortholog mappings."""

    def test_save_and_load(self):
        mapping = OrthologMapping(
            source_species="human",
            target_species="mouse",
            forward={"TP53": "Trp53", "BRCA1": "Brca1"},
            reverse={"Trp53": "TP53", "Brca1": "BRCA1"},
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_file = Path(tmpdir) / "human_mouse.json"
            _save_cache(mapping, cache_file)

            assert cache_file.exists()

            loaded = _load_cache(cache_file)
            assert loaded is not None
            assert loaded.source_species == "human"
            assert loaded.target_species == "mouse"
            assert loaded.n_orthologs == 2
            assert loaded.map_gene("TP53") == "Trp53"
            assert loaded.map_gene("Trp53", "reverse") == "TP53"

    def test_load_missing_file(self):
        result = _load_cache(Path("/nonexistent/file.json"))
        assert result is None

    def test_load_corrupt_cache(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_file = Path(tmpdir) / "corrupt.json"
            cache_file.write_text("not valid json{{{")
            result = _load_cache(cache_file)
            assert result is None

    def test_cache_path_format(self):
        path = _cache_path("human", "mouse", Path("/data/orthologs"))
        assert path == Path("/data/orthologs/human_mouse.json")


class TestLoadOrthologMapping:
    """Test the main load_ortholog_mapping function."""

    def test_from_cache(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Pre-populate cache
            cache_file = Path(tmpdir) / "human_mouse.json"
            data = {
                "source_species": "human",
                "target_species": "mouse",
                "forward": {"TP53": "Trp53"},
            }
            with open(cache_file, "w") as f:
                json.dump(data, f)

            mapping = load_ortholog_mapping(
                "human", "mouse", cache_dir=tmpdir, use_biomart=False
            )
            assert mapping.n_orthologs == 1
            assert mapping.map_gene("TP53") == "Trp53"

    def test_reverse_cache_lookup(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Only have mouse->human cache, but request human->mouse
            cache_file = Path(tmpdir) / "mouse_human.json"
            data = {
                "source_species": "mouse",
                "target_species": "human",
                "forward": {"Trp53": "TP53"},
            }
            with open(cache_file, "w") as f:
                json.dump(data, f)

            mapping = load_ortholog_mapping(
                "human", "mouse", cache_dir=tmpdir, use_biomart=False
            )
            assert mapping.source_species == "human"
            assert mapping.target_species == "mouse"
            assert mapping.map_gene("TP53") == "Trp53"

    def test_no_cache_no_biomart_returns_empty(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mapping = load_ortholog_mapping(
                "human", "mouse", cache_dir=tmpdir, use_biomart=False
            )
            assert mapping.n_orthologs == 0
            assert mapping.source_species == "human"
            assert mapping.target_species == "mouse"


class TestBioMartXML:
    """Test BioMart XML query generation."""

    def test_human_mouse_xml(self):
        xml = _build_biomart_xml("human", "mouse")
        assert "hsapiens_gene_ensembl" in xml
        assert "mmusculus_homolog" in xml
        assert "orthology_type" in xml

    def test_human_zebrafish_xml(self):
        xml = _build_biomart_xml("human", "zebrafish")
        assert "hsapiens_gene_ensembl" in xml
        assert "drerio_homolog" in xml

    def test_mouse_zebrafish_xml(self):
        xml = _build_biomart_xml("mouse", "zebrafish")
        assert "mmusculus_gene_ensembl" in xml
        assert "drerio_homolog" in xml
