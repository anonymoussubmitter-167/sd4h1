"""Ortholog mapping for cross-species gene name translation.

Supports Ensembl BioMart integration for one-to-one ortholog retrieval,
with JSON caching and a gene-name-matching fallback.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Ensembl BioMart species identifiers
SPECIES_ENSEMBL = {
    "human": "hsapiens",
    "mouse": "mmusculus",
    "zebrafish": "drerio",
}

# BioMart dataset names
SPECIES_DATASET = {
    "human": "hsapiens_gene_ensembl",
    "mouse": "mmusculus_gene_ensembl",
    "zebrafish": "drerio_gene_ensembl",
}

# Default cache directory (relative to project root)
_DEFAULT_CACHE_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "orthologs"


@dataclass
class OrthologMapping:
    """Bidirectional mapping of orthologous genes between two species."""

    source_species: str
    target_species: str
    # gene_name -> gene_name mapping
    forward: dict[str, str] = field(default_factory=dict)
    reverse: dict[str, str] = field(default_factory=dict)

    def map_gene(self, gene: str, direction: str = "forward") -> str | None:
        """Map a gene name from source to target species (or reverse)."""
        mapping = self.forward if direction == "forward" else self.reverse
        return mapping.get(gene)

    @property
    def n_orthologs(self) -> int:
        """Number of ortholog pairs."""
        return len(self.forward)


def _build_biomart_xml(source: str, target: str) -> str:
    """Build BioMart XML query for one-to-one orthologs.

    Queries the source species dataset for gene names and their
    one-to-one ortholog gene names in the target species.
    """
    source_dataset = SPECIES_DATASET[source]
    target_prefix = SPECIES_ENSEMBL[target]

    xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE Query>
<Query virtualSchemaName="default" formatter="TSV" header="1"
       uniqueRows="1" count="" datasetConfigVersion="0.6">
  <Dataset name="{source_dataset}" interface="default">
    <Filter name="with_{target_prefix}_homolog" excluded="0"/>
    <Attribute name="external_gene_name"/>
    <Attribute name="{target_prefix}_homolog_associated_gene_name"/>
    <Attribute name="{target_prefix}_homolog_orthology_type"/>
  </Dataset>
</Query>"""
    return xml


def fetch_orthologs_biomart(
    source: str,
    target: str,
    timeout: int = 120,
) -> dict[str, str]:
    """Query Ensembl BioMart REST API for one-to-one orthologs.

    Args:
        source: Source species name (human, mouse, zebrafish).
        target: Target species name (human, mouse, zebrafish).
        timeout: Request timeout in seconds.

    Returns:
        Dict mapping source gene names to target gene names.
    """
    import urllib.parse
    import urllib.request

    if source not in SPECIES_ENSEMBL:
        raise ValueError(f"Unknown source species: {source}. Supported: {list(SPECIES_ENSEMBL)}")
    if target not in SPECIES_ENSEMBL:
        raise ValueError(f"Unknown target species: {target}. Supported: {list(SPECIES_ENSEMBL)}")

    xml = _build_biomart_xml(source, target)
    url = "http://www.ensembl.org/biomart/martservice"

    data = urllib.parse.urlencode({"query": xml}).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST")

    logger.info("Fetching %s->%s orthologs from Ensembl BioMart...", source, target)

    response = urllib.request.urlopen(req, timeout=timeout)
    text = response.read().decode("utf-8")

    mapping: dict[str, str] = {}
    lines = text.strip().split("\n")
    for line in lines[1:]:  # skip header
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) >= 3:
            src_gene = parts[0].strip()
            tgt_gene = parts[1].strip()
            orth_type = parts[2].strip()
            if src_gene and tgt_gene and orth_type == "ortholog_one2one":
                mapping[src_gene] = tgt_gene

    logger.info("Retrieved %d one-to-one orthologs for %s->%s", len(mapping), source, target)
    return mapping


def build_ortholog_mapping_from_gene_names(
    source_genes: list[str],
    target_genes: list[str],
    source_species: str = "human",
    target_species: str = "mouse",
) -> OrthologMapping:
    """Build ortholog mapping using simple gene name matching (uppercase normalization).

    Fallback when BioMart is unreachable. Matches genes by uppercased name.
    Less accurate than true ortholog mapping but provides a degraded baseline.

    Args:
        source_genes: Gene names from source species.
        target_genes: Gene names from target species.
        source_species: Source species name.
        target_species: Target species name.

    Returns:
        OrthologMapping with name-matched gene pairs.
    """
    # Build lookup by uppercase name
    target_by_upper: dict[str, str] = {}
    for g in target_genes:
        upper = g.upper()
        if upper not in target_by_upper:
            target_by_upper[upper] = g

    forward: dict[str, str] = {}
    reverse: dict[str, str] = {}
    for g in source_genes:
        upper = g.upper()
        if upper in target_by_upper:
            tgt = target_by_upper[upper]
            forward[g] = tgt
            reverse[tgt] = g

    logger.info(
        "Gene name matching: %d orthologs for %s->%s (from %d source, %d target genes)",
        len(forward), source_species, target_species, len(source_genes), len(target_genes),
    )
    return OrthologMapping(
        source_species=source_species,
        target_species=target_species,
        forward=forward,
        reverse=reverse,
    )


def _cache_path(source: str, target: str, cache_dir: Path) -> Path:
    """Get the cache file path for a species pair."""
    return cache_dir / f"{source}_{target}.json"


def _save_cache(mapping: OrthologMapping, path: Path) -> None:
    """Save ortholog mapping to JSON cache."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data: dict[str, Any] = {
        "source_species": mapping.source_species,
        "target_species": mapping.target_species,
        "forward": mapping.forward,
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    logger.info("Cached %d orthologs to %s", mapping.n_orthologs, path)


def _load_cache(path: Path) -> OrthologMapping | None:
    """Load ortholog mapping from JSON cache, or None if not found."""
    if not path.exists():
        return None
    try:
        with open(path) as f:
            data = json.load(f)
        forward = data["forward"]
        reverse = {v: k for k, v in forward.items()}
        mapping = OrthologMapping(
            source_species=data["source_species"],
            target_species=data["target_species"],
            forward=forward,
            reverse=reverse,
        )
        logger.info("Loaded %d orthologs from cache: %s", mapping.n_orthologs, path)
        return mapping
    except (json.JSONDecodeError, KeyError) as e:
        logger.warning("Failed to load cache %s: %s", path, e)
        return None


def load_ortholog_mapping(
    source_species: str = "human",
    target_species: str = "mouse",
    cache_dir: str | Path | None = None,
    use_biomart: bool = True,
) -> OrthologMapping:
    """Load ortholog mapping between two species.

    Tries in order:
    1. Load from JSON cache
    2. Fetch from Ensembl BioMart (if use_biomart=True)
    3. Return empty mapping (caller can use gene name fallback)

    Args:
        source_species: Source species (human, mouse, zebrafish).
        target_species: Target species (human, mouse, zebrafish).
        cache_dir: Directory for cached mappings. Defaults to data/orthologs/.
        use_biomart: Whether to try BioMart if cache miss.

    Returns:
        Populated OrthologMapping (or empty if all methods fail).
    """
    if cache_dir is None:
        cache_dir = _DEFAULT_CACHE_DIR
    cache_dir = Path(cache_dir)

    # Try cache first
    path = _cache_path(source_species, target_species, cache_dir)
    cached = _load_cache(path)
    if cached is not None:
        return cached

    # Also check reverse cache (if we have target->source, invert it)
    path_rev = _cache_path(target_species, source_species, cache_dir)
    cached_rev = _load_cache(path_rev)
    if cached_rev is not None:
        mapping = OrthologMapping(
            source_species=source_species,
            target_species=target_species,
            forward=cached_rev.reverse,
            reverse=cached_rev.forward,
        )
        # Save forward direction cache too
        _save_cache(mapping, path)
        return mapping

    # Try BioMart
    if use_biomart:
        try:
            forward = fetch_orthologs_biomart(source_species, target_species)
            reverse = {v: k for k, v in forward.items()}
            mapping = OrthologMapping(
                source_species=source_species,
                target_species=target_species,
                forward=forward,
                reverse=reverse,
            )
            _save_cache(mapping, path)
            return mapping
        except Exception as e:
            logger.warning("BioMart fetch failed for %s->%s: %s", source_species, target_species, e)

    # Return empty mapping
    logger.warning(
        "No ortholog mapping available for %s->%s. Use build_ortholog_mapping_from_gene_names() as fallback.",
        source_species, target_species,
    )
    return OrthologMapping(
        source_species=source_species,
        target_species=target_species,
    )
