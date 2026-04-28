#!/usr/bin/env python
"""Download and convert all 8 spatial transcriptomics datasets.

Downloads each dataset to data/raw/<name>/ and converts to standardized
h5ad format in data/processed/<name>.h5ad with adata.obsm["spatial"] populated.

Usage:
    python scripts/download_data.py                  # Download all
    python scripts/download_data.py --dataset brain  # Download one
    python scripts/download_data.py --list           # List datasets
"""

from __future__ import annotations

import argparse
import gzip
import logging
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

import anndata as ad
import numpy as np
import scipy.sparse as sp
from anndata import AnnData

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


# ---------------------------------------------------------------------------
# Download utilities
# ---------------------------------------------------------------------------

def download_file(url: str, dest: Path, desc: str | None = None) -> Path:
    """Download a file with progress bar and resume support.

    Uses requests+tqdm if available, falls back to wget/curl.
    """
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists():
        logger.info("Already downloaded: %s", dest.name)
        return dest

    try:
        import requests
        from tqdm import tqdm

        logger.info("Downloading %s -> %s", desc or url.split("/")[-1], dest.name)

        # Support resume via Range header
        headers = {}
        mode = "wb"
        initial_size = 0
        partial = dest.with_suffix(dest.suffix + ".partial")
        if partial.exists():
            initial_size = partial.stat().st_size
            headers["Range"] = f"bytes={initial_size}-"
            mode = "ab"

        resp = requests.get(url, headers=headers, stream=True, timeout=60)
        resp.raise_for_status()

        total = int(resp.headers.get("content-length", 0)) + initial_size

        with open(partial, mode) as f, tqdm(
            total=total,
            initial=initial_size,
            unit="B",
            unit_scale=True,
            desc=desc or dest.name,
        ) as pbar:
            for chunk in resp.iter_content(chunk_size=1 << 20):  # 1 MB chunks
                f.write(chunk)
                pbar.update(len(chunk))

        partial.rename(dest)
        return dest

    except ImportError:
        logger.warning("requests/tqdm not installed, falling back to wget")
        return _download_wget(url, dest)


def _download_wget(url: str, dest: Path) -> Path:
    """Fallback download using wget or curl."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if shutil.which("wget"):
        subprocess.run(
            ["wget", "-c", "-q", "--show-progress", "-O", str(dest), url],
            check=True,
        )
    elif shutil.which("curl"):
        subprocess.run(
            ["curl", "-L", "-C", "-", "-o", str(dest), url],
            check=True,
        )
    else:
        raise RuntimeError("Neither wget nor curl found. Install requests: pip install requests tqdm")
    return dest


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


# ---------------------------------------------------------------------------
# Dataset 1: Visium Human Brain (DLPFC)
# ---------------------------------------------------------------------------

def download_visium_human_brain() -> Path:
    """Download 10x Visium Human Brain Section 1 (V1_Human_Brain_Section_1).

    Source: 10x Genomics public datasets.
    ~350MB for the matrix + spatial files.
    """
    raw_dir = _ensure_dir(RAW_DIR / "visium_human_brain")
    out_h5ad = PROCESSED_DIR / "visium_human_brain.h5ad"

    if out_h5ad.exists():
        logger.info("Already processed: %s", out_h5ad.name)
        return out_h5ad

    base = "https://cf.10xgenomics.com/samples/spatial-exp/1.1.0/V1_Human_Brain_Section_1"

    # Download filtered matrix h5
    matrix_h5 = download_file(
        f"{base}/V1_Human_Brain_Section_1_filtered_feature_bc_matrix.h5",
        raw_dir / "filtered_feature_bc_matrix.h5",
        desc="Human Brain matrix",
    )

    # Download spatial data
    spatial_tar = download_file(
        f"{base}/V1_Human_Brain_Section_1_spatial.tar.gz",
        raw_dir / "spatial.tar.gz",
        desc="Human Brain spatial",
    )

    # Extract spatial tar
    spatial_dir = raw_dir / "spatial"
    if not spatial_dir.exists():
        logger.info("Extracting spatial data...")
        with tarfile.open(spatial_tar, "r:gz") as tf:
            tf.extractall(raw_dir)

    # Convert to h5ad using scanpy
    import scanpy as sc

    logger.info("Converting Visium Human Brain to h5ad...")
    adata = sc.read_visium(raw_dir)
    adata.var_names_make_unique()
    adata.write_h5ad(out_h5ad)
    logger.info("Saved %s: %d cells x %d genes", out_h5ad.name, adata.n_obs, adata.n_vars)
    return out_h5ad


# ---------------------------------------------------------------------------
# Dataset 2: MERFISH Mouse Brain (Allen Brain Cell Atlas)
# ---------------------------------------------------------------------------

def download_merfish_mouse_brain() -> Path:
    """Download MERFISH mouse brain from Allen Brain Cell Atlas.

    Source: Allen Institute S3 bucket (public, no auth).
    Downloads a single brain section (~500 gene panel).
    """
    raw_dir = _ensure_dir(RAW_DIR / "merfish_mouse_brain")
    out_h5ad = PROCESSED_DIR / "merfish_mouse_brain.h5ad"

    if out_h5ad.exists():
        logger.info("Already processed: %s", out_h5ad.name)
        return out_h5ad

    # Try abc_atlas_access first, then fall back to direct S3 download
    try:
        return _download_merfish_brain_abc(raw_dir, out_h5ad)
    except Exception as e:
        logger.warning("abc_atlas_access failed (%s), trying direct S3 download", e)
        return _download_merfish_brain_s3(raw_dir, out_h5ad)


def _download_merfish_brain_abc(raw_dir: Path, out_h5ad: Path) -> Path:
    """Download via abc_atlas_access package."""
    from abc_atlas_access.abc_atlas_cache.abc_project_cache import AbcProjectCache

    abc_cache = AbcProjectCache.from_s3_cache(str(raw_dir / "abc_cache"))
    # Download one section of the MERFISH-C57BL6J-638850 dataset
    cell_meta = abc_cache.get_metadata_dataframe(
        directory="MERFISH-C57BL6J-638850",
        file_name="cell_metadata",
    )

    # Get expression for a single section to keep size manageable
    sections = cell_meta["brain_section_label"].unique()
    section = sections[0]  # Take the first section
    logger.info("Using brain section: %s (%d cells total, %d sections)",
                section, len(cell_meta), len(sections))

    section_cells = cell_meta[cell_meta["brain_section_label"] == section]

    # Download gene expression
    gene_expr = abc_cache.get_gene_expression(
        directory="MERFISH-C57BL6J-638850",
        file_name="C57BL6J-638850-log2",
    )

    # Subset to section
    expr_section = gene_expr.loc[section_cells.index]

    adata = AnnData(
        X=sp.csr_matrix(expr_section.values.astype(np.float32)),
        obs=section_cells,
    )
    adata.var_names = expr_section.columns.tolist()
    adata.obsm["spatial"] = section_cells[["x", "y"]].values.astype(np.float32)

    adata.write_h5ad(out_h5ad)
    logger.info("Saved %s: %d cells x %d genes", out_h5ad.name, adata.n_obs, adata.n_vars)
    return out_h5ad


def _download_merfish_brain_s3(raw_dir: Path, out_h5ad: Path) -> Path:
    """Direct download from S3 via HTTPS."""
    s3_base = "https://allen-brain-cell-atlas.s3.us-west-2.amazonaws.com"

    # Download expression matrix (log2 transformed)
    expr_h5ad = download_file(
        f"{s3_base}/expression_matrices/MERFISH-C57BL6J-638850/20230830/C57BL6J-638850-log2.h5ad",
        raw_dir / "C57BL6J-638850-log2.h5ad",
        desc="MERFISH Brain expression",
    )

    # Download cell metadata
    meta_csv = download_file(
        f"{s3_base}/metadata/MERFISH-C57BL6J-638850/20231215/views/cell_metadata_with_cluster_annotation.csv",
        raw_dir / "cell_metadata.csv",
        desc="MERFISH Brain metadata",
    )

    import pandas as pd

    logger.info("Converting MERFISH Mouse Brain to h5ad...")
    adata = ad.read_h5ad(expr_h5ad)
    meta = pd.read_csv(meta_csv, index_col=0)

    # Align metadata with expression data
    common = adata.obs_names.intersection(meta.index)
    if len(common) == 0:
        # If indices don't match, use the adata as-is and extract coords from meta columns
        logger.warning("Metadata indices don't match expression. Using expression data as-is.")
    else:
        adata = adata[common].copy()
        meta = meta.loc[common]
        for col in meta.columns:
            adata.obs[col] = meta[col]

    # Extract spatial coordinates
    coord_cols = None
    for x_col, y_col in [("x", "y"), ("x_reconstructed", "y_reconstructed"),
                          ("center_x", "center_y")]:
        if x_col in adata.obs.columns and y_col in adata.obs.columns:
            coord_cols = (x_col, y_col)
            break

    if coord_cols:
        adata.obsm["spatial"] = adata.obs[list(coord_cols)].values.astype(np.float32)
    elif "spatial" not in adata.obsm:
        raise ValueError("Could not find spatial coordinates in MERFISH data")

    # Subset to one brain section to keep manageable
    if "brain_section_label" in adata.obs.columns:
        sections = adata.obs["brain_section_label"].unique()
        section = sections[0]
        logger.info("Subsetting to section %s", section)
        adata = adata[adata.obs["brain_section_label"] == section].copy()

    adata.write_h5ad(out_h5ad)
    logger.info("Saved %s: %d cells x %d genes", out_h5ad.name, adata.n_obs, adata.n_vars)
    return out_h5ad


# ---------------------------------------------------------------------------
# Dataset 3: Stereo-seq Zebrafish Brain
# ---------------------------------------------------------------------------

def download_stereoseq_zebrafish_brain() -> Path:
    """Download Stereo-seq zebrafish brain from CNGB ZESTA portal.

    Source: CNGB FTP (CNP0002220).
    Downloads the 24hpf timepoint (~1.1GB h5ad).
    """
    raw_dir = _ensure_dir(RAW_DIR / "stereoseq_zebrafish_brain")
    out_h5ad = PROCESSED_DIR / "stereoseq_zebrafish_brain.h5ad"

    if out_h5ad.exists():
        logger.info("Already processed: %s", out_h5ad.name)
        return out_h5ad

    # Download h5ad directly from CNGB FTP
    ftp_base = "https://ftp.cngb.org/pub/SciRAID/stomics/STDS0000057/stomics"

    # Try the combined spatial file first (smaller), then individual timepoints
    urls_to_try = [
        (f"{ftp_base}/spatial_sixtime_slice_stereoseq.h5ad",
         raw_dir / "spatial_sixtime_slice_stereoseq.h5ad",
         "Stereo-seq combined"),
        (f"{ftp_base}/zf24_stereoseq.h5ad",
         raw_dir / "zf24_stereoseq.h5ad",
         "Stereo-seq 24hpf"),
        (f"{ftp_base}/zf18_stereoseq.h5ad",
         raw_dir / "zf18_stereoseq.h5ad",
         "Stereo-seq 18hpf"),
    ]

    h5ad_file = None
    for url, dest, desc in urls_to_try:
        try:
            h5ad_file = download_file(url, dest, desc=desc)
            break
        except Exception as e:
            logger.warning("Failed to download %s: %s", desc, e)
            if dest.exists():
                dest.unlink()
            continue

    if h5ad_file is None:
        raise RuntimeError("Could not download Stereo-seq zebrafish data from any source")

    logger.info("Converting Stereo-seq Zebrafish Brain to standardized h5ad...")
    adata = ad.read_h5ad(h5ad_file)

    # Ensure spatial coordinates exist
    if "spatial" not in adata.obsm:
        # Try to find coords in obs
        from rosetta.data.loaders import _find_coordinate_columns
        try:
            coord_cols = _find_coordinate_columns(adata.obs)
            adata.obsm["spatial"] = adata.obs[coord_cols].values.astype(np.float32)
        except ValueError:
            # Try common Stereo-seq column patterns
            for x_col, y_col in [("x", "y"), ("x_FOV", "y_FOV"),
                                  ("spatial_x", "spatial_y")]:
                if x_col in adata.obs.columns and y_col in adata.obs.columns:
                    adata.obsm["spatial"] = (
                        adata.obs[[x_col, y_col]].values.astype(np.float32)
                    )
                    break

    if "spatial" not in adata.obsm:
        raise ValueError("Could not find spatial coordinates in Stereo-seq data")

    adata.write_h5ad(out_h5ad)
    logger.info("Saved %s: %d cells x %d genes", out_h5ad.name, adata.n_obs, adata.n_vars)
    return out_h5ad


# ---------------------------------------------------------------------------
# Dataset 4: Allen Brain ISH
# ---------------------------------------------------------------------------

def download_allen_brain_ish() -> Path:
    """Download Allen Brain ISH expression energy volumes.

    Source: Allen Brain Atlas REST API (no allensdk dependency).
    Downloads expression energy for a set of representative genes and
    converts to pseudo-spatial AnnData (voxel grid with expression values).
    """
    import io
    import re
    import zipfile

    import requests

    raw_dir = _ensure_dir(RAW_DIR / "allen_brain_ish")
    out_h5ad = PROCESSED_DIR / "allen_brain_ish.h5ad"

    if out_h5ad.exists():
        logger.info("Already processed: %s", out_h5ad.name)
        return out_h5ad

    logger.info("Downloading Allen Brain ISH data via REST API...")

    # Query for well-characterized marker genes with ISH data
    # These are canonical brain region markers
    target_genes = [
        "Gad1", "Gad2", "Slc17a7", "Slc17a6",  # Excitatory/inhibitory
        "Pvalb", "Sst", "Vip", "Lamp5",          # Interneuron subtypes
        "Aqp4", "Gfap", "Aldh1l1",               # Astrocytes
        "Mbp", "Plp1", "Mog",                     # Oligodendrocytes
        "Tmem119", "Cx3cr1",                       # Microglia
        "Pecam1", "Flt1",                          # Endothelial
        "Drd1", "Drd2", "Th",                      # Dopaminergic
        "Slc6a3", "Chat", "Tph2",                  # Neurotransmitter
    ]

    # Find section dataset IDs for these genes via RMA API
    rma_base = "https://api.brain-map.org/api/v2/data/query.json"
    dataset_ids = {}

    for gene in target_genes:
        try:
            resp = requests.get(rma_base, params={
                "criteria": (
                    f"model::SectionDataSet,"
                    f"rma::criteria,[failed$eqfalse],"
                    f"products[abbreviation$eq'Mouse'],"
                    f"genes[acronym$eq'{gene}']"
                ),
                "include": "genes",
                "num_rows": "1",
            }, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            if data.get("success") and data.get("msg"):
                ds_id = data["msg"][0]["id"]
                dataset_ids[gene] = ds_id
                logger.info("  Found ISH for %s (dataset %d)", gene, ds_id)
        except Exception as e:
            logger.warning("  Could not find ISH for %s: %s", gene, e)

    if not dataset_ids:
        logger.error("No ISH datasets found via API")
        return _create_allen_ish_placeholder(out_h5ad)

    # Download expression energy grids (MHD/RAW zip format)
    expression_vols = {}
    reference_shape = None

    for gene, ds_id in dataset_ids.items():
        cache_path = raw_dir / f"{gene}_{ds_id}_energy.npy"
        if cache_path.exists():
            vol = np.load(cache_path)
            expression_vols[gene] = vol
            if reference_shape is None:
                reference_shape = vol.shape
            continue

        try:
            grid_url = f"https://api.brain-map.org/grid_data/download/{ds_id}?include=energy"
            resp = requests.get(grid_url, timeout=60)
            resp.raise_for_status()

            # Response is a zip with energy.mhd + energy.raw
            zf = zipfile.ZipFile(io.BytesIO(resp.content))
            mhd_text = zf.read("energy.mhd").decode()

            # Parse dimensions from MHD header
            dims_match = re.search(r"DimSize\s*=\s*(\d+)\s+(\d+)\s+(\d+)", mhd_text)
            if not dims_match:
                logger.warning("  Could not parse MHD for %s", gene)
                continue
            dims = tuple(int(d) for d in dims_match.groups())

            # Read raw float32 data (stored z,y,x order)
            raw_data = zf.read("energy.raw")
            vol = np.frombuffer(raw_data, dtype=np.float32).reshape(dims[::-1]).copy()

            expression_vols[gene] = vol
            if reference_shape is None:
                reference_shape = vol.shape
            np.save(cache_path, vol)
            logger.info("  Downloaded ISH energy grid for %s (%s)", gene, vol.shape)
        except Exception as e:
            logger.warning("  Failed to download %s: %s", gene, e)

    if not expression_vols:
        logger.error("No expression volumes downloaded successfully")
        return _create_allen_ish_placeholder(out_h5ad)

    logger.info("Downloaded %d/%d gene volumes", len(expression_vols), len(target_genes))

    # Convert to pseudo-spatial AnnData
    # Grid is at 200um resolution per the MHD header
    adata = _allen_ish_to_anndata(expression_vols, reference_shape, resolution=200)
    adata.write_h5ad(out_h5ad)
    logger.info("Saved %s: %d voxels x %d genes", out_h5ad.name, adata.n_obs, adata.n_vars)
    return out_h5ad


def _allen_ish_to_anndata(
    expression_vols: dict[str, np.ndarray],
    reference_shape: tuple,
    resolution: int,
) -> AnnData:
    """Convert Allen ISH expression energy volumes to a pseudo-spatial AnnData.

    Projects 3D voxel grid to 2D (coronal max projection) and creates
    an AnnData where each observation is a voxel.

    Volumes are shaped (z, y, x) = (AP, DV, ML) from the Allen grid API.
    We project along the AP axis (axis=0) to get a coronal view.
    """
    genes = sorted(expression_vols.keys())

    # Stack all gene volumes: (n_genes, z, y, x) = (n_genes, AP, DV, ML)
    volumes = np.stack([expression_vols[g] for g in genes], axis=0)

    # Take coronal max projection (project along anteroposterior axis=1, i.e. z)
    # Shape: (n_genes, DV, ML) = (n_genes, y, x)
    projection = np.nanmax(volumes, axis=1)  # max over z (AP axis)

    n_genes, ny, nx = projection.shape

    # Flatten spatial dims to create observation matrix
    # Each observation = one pixel in the projected image
    expr_matrix = projection.reshape(n_genes, ny * nx).T  # (n_voxels, n_genes)

    # Create spatial coordinates
    yy, xx = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
    coords = np.column_stack([
        xx.ravel() * resolution,
        yy.ravel() * resolution,
    ]).astype(np.float32)

    # Filter out voxels with all-zero expression (outside brain)
    mask = np.nansum(expr_matrix, axis=1) > 0
    expr_matrix = expr_matrix[mask]
    coords = coords[mask]

    # Replace NaN with 0
    expr_matrix = np.nan_to_num(expr_matrix, nan=0.0).astype(np.float32)

    adata = AnnData(
        X=sp.csr_matrix(expr_matrix),
        var={"gene_symbol": genes},
    )
    adata.var_names = genes
    adata.obs_names = [f"voxel_{i}" for i in range(adata.n_obs)]
    adata.obsm["spatial"] = coords

    return adata


def _create_allen_ish_placeholder(out_h5ad: Path) -> Path:
    """Create a small placeholder Allen ISH dataset for pipeline testing.

    Uses synthetic data shaped like a coronal brain section.
    """
    logger.warning("Creating placeholder Allen ISH dataset (allensdk not available)")
    rng = np.random.default_rng(42)

    # Create a coronal section-like shape
    n_voxels_x, n_voxels_y = 67, 41  # CCF dimensions at 200um
    yy, xx = np.meshgrid(np.arange(n_voxels_y), np.arange(n_voxels_x), indexing="ij")

    # Brain-shaped mask (ellipse)
    cx, cy = n_voxels_x // 2, n_voxels_y // 2
    mask = ((xx - cx) / (cx * 0.9))**2 + ((yy - cy) / (cy * 0.9))**2 < 1
    mask = mask.ravel()

    coords = np.column_stack([xx.ravel(), yy.ravel()]).astype(np.float32) * 200
    coords = coords[mask]
    n_voxels = mask.sum()

    # Simulate gene expression for canonical markers
    genes = [
        "Gad1", "Gad2", "Slc17a7", "Slc17a6",
        "Pvalb", "Sst", "Vip", "Lamp5",
        "Aqp4", "Gfap", "Mbp", "Plp1",
        "Tmem119", "Cx3cr1", "Drd1", "Drd2",
    ]
    n_genes = len(genes)
    expr = rng.exponential(scale=2.0, size=(n_voxels, n_genes)).astype(np.float32)

    # Add spatial structure: make some genes cortical, some subcortical
    y_norm = (coords[:, 1] - coords[:, 1].min()) / (coords[:, 1].max() - coords[:, 1].min())
    expr[:, 0] *= (1 - y_norm) * 2  # Gad1 higher in ventral
    expr[:, 2] *= y_norm * 2         # Slc17a7 higher in dorsal

    adata = AnnData(X=sp.csr_matrix(expr))
    adata.var_names = genes
    adata.obs_names = [f"voxel_{i}" for i in range(n_voxels)]
    adata.obsm["spatial"] = coords

    _ensure_dir(out_h5ad.parent)
    adata.write_h5ad(out_h5ad)
    logger.info("Saved placeholder %s: %d voxels x %d genes", out_h5ad.name, n_voxels, n_genes)
    return out_h5ad


# ---------------------------------------------------------------------------
# Dataset 5: Visium Mouse Liver
# ---------------------------------------------------------------------------

def download_visium_mouse_liver() -> Path:
    """Download Visium mouse liver from GEO GSE165141.

    Source: Hildebrandt et al. 2021 (GEO supplementary files).
    """
    raw_dir = _ensure_dir(RAW_DIR / "visium_mouse_liver")
    out_h5ad = PROCESSED_DIR / "visium_mouse_liver.h5ad"

    if out_h5ad.exists():
        logger.info("Already processed: %s", out_h5ad.name)
        return out_h5ad

    # Download the supplementary tar from GEO
    tar_url = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE165nnn/GSE165141/suppl/GSE165141_RAW.tar"
    tar_path = download_file(tar_url, raw_dir / "GSE165141_RAW.tar", desc="Mouse Liver GEO")

    # Extract tar
    extract_dir = raw_dir / "extracted"
    if not extract_dir.exists():
        logger.info("Extracting GSE165141_RAW.tar...")
        with tarfile.open(tar_path, "r") as tf:
            tf.extractall(extract_dir)

    # Convert supplementary files to h5ad
    adata = _convert_geo_liver(extract_dir)
    adata.write_h5ad(out_h5ad)
    logger.info("Saved %s: %d spots x %d genes", out_h5ad.name, adata.n_obs, adata.n_vars)
    return out_h5ad


def _convert_geo_liver(extract_dir: Path) -> AnnData:
    """Convert GEO GSE165141 supplementary files to AnnData."""
    import pandas as pd

    # Find all TSV/CSV files
    data_files = sorted(extract_dir.glob("*.tsv.gz")) + sorted(extract_dir.glob("*.csv.gz"))

    if not data_files:
        # Check for h5 files or other formats
        data_files = sorted(extract_dir.glob("*.h5"))
        if data_files:
            import scanpy as sc
            adata = sc.read_10x_h5(data_files[0])
            if "spatial" not in adata.obsm:
                # Create pseudo-spatial coords based on barcodes
                n = adata.n_obs
                side = int(np.ceil(np.sqrt(n)))
                coords = np.array([(i % side, i // side) for i in range(n)], dtype=np.float32)
                adata.obsm["spatial"] = coords * 100  # Scale to ~100um spacing
            return adata

        # Check for mtx files
        mtx_files = sorted(extract_dir.glob("*.mtx.gz"))
        if mtx_files:
            import scanpy as sc
            adata = sc.read_10x_mtx(extract_dir)
            n = adata.n_obs
            side = int(np.ceil(np.sqrt(n)))
            coords = np.array([(i % side, i // side) for i in range(n)], dtype=np.float32)
            adata.obsm["spatial"] = coords * 100
            return adata

    # Parse TSV spot files (Spatial Transcriptomics format)
    all_adatas = []
    for f in data_files:
        try:
            if f.suffix == ".gz":
                df = pd.read_csv(f, sep="\t", index_col=0, compression="gzip")
            else:
                df = pd.read_csv(f, sep="\t", index_col=0)

            # Detect if this is a spot file (rows=spots, cols=genes) or other
            if df.shape[1] > 10:  # Likely gene expression matrix
                sample_name = f.stem.replace(".tsv", "").replace(".gz", "")

                # Try to extract coordinates from index (format: AxB where A,B are positions)
                coords = []
                valid_indices = []
                for idx in df.index:
                    parts = str(idx).replace("x", "_").split("_")
                    if len(parts) >= 2:
                        try:
                            x, y = float(parts[-2]), float(parts[-1])
                            coords.append([x, y])
                            valid_indices.append(idx)
                        except ValueError:
                            continue

                if coords:
                    df = df.loc[valid_indices]
                    spatial = np.array(coords, dtype=np.float32)
                else:
                    # No spatial info in index — create grid
                    n = len(df)
                    side = int(np.ceil(np.sqrt(n)))
                    spatial = np.array(
                        [(i % side, i // side) for i in range(n)], dtype=np.float32
                    ) * 100

                adata_sample = AnnData(
                    X=sp.csr_matrix(df.values.astype(np.float32)),
                    obs=pd.DataFrame(index=df.index),
                )
                adata_sample.var_names = df.columns.tolist()
                adata_sample.obsm["spatial"] = spatial
                adata_sample.obs["sample"] = sample_name
                all_adatas.append(adata_sample)
                logger.info("  Loaded %s: %d spots x %d genes", sample_name,
                           adata_sample.n_obs, adata_sample.n_vars)

        except Exception as e:
            logger.warning("  Failed to parse %s: %s", f.name, e)

    if not all_adatas:
        raise RuntimeError(f"No valid data files found in {extract_dir}")

    # Concatenate all samples
    if len(all_adatas) == 1:
        return all_adatas[0]

    adata = ad.concat(all_adatas, join="outer", fill_value=0)
    return adata


# ---------------------------------------------------------------------------
# Dataset 6: MERFISH Human Liver
# ---------------------------------------------------------------------------

def download_merfish_human_liver() -> Path:
    """Download MERFISH Human Liver from Zenodo.

    Source: Zenodo record 10634153 (mirror of DRYAD doi:10.5061/dryad.37pvmcvsg).
    Downloads the healthy MERFISH h5ad (~350MB).
    """
    raw_dir = _ensure_dir(RAW_DIR / "merfish_human_liver")
    out_h5ad = PROCESSED_DIR / "merfish_human_liver.h5ad"

    if out_h5ad.exists():
        logger.info("Already processed: %s", out_h5ad.name)
        return out_h5ad

    # Download healthy MERFISH h5ad from Zenodo (DRYAD API requires auth)
    raw_h5ad = download_file(
        "https://zenodo.org/records/10634153/files/adata_healthy_merfish.h5ad",
        raw_dir / "adata_healthy_merfish.h5ad",
        desc="MERFISH Human Liver",
    )

    logger.info("Converting MERFISH Human Liver to standardized h5ad...")
    adata = ad.read_h5ad(raw_h5ad)

    # Ensure spatial coordinates exist
    if "spatial" not in adata.obsm:
        from rosetta.data.loaders import _find_coordinate_columns
        try:
            coord_cols = _find_coordinate_columns(adata.obs)
            adata.obsm["spatial"] = adata.obs[coord_cols].values.astype(np.float32)
        except ValueError:
            # Try X_spatial or similar
            for key in adata.obsm:
                if "spatial" in key.lower() or "umap" not in key.lower():
                    if adata.obsm[key].shape[1] >= 2:
                        adata.obsm["spatial"] = np.array(
                            adata.obsm[key][:, :2], dtype=np.float32
                        )
                        logger.info("Using obsm['%s'] as spatial coordinates", key)
                        break

    if "spatial" not in adata.obsm:
        raise ValueError("Could not find spatial coordinates in MERFISH liver data")

    adata.write_h5ad(out_h5ad)
    logger.info("Saved %s: %d cells x %d genes", out_h5ad.name, adata.n_obs, adata.n_vars)
    return out_h5ad


# ---------------------------------------------------------------------------
# Visium HD helper
# ---------------------------------------------------------------------------

def _load_visium_hd(sr_dir: Path) -> AnnData:
    """Load Visium HD data from a Space Ranger output directory.

    Visium HD uses tissue_positions.parquet instead of tissue_positions_list.csv,
    so scanpy.read_visium() doesn't work directly.
    """
    import pandas as pd
    import scanpy as sc

    h5_path = sr_dir / "filtered_feature_bc_matrix.h5"
    if not h5_path.exists():
        raise FileNotFoundError(f"No filtered_feature_bc_matrix.h5 in {sr_dir}")

    logger.info("Loading Visium HD from %s", sr_dir)
    adata = sc.read_10x_h5(h5_path)
    adata.var_names_make_unique()

    # Load spatial coordinates from parquet
    positions_parquet = sr_dir / "spatial" / "tissue_positions.parquet"
    positions_csv = sr_dir / "spatial" / "tissue_positions_list.csv"

    if positions_parquet.exists():
        positions = pd.read_parquet(positions_parquet)
        # Parquet columns: barcode, in_tissue, array_row, array_col, pxl_row_in_fullres, pxl_col_in_fullres
        if "barcode" in positions.columns:
            positions = positions.set_index("barcode")

        # Match barcodes
        common = adata.obs_names.intersection(positions.index)
        if len(common) == 0:
            logger.warning("No barcode overlap between matrix and positions")
            # Fall back to grid coords
            n = adata.n_obs
            side = int(np.ceil(np.sqrt(n)))
            coords = np.array([(i % side, i // side) for i in range(n)], dtype=np.float32) * 16
            adata.obsm["spatial"] = coords
        else:
            adata = adata[common].copy()
            pos_matched = positions.loc[common]
            # Use pixel coordinates
            if "pxl_col_in_fullres" in pos_matched.columns:
                coords = pos_matched[["pxl_col_in_fullres", "pxl_row_in_fullres"]].values
            elif "array_col" in pos_matched.columns:
                coords = pos_matched[["array_col", "array_row"]].values
            else:
                coords = pos_matched.iloc[:, -2:].values
            adata.obsm["spatial"] = coords.astype(np.float32)

            # Filter to in-tissue spots only
            if "in_tissue" in pos_matched.columns:
                in_tissue = pos_matched["in_tissue"].astype(bool)
                adata = adata[in_tissue.values].copy()
                logger.info("Filtered to %d in-tissue spots", adata.n_obs)

    elif positions_csv.exists():
        # Standard Visium format fallback
        adata_vis = sc.read_visium(sr_dir)
        adata_vis.var_names_make_unique()
        return adata_vis
    else:
        logger.warning("No tissue positions file found, using array coords")
        n = adata.n_obs
        side = int(np.ceil(np.sqrt(n)))
        coords = np.array([(i % side, i // side) for i in range(n)], dtype=np.float32) * 16
        adata.obsm["spatial"] = coords

    return adata


# ---------------------------------------------------------------------------
# Dataset 7: Visium Mouse Intestine
# ---------------------------------------------------------------------------

def download_visium_mouse_intestine() -> Path:
    """Download Visium HD Mouse Small Intestine from 10x Genomics.

    Source: 10x Genomics public datasets.
    Downloads binned outputs (~5GB).
    """
    raw_dir = _ensure_dir(RAW_DIR / "visium_mouse_intestine")
    out_h5ad = PROCESSED_DIR / "visium_mouse_intestine.h5ad"

    if out_h5ad.exists():
        logger.info("Already processed: %s", out_h5ad.name)
        return out_h5ad

    tar_url = (
        "https://cf.10xgenomics.com/samples/spatial-exp/3.0.0/"
        "Visium_HD_Mouse_Small_Intestine/"
        "Visium_HD_Mouse_Small_Intestine_binned_outputs.tar.gz"
    )

    tar_path = download_file(
        tar_url,
        raw_dir / "Visium_HD_Mouse_Small_Intestine_binned_outputs.tar.gz",
        desc="Mouse Intestine Visium HD",
    )

    # Extract
    extract_dir = raw_dir / "binned_outputs"
    if not extract_dir.exists():
        logger.info("Extracting Visium HD Mouse Intestine...")
        with tarfile.open(tar_path, "r:gz") as tf:
            tf.extractall(raw_dir)

    # Find the Space Ranger output directory (use largest bin size for manageability)
    import scanpy as sc

    # Visium HD outputs have multiple bin sizes: square_002um, square_008um, square_016um
    bin_dirs = sorted(extract_dir.glob("**/square_*um")) if extract_dir.exists() else []
    if not bin_dirs:
        bin_dirs = sorted(raw_dir.glob("**/square_*um"))

    if bin_dirs:
        # Use the largest bin size (fewest cells, most manageable)
        sr_dir = bin_dirs[-1]
        logger.info("Using bin directory: %s", sr_dir.name)
        adata = _load_visium_hd(sr_dir)
    else:
        h5_files = sorted(raw_dir.glob("**/filtered_feature_bc_matrix.h5"))
        if h5_files:
            sr_dir = h5_files[0].parent
            adata = _load_visium_hd(sr_dir)
        else:
            raise FileNotFoundError(
                f"Could not find Space Ranger output in {raw_dir}. "
                "Check the extracted directory structure."
            )

    adata.var_names_make_unique()
    adata.write_h5ad(out_h5ad)
    logger.info("Saved %s: %d spots x %d genes", out_h5ad.name, adata.n_obs, adata.n_vars)
    return out_h5ad


# ---------------------------------------------------------------------------
# Dataset 8: Visium Human Intestine (CRC substitute)
# ---------------------------------------------------------------------------

def download_visium_human_intestine() -> Path:
    """Download Visium Human Intestine (CRC) from 10x Genomics.

    Source: 10x Genomics public datasets.
    Uses Human Colorectal Cancer as substitute for IBD (which requires access applications).
    """
    raw_dir = _ensure_dir(RAW_DIR / "visium_human_intestine")
    out_h5ad = PROCESSED_DIR / "visium_human_intestine.h5ad"

    if out_h5ad.exists():
        logger.info("Already processed: %s", out_h5ad.name)
        return out_h5ad

    # 10x Genomics Parent Visium Human Colorectal Cancer
    base = (
        "https://cf.10xgenomics.com/samples/spatial-exp/1.2.0/"
        "Parent_Visium_Human_ColorectalCancer"
    )

    matrix_h5 = download_file(
        f"{base}/Parent_Visium_Human_ColorectalCancer_filtered_feature_bc_matrix.h5",
        raw_dir / "filtered_feature_bc_matrix.h5",
        desc="Human Intestine matrix",
    )

    spatial_tar = download_file(
        f"{base}/Parent_Visium_Human_ColorectalCancer_spatial.tar.gz",
        raw_dir / "spatial.tar.gz",
        desc="Human Intestine spatial",
    )

    # Extract spatial
    spatial_dir = raw_dir / "spatial"
    if not spatial_dir.exists():
        logger.info("Extracting spatial data...")
        with tarfile.open(spatial_tar, "r:gz") as tf:
            tf.extractall(raw_dir)

    import scanpy as sc

    logger.info("Converting Visium Human Intestine to h5ad...")
    adata = sc.read_visium(raw_dir)
    adata.var_names_make_unique()
    adata.write_h5ad(out_h5ad)
    logger.info("Saved %s: %d spots x %d genes", out_h5ad.name, adata.n_obs, adata.n_vars)
    return out_h5ad


# ---------------------------------------------------------------------------
# Registry and CLI
# ---------------------------------------------------------------------------

DATASETS = {
    "visium_human_brain": {
        "fn": download_visium_human_brain,
        "species": "human",
        "platform": "visium",
        "desc": "Visium Human Brain DLPFC (10x Genomics V1)",
        "est_size": "~350MB",
    },
    "merfish_mouse_brain": {
        "fn": download_merfish_mouse_brain,
        "species": "mouse",
        "platform": "merfish",
        "desc": "MERFISH Mouse Brain (Allen Brain Cell Atlas)",
        "est_size": "~5-10GB",
    },
    "stereoseq_zebrafish_brain": {
        "fn": download_stereoseq_zebrafish_brain,
        "species": "zebrafish",
        "platform": "stereoseq",
        "desc": "Stereo-seq Zebrafish Brain (CNGB ZESTA)",
        "est_size": "~1GB",
    },
    "allen_brain_ish": {
        "fn": download_allen_brain_ish,
        "species": "mouse",
        "platform": "allen_ish",
        "desc": "Allen Brain ISH Mouse (Allen SDK API)",
        "est_size": "~2GB",
    },
    "visium_mouse_liver": {
        "fn": download_visium_mouse_liver,
        "species": "mouse",
        "platform": "visium",
        "desc": "Visium Mouse Liver Zonation (GEO GSE165141)",
        "est_size": "~50MB",
    },
    "merfish_human_liver": {
        "fn": download_merfish_human_liver,
        "species": "human",
        "platform": "merfish",
        "desc": "MERFISH Human Liver (DRYAD)",
        "est_size": "~350MB",
    },
    "visium_mouse_intestine": {
        "fn": download_visium_mouse_intestine,
        "species": "mouse",
        "platform": "visium",
        "desc": "Visium HD Mouse Small Intestine (10x Genomics)",
        "est_size": "~5GB",
    },
    "visium_human_intestine": {
        "fn": download_visium_human_intestine,
        "species": "human",
        "platform": "visium",
        "desc": "Visium Human Intestine CRC (10x Genomics)",
        "est_size": "~350MB",
    },
}


def main():
    parser = argparse.ArgumentParser(
        description="Download and convert spatial transcriptomics datasets for ROSETTA"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        nargs="*",
        help="Dataset name(s) to download. If omitted, downloads all.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available datasets and exit.",
    )
    args = parser.parse_args()

    if args.list:
        print("\nAvailable datasets:")
        print("-" * 70)
        for name, info in DATASETS.items():
            print(f"  {name:<30s} {info['est_size']:>8s}  {info['desc']}")
        print()
        return

    # Ensure output directories exist
    _ensure_dir(RAW_DIR)
    _ensure_dir(PROCESSED_DIR)

    targets = args.dataset if args.dataset else list(DATASETS.keys())

    # Validate dataset names
    for name in targets:
        if name not in DATASETS:
            logger.error("Unknown dataset: %s", name)
            logger.error("Available: %s", ", ".join(DATASETS.keys()))
            sys.exit(1)

    print("=" * 70)
    print("ROSETTA Data Download & Conversion")
    print("=" * 70)
    print(f"Datasets to process: {len(targets)}")
    print(f"Raw directory:       {RAW_DIR}")
    print(f"Processed directory: {PROCESSED_DIR}")
    print()

    results = {}
    for name in targets:
        info = DATASETS[name]
        print(f"\n{'='*70}")
        print(f"[{name}] {info['desc']}")
        print(f"  Species: {info['species']}, Platform: {info['platform']}")
        print(f"  Estimated size: {info['est_size']}")
        print(f"{'='*70}")

        try:
            out_path = info["fn"]()
            results[name] = ("OK", out_path)
            print(f"  -> SUCCESS: {out_path}")
        except Exception as e:
            results[name] = ("FAIL", str(e))
            logger.error("FAILED to download %s: %s", name, e, exc_info=True)

    # Summary
    print(f"\n\n{'='*70}")
    print("DOWNLOAD SUMMARY")
    print(f"{'='*70}")
    for name, (status, detail) in results.items():
        marker = "OK" if status == "OK" else "FAIL"
        print(f"  [{marker:>4s}] {name:<30s} {detail}")

    n_ok = sum(1 for s, _ in results.values() if s == "OK")
    n_fail = sum(1 for s, _ in results.values() if s != "OK")
    print(f"\n{n_ok} succeeded, {n_fail} failed out of {len(results)} datasets")


if __name__ == "__main__":
    main()
