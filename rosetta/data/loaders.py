"""Data loading utilities for spatial transcriptomics datasets."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import anndata as ad
import numpy as np
import scipy.sparse as sp
from anndata import AnnData
from numpy.typing import NDArray

from rosetta.data.preprocessing import preprocess_pipeline
from rosetta.utils.config import PreprocessingConfig

logger = logging.getLogger(__name__)


@dataclass
class SpatialDataset:
    """Container for a spatial transcriptomics dataset."""

    adata: AnnData
    spatial_coords: NDArray  # (n_cells, 2)
    platform: str  # e.g. "visium", "merfish", "slide-seq"


def _extract_spatial_coords(adata: AnnData) -> NDArray:
    """Extract and validate spatial coordinates from an AnnData object."""
    if "spatial" not in adata.obsm:
        raise ValueError(
            "No spatial coordinates found. Expected adata.obsm['spatial']."
        )

    spatial_coords = np.array(adata.obsm["spatial"], dtype=np.float32)
    if spatial_coords.ndim != 2 or spatial_coords.shape[1] < 2:
        raise ValueError(
            f"Spatial coordinates have unexpected shape {spatial_coords.shape}. "
            "Expected (n_cells, 2) or (n_cells, 3)."
        )
    # Take first 2 dims if 3D
    return spatial_coords[:, :2]


def load_h5ad(path: str | Path, platform: str = "visium") -> SpatialDataset:
    """Load a .h5ad file and extract spatial coordinates.

    Expects spatial coordinates in adata.obsm["spatial"].
    """
    path = Path(path)
    adata = ad.read_h5ad(path)
    spatial_coords = _extract_spatial_coords(adata)

    return SpatialDataset(adata=adata, spatial_coords=spatial_coords, platform=platform)


def load_visium(path: str | Path) -> SpatialDataset:
    """Load a 10x Visium dataset from Space Ranger output directory.

    Expects the standard Space Ranger output structure:
        path/
        ├── filtered_feature_bc_matrix.h5 (or raw_feature_bc_matrix.h5)
        └── spatial/
            ├── tissue_positions_list.csv (or tissue_positions.csv)
            ├── scalefactors_json.json
            └── tissue_hires_image.png

    Alternatively, accepts a single .h5ad file path.
    """
    import scanpy as sc

    path = Path(path)

    # If path is an h5ad file, load directly
    if path.suffix == ".h5ad":
        return load_h5ad(path, platform="visium")

    # If path is an h5 file, assume it's the matrix file and parent is the dir
    if path.suffix == ".h5":
        spaceranger_dir = path.parent
    else:
        spaceranger_dir = path

    # Try to find the filtered matrix
    h5_candidates = [
        spaceranger_dir / "filtered_feature_bc_matrix.h5",
        spaceranger_dir / "raw_feature_bc_matrix.h5",
    ]
    h5_file = None
    for candidate in h5_candidates:
        if candidate.exists():
            h5_file = candidate
            break

    if h5_file is None:
        # Maybe the directory itself is the Space Ranger output with outs/
        outs_dir = spaceranger_dir / "outs"
        if outs_dir.exists():
            return load_visium(outs_dir)
        raise FileNotFoundError(
            f"Could not find filtered_feature_bc_matrix.h5 in {spaceranger_dir}. "
            "Expected a 10x Space Ranger output directory."
        )

    logger.info("Loading Visium data from %s", spaceranger_dir)
    adata = sc.read_visium(spaceranger_dir)
    adata.var_names_make_unique()

    spatial_coords = _extract_spatial_coords(adata)
    return SpatialDataset(adata=adata, spatial_coords=spatial_coords, platform="visium")


def load_merfish(path: str | Path) -> SpatialDataset:
    """Load a MERFISH dataset.

    Supports:
    1. Pre-built .h5ad file with obsm["spatial"]
    2. Directory with cell_metadata.csv and cell_by_gene.csv
    3. Directory with parquet files (Allen Brain Cell Atlas format)
    """
    path = Path(path)

    # Case 1: h5ad file
    if path.suffix == ".h5ad":
        return load_h5ad(path, platform="merfish")

    # Case 2: Directory with CSV files
    if path.is_dir():
        meta_csv = path / "cell_metadata.csv"
        expr_csv = path / "cell_by_gene.csv"

        if meta_csv.exists() and expr_csv.exists():
            import pandas as pd

            logger.info("Loading MERFISH from CSV files in %s", path)
            meta = pd.read_csv(meta_csv, index_col=0)
            expr = pd.read_csv(expr_csv, index_col=0)

            # Align cell indices
            common_cells = meta.index.intersection(expr.index)
            meta = meta.loc[common_cells]
            expr = expr.loc[common_cells]

            adata = AnnData(
                X=sp.csr_matrix(expr.values.astype(np.float32)),
                obs=meta,
            )
            adata.var_names = expr.columns.tolist()

            # Extract spatial coords from metadata
            coord_cols = _find_coordinate_columns(meta)
            adata.obsm["spatial"] = meta[coord_cols].values.astype(np.float32)

            spatial_coords = _extract_spatial_coords(adata)
            return SpatialDataset(
                adata=adata, spatial_coords=spatial_coords, platform="merfish"
            )

        # Check for h5ad files in the directory
        h5ad_files = list(path.glob("*.h5ad"))
        if h5ad_files:
            return load_h5ad(h5ad_files[0], platform="merfish")

    raise FileNotFoundError(
        f"Could not load MERFISH data from {path}. "
        "Expected .h5ad file or directory with cell_metadata.csv + cell_by_gene.csv."
    )


def load_stereoseq(path: str | Path) -> SpatialDataset:
    """Load a Stereo-seq dataset.

    Supports:
    1. Pre-built .h5ad file with obsm["spatial"]
    2. GEF/GEM file format (requires stereo-seq tools)
    """
    path = Path(path)

    # Case 1: h5ad file
    if path.suffix == ".h5ad":
        dataset = load_h5ad(path, platform="stereoseq")

        # Stereo-seq may store coords differently
        if "spatial" not in dataset.adata.obsm:
            # Try to reconstruct from obs columns
            coord_cols = _find_coordinate_columns(dataset.adata.obs)
            if coord_cols:
                dataset.adata.obsm["spatial"] = (
                    dataset.adata.obs[coord_cols].values.astype(np.float32)
                )
                dataset.spatial_coords = dataset.adata.obsm["spatial"][:, :2]

        return dataset

    # Case 2: Directory — look for h5ad files
    if path.is_dir():
        h5ad_files = list(path.glob("*.h5ad"))
        if h5ad_files:
            return load_stereoseq(h5ad_files[0])

    raise FileNotFoundError(
        f"Could not load Stereo-seq data from {path}. Expected .h5ad file."
    )


def _find_coordinate_columns(df) -> list[str]:
    """Find spatial coordinate columns in a DataFrame.

    Checks for common column naming conventions across platforms.
    """
    # Common naming patterns for x, y coordinates
    x_candidates = ["x", "x_centroid", "center_x", "x_um", "spatial_x", "X"]
    y_candidates = ["y", "y_centroid", "center_y", "y_um", "spatial_y", "Y"]

    x_col = None
    y_col = None
    cols = list(df.columns)

    for xc in x_candidates:
        if xc in cols:
            x_col = xc
            break
    for yc in y_candidates:
        if yc in cols:
            y_col = yc
            break

    if x_col is None or y_col is None:
        raise ValueError(
            f"Could not find spatial coordinate columns. "
            f"Available columns: {cols[:20]}"
        )

    return [x_col, y_col]


def load_and_preprocess(
    path: str | Path,
    platform: str = "visium",
    config: PreprocessingConfig | None = None,
) -> SpatialDataset:
    """Load a .h5ad file, preprocess, and return a SpatialDataset."""
    dataset = load_h5ad(path, platform)
    dataset.adata = preprocess_pipeline(dataset.adata, config)
    # Update spatial coords to match filtered cells
    dataset.spatial_coords = np.array(
        dataset.adata.obsm["spatial"][:, :2], dtype=np.float32
    )
    return dataset
