"""PyTorch Lightning training module for CHISEL.

Provides CHISELLitModule (wraps encoder + decoder + losses) and
SpatialGraphDataModule (loads h5ad, builds graph, handles splits).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pytorch_lightning as pl
import scipy.sparse as sp
import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from rosetta.chisel.encoders import CHISELEncoder
from rosetta.chisel.losses import GeneMasker, MaskedExpressionLoss, SpatialContrastiveLoss
from rosetta.utils.config import (
    CHISELConfig,
    MultiScaleConfig,
    PreprocessingConfig,
    SpatialGNNConfig,
    TrainingConfig,
)
from rosetta.utils.metrics import morans_i

logger = logging.getLogger(__name__)


class CHISELLitModule(pl.LightningModule):
    """Lightning module for self-supervised CHISEL training.

    Wraps CHISELEncoder + mask decoder MLP + all loss functions.
    Training uses masked gene expression prediction as the primary
    objective, spatial contrastive learning as secondary, and DiffPool
    regularization losses.

    Args:
        chisel_config: Architecture configuration.
        training_config: Training hyperparameters.
        input_dim: Number of input genes (set dynamically from data).
        max_nodes: Max nodes for DiffPool sizing.
    """

    def __init__(
        self,
        chisel_config: CHISELConfig,
        training_config: TrainingConfig,
        input_dim: int,
        max_nodes: int,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["chisel_config", "training_config"])
        self.chisel_config = chisel_config
        self.training_config = training_config

        # Override input_dim from data
        chisel_config.spatial_gnn.input_dim = input_dim
        self.input_dim = input_dim

        # Encoder
        self.encoder = CHISELEncoder(chisel_config, max_nodes=max_nodes)

        # Mask decoder: z_spot (embed_dim) -> reconstructed expression (input_dim)
        embed_dim = chisel_config.multi_scale.embed_dim
        hidden_dim = chisel_config.spatial_gnn.hidden_dim
        self.mask_decoder = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, input_dim),
        )

        # Loss functions
        self.gene_masker = GeneMasker(
            mask_ratio=training_config.mask_ratio,
            mask_nonzero_only=True,
        )
        self.masked_expr_loss = MaskedExpressionLoss()
        self.contrastive_loss = SpatialContrastiveLoss(
            temperature=training_config.contrastive_temperature,
            n_negatives=training_config.contrastive_n_negatives,
        )

    def forward(self, data: Data) -> dict[str, Tensor]:
        """Forward pass through encoder."""
        return self.encoder(data.x, data.edge_index, data.edge_attr)

    def _shared_step(self, data: Data, node_mask: Tensor | None) -> dict[str, Tensor]:
        """Shared logic for training and validation steps."""
        # Mask genes
        original_x = data.x
        masked_x, gene_mask = self.gene_masker(original_x)

        # Encode masked input
        out = self.encoder(masked_x, data.edge_index, data.edge_attr)

        # Decode to reconstruct expression
        pred_expr = self.mask_decoder(out["z_spot"])

        # Masked expression loss (primary)
        loss_mask = self.masked_expr_loss(pred_expr, original_x, gene_mask, node_mask)

        # Spatial contrastive loss (secondary)
        loss_contrast = self.contrastive_loss(
            out["z_spot"], data.edge_index, node_mask
        )

        # DiffPool regularization
        loss_link = out["link_loss"]
        loss_entropy = out["entropy_loss"]

        # Total loss
        tc = self.training_config
        total_loss = (
            loss_mask
            + tc.lambda_contrast * loss_contrast
            + tc.lambda_link * loss_link
            + tc.lambda_entropy * loss_entropy
        )

        return {
            "loss": total_loss,
            "loss_mask": loss_mask.detach(),
            "loss_contrast": loss_contrast.detach(),
            "loss_link": loss_link.detach(),
            "loss_entropy": loss_entropy.detach(),
            "z_spot": out["z_spot"].detach(),
        }

    def training_step(self, batch: Data, batch_idx: int) -> Tensor:
        train_mask = batch.train_mask if hasattr(batch, "train_mask") else None
        result = self._shared_step(batch, train_mask)

        self.log("train/loss", result["loss"], prog_bar=True)
        self.log("train/loss_mask", result["loss_mask"])
        self.log("train/loss_contrast", result["loss_contrast"])
        self.log("train/loss_link", result["loss_link"])
        self.log("train/loss_entropy", result["loss_entropy"])

        return result["loss"]

    def validation_step(self, batch: Data, batch_idx: int) -> dict[str, Tensor]:
        val_mask = batch.val_mask if hasattr(batch, "val_mask") else None
        result = self._shared_step(batch, val_mask)

        self.log("val/loss", result["loss"], prog_bar=True)
        self.log("val/loss_mask", result["loss_mask"])
        self.log("val/loss_contrast", result["loss_contrast"])

        # Compute Moran's I on validation embeddings
        if hasattr(batch, "pos") and batch.pos is not None:
            z_np = result["z_spot"].cpu().numpy()
            pos_np = batch.pos.cpu().numpy()
            mi = morans_i(z_np, pos_np, k=min(6, z_np.shape[0] - 1))
            self.log("val/morans_i", mi, prog_bar=True)

        return result

    def configure_optimizers(self) -> dict[str, Any]:
        tc = self.training_config
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=tc.learning_rate,
            weight_decay=tc.weight_decay,
        )

        config: dict[str, Any] = {"optimizer": optimizer}

        if tc.lr_scheduler == "cosine":
            # Total steps for cosine schedule (after warmup)
            total_epochs = tc.max_epochs
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max(1, total_epochs - tc.warmup_epochs)
            )

            if tc.warmup_epochs > 0:
                warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                    optimizer,
                    start_factor=0.01,
                    end_factor=1.0,
                    total_iters=tc.warmup_epochs,
                )
                scheduler = torch.optim.lr_scheduler.SequentialLR(
                    optimizer,
                    schedulers=[warmup_scheduler, scheduler],
                    milestones=[tc.warmup_epochs],
                )

            config["lr_scheduler"] = {
                "scheduler": scheduler,
                "interval": "epoch",
            }

        return config


class SpatialGraphDataModule(pl.LightningDataModule):
    """Lightning data module for loading and preparing spatial graph data.

    Loads h5ad -> preprocesses -> builds spatial graph -> train/val split.
    The full graph is used for message passing; loss is computed only on
    the split's nodes.

    Args:
        h5ad_path: Path to h5ad file.
        dataset_config: Dict with preprocessing and graph config overrides.
        training_config: Training hyperparameters.
        max_nodes: Maximum nodes (subsample if larger).
    """

    def __init__(
        self,
        h5ad_path: str | Path,
        dataset_config: dict,
        training_config: TrainingConfig,
        max_nodes: int = 5000,
    ):
        super().__init__()
        self.h5ad_path = Path(h5ad_path)
        self.dataset_config = dataset_config
        self.training_config = training_config
        self.max_nodes = max_nodes
        self._graph_data: Data | None = None
        self._input_dim: int | None = None

    @property
    def input_dim(self) -> int:
        """Number of input genes (available after setup)."""
        if self._input_dim is None:
            raise RuntimeError("Call setup() before accessing input_dim")
        return self._input_dim

    @property
    def n_nodes(self) -> int:
        """Number of nodes in the graph (available after setup)."""
        if self._graph_data is None:
            raise RuntimeError("Call setup() before accessing n_nodes")
        return self._graph_data.x.shape[0]

    def setup(self, stage: str | None = None) -> None:
        if self._graph_data is not None:
            return

        import anndata as ad

        from rosetta.chisel.graph_construction import build_spatial_graph

        # Load
        logger.info("Loading %s", self.h5ad_path)
        adata = ad.read_h5ad(self.h5ad_path)

        # Preprocess (uses same logic as validate_real_data.py)
        adata = self._preprocess(adata)

        # Subsample if needed
        if adata.n_obs > self.max_nodes:
            logger.info("Subsampling %d -> %d nodes", adata.n_obs, self.max_nodes)
            rng = np.random.default_rng(42)
            idx = rng.choice(adata.n_obs, size=self.max_nodes, replace=False)
            idx.sort()
            adata = adata[idx].copy()

        # Build graph
        spatial_coords = torch.tensor(
            np.array(adata.obsm["spatial"][:, :2], dtype=np.float32)
        )
        X = adata.X
        if sp.issparse(X):
            X = X.toarray()
        expression = torch.tensor(np.array(X, dtype=np.float32))

        cfg = self.dataset_config
        data = build_spatial_graph(
            spatial_coords=spatial_coords,
            expression=expression,
            k_spatial=cfg.get("graph_k_spatial", 6),
            k_expression=cfg.get("graph_k_expression", 3),
            expression_threshold=0.3,
        )

        # Train/val node split
        n = data.x.shape[0]
        n_val = max(1, int(n * self.training_config.val_fraction))
        perm = torch.randperm(n)
        val_idx = perm[:n_val]
        train_idx = perm[n_val:]

        train_mask = torch.zeros(n, dtype=torch.bool)
        val_mask = torch.zeros(n, dtype=torch.bool)
        train_mask[train_idx] = True
        val_mask[val_idx] = True

        data.train_mask = train_mask
        data.val_mask = val_mask

        self._graph_data = data
        self._input_dim = data.x.shape[1]

        logger.info(
            "Graph: %d nodes, %d edges, %d genes, train=%d, val=%d",
            n, data.edge_index.shape[1], self._input_dim,
            train_mask.sum().item(), val_mask.sum().item(),
        )

    def _preprocess(self, adata):
        """Preprocess with dataset-specific config."""
        import scanpy as sc

        adata = adata.copy()
        adata.layers["counts"] = adata.X.copy()

        cfg = self.dataset_config
        pp = cfg.get("preprocessing", PreprocessingConfig())

        sc.pp.filter_cells(adata, min_genes=pp.min_genes)
        sc.pp.filter_genes(adata, min_cells=pp.min_cells)

        if adata.n_obs == 0 or adata.n_vars == 0:
            raise ValueError("All cells or genes filtered out during preprocessing")

        if not cfg.get("skip_normalize", False):
            sc.pp.normalize_total(adata, target_sum=pp.target_sum)
            sc.pp.log1p(adata)

        if not cfg.get("skip_hvg", False) and adata.n_vars > pp.n_top_genes:
            try:
                adata_hvg = adata.copy()
                adata_hvg.X = adata_hvg.layers["counts"].copy()
                sc.pp.highly_variable_genes(
                    adata_hvg, n_top_genes=pp.n_top_genes, flavor="seurat_v3",
                )
                adata.var["highly_variable"] = adata_hvg.var["highly_variable"]
                adata = adata[:, adata.var["highly_variable"]].copy()
            except Exception as e:
                logger.warning("HVG selection failed (%s), using top %d by variance", e, pp.n_top_genes)
                X = adata.X.toarray() if sp.issparse(adata.X) else np.array(adata.X)
                var = np.var(X, axis=0)
                top_idx = np.argsort(var)[-pp.n_top_genes:]
                adata = adata[:, top_idx].copy()

        return adata

    def train_dataloader(self) -> DataLoader:
        return DataLoader([self._graph_data], batch_size=1, shuffle=False)

    def val_dataloader(self) -> DataLoader:
        return DataLoader([self._graph_data], batch_size=1, shuffle=False)
