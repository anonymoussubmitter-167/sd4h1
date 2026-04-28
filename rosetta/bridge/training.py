"""BRIDGE training module for cross-species alignment.

Alternating optimization:
1. FGW step (no gradient): Compute transport plan T from current embeddings
2. Fine-tune step (gradient): Update CHISEL encoders using cross-species losses
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
from torch.utils.data import DataLoader as TorchDataLoader
from torch_geometric.data import Data

from rosetta.bridge.alignment import FGWAligner, subsample_for_fgw
from rosetta.bridge.losses import CrossSpeciesContrastiveLoss, FGWAlignmentLoss, MMDLoss
from rosetta.bridge.shared_space import build_shared_gene_space
from rosetta.chisel.encoders import CHISELEncoder
from rosetta.chisel.graph_construction import build_spatial_graph
from rosetta.chisel.losses import GeneMasker, MaskedExpressionLoss, SpatialContrastiveLoss
from rosetta.data.ortholog_db import OrthologMapping, load_ortholog_mapping
from rosetta.utils.config import BRIDGEConfig, CHISELConfig, TrainingConfig

logger = logging.getLogger(__name__)


class SharedProjectionHead(nn.Module):
    """Shared MLP projection head that maps both species into a comparable space.

    Same weights are applied to both source and target embeddings, forcing
    them into a shared representation space where cross-species distances
    are meaningful.
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        if hidden_dim <= 0:
            # Identity projection (ablation: no projection head)
            self.net = nn.Identity()
        else:
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, output_dim),
            )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class BRIDGELitModule(pl.LightningModule):
    """Lightning module for cross-species BRIDGE alignment.

    Wraps two CHISEL encoders (source + target species) and alternates
    between FGW transport plan computation and encoder fine-tuning with
    cross-species contrastive losses.

    Args:
        encoder_source: Pretrained CHISEL encoder for source species.
        encoder_target: Pretrained CHISEL encoder for target species.
        bridge_config: BRIDGE hyperparameters.
        training_config: Base training config (for self-supervised losses).
        input_dim_source: Number of input genes for source.
        input_dim_target: Number of input genes for target.
    """

    def __init__(
        self,
        encoder_source: CHISELEncoder,
        encoder_target: CHISELEncoder,
        bridge_config: BRIDGEConfig,
        training_config: TrainingConfig,
        input_dim_source: int,
        input_dim_target: int,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=[
            "encoder_source", "encoder_target",
            "bridge_config", "training_config",
        ])
        self.encoder_source = encoder_source
        self.encoder_target = encoder_target
        self.bridge_config = bridge_config
        self.training_config = training_config
        self.automatic_optimization = False

        # FGW aligner (numpy-based, no gradient)
        self.aligner = FGWAligner(
            alpha=bridge_config.alpha,
            epsilon=bridge_config.epsilon,
            max_iter=bridge_config.fgw_max_iter,
            sinkhorn_max_iter=bridge_config.sinkhorn_max_iter,
        )

        # Shared projection head
        embed_dim = encoder_source.config.multi_scale.embed_dim
        proj_hidden = bridge_config.projection_hidden_dim
        self.projection_head = SharedProjectionHead(embed_dim, proj_hidden, embed_dim)

        # Encoder freezing
        self._freeze_epochs = bridge_config.freeze_encoder_epochs
        if self._freeze_epochs > 0:
            self.encoder_source.requires_grad_(False)
            self.encoder_target.requires_grad_(False)

        # Cross-species losses
        self.cross_contrastive_loss = CrossSpeciesContrastiveLoss(
            top_k=bridge_config.cross_contrastive_top_k,
            temperature=bridge_config.cross_contrastive_temperature,
        )
        self.fgw_alignment_loss = FGWAlignmentLoss()
        self.mmd_loss = MMDLoss()

        # Self-supervised losses (prevent catastrophic forgetting)
        hidden_dim = encoder_source.config.spatial_gnn.hidden_dim
        self.mask_decoder_source = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, input_dim_source),
        )
        self.mask_decoder_target = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, input_dim_target),
        )
        self.gene_masker = GeneMasker(
            mask_ratio=training_config.mask_ratio,
            mask_nonzero_only=True,
        )
        self.masked_expr_loss = MaskedExpressionLoss()
        self.spatial_contrastive_loss = SpatialContrastiveLoss(
            temperature=training_config.contrastive_temperature,
            n_negatives=training_config.contrastive_n_negatives,
        )

        # Current transport plan (updated during FGW steps)
        self._transport_plan: Tensor | None = None
        self._current_iter: int = 0

    def forward(
        self, data_source: Data, data_target: Data
    ) -> dict[str, Tensor]:
        """Forward pass through both encoders."""
        out_s = self.encoder_source(
            data_source.x, data_source.edge_index, data_source.edge_attr
        )
        out_t = self.encoder_target(
            data_target.x, data_target.edge_index, data_target.edge_attr
        )
        z_both = self.projection_head(
            torch.cat([out_s["z_spot"], out_t["z_spot"]], dim=0)
        )
        z_both = torch.nn.functional.normalize(z_both, dim=1)
        n_s = out_s["z_spot"].shape[0]
        out_s["z_spot_proj"] = z_both[:n_s]
        out_t["z_spot_proj"] = z_both[n_s:]
        return {"source": out_s, "target": out_t}

    def on_train_epoch_start(self) -> None:
        """Unfreeze encoders after the freeze period."""
        if self.current_epoch == self._freeze_epochs and self._freeze_epochs > 0:
            self.encoder_source.requires_grad_(True)
            self.encoder_target.requires_grad_(True)
            logger.info(
                "Epoch %d: unfreezing encoders for joint fine-tuning",
                self.current_epoch,
            )

    def training_step(self, batch: dict[str, Data], batch_idx: int) -> None:
        """Single training step with alternating FGW + fine-tune."""
        opt = self.optimizers()
        data_s = batch["source"]
        data_t = batch["target"]

        # Encode both species
        out_s = self.encoder_source(data_s.x, data_s.edge_index, data_s.edge_attr)
        out_t = self.encoder_target(data_t.x, data_t.edge_index, data_t.edge_attr)
        z_s_raw = out_s["z_spot"]
        z_t_raw = out_t["z_spot"]

        # Project into shared space for cross-species losses
        # Joint BN: concatenate so BatchNorm normalizes both species together
        n_s = z_s_raw.shape[0]
        z_both = self.projection_head(torch.cat([z_s_raw, z_t_raw], dim=0))
        z_both = torch.nn.functional.normalize(z_both, dim=1)  # L2 normalize
        z_s, z_t = z_both[:n_s], z_both[n_s:]

        # FGW step: compute transport plan (no gradient, uses projected embeddings)
        with torch.no_grad():
            z_s_np = z_s.detach().cpu().numpy()
            z_t_np = z_t.detach().cpu().numpy()
            coords_s_np = data_s.pos.cpu().numpy()
            coords_t_np = data_t.pos.cpu().numpy()

            # Subsample with epoch-varying RNG so each epoch sees different cells
            max_n = self.bridge_config.max_nodes_per_species
            fgw_rng = np.random.default_rng(42 + self.current_epoch)
            z_s_sub, coords_s_sub, idx_s = subsample_for_fgw(z_s_np, coords_s_np, max_n, rng=fgw_rng)
            z_t_sub, coords_t_sub, idx_t = subsample_for_fgw(z_t_np, coords_t_np, max_n, rng=fgw_rng)

            T_np, fgw_dist = self.aligner.compute_transport_plan(
                z_s_sub, z_t_sub, coords_s_sub, coords_t_sub
            )
            T = torch.tensor(T_np, dtype=torch.float32, device=z_s.device)

        self.log("train/fgw_dist", fgw_dist, prog_bar=True)

        # Log transport plan entropy ratio
        with torch.no_grad():
            T_pos = T.clamp(min=1e-30)
            entropy = -(T_pos * T_pos.log()).sum().item()
            n_s, n_t = T.shape
            max_entropy = np.log(n_s * n_t)
            entropy_ratio = entropy / max_entropy if max_entropy > 0 else 0.0
        self.log("train/transport_entropy_ratio", entropy_ratio)

        # Use subsampled projected embeddings for loss computation
        z_s_sub_t = z_s[torch.tensor(idx_s, device=z_s.device).long()]
        z_t_sub_t = z_t[torch.tensor(idx_t, device=z_t.device).long()]

        # Cross-species contrastive loss (in projected space)
        loss_cross = self.cross_contrastive_loss(z_s_sub_t, z_t_sub_t, T)

        # FGW alignment loss (in projected space)
        loss_fgw = self.fgw_alignment_loss(z_s_sub_t, z_t_sub_t, T)

        # MMD distribution matching loss (forces global overlap of projected embeddings)
        loss_mmd = self.mmd_loss(z_s_sub_t, z_t_sub_t)

        # Self-supervised losses use raw embeddings (forgetting prevention)
        loss_self = self._self_supervised_loss(data_s, out_s, self.mask_decoder_source) + \
                    self._self_supervised_loss(data_t, out_t, self.mask_decoder_target)

        # Total loss
        bc = self.bridge_config
        total_loss = (
            bc.lambda_fgw * loss_fgw
            + bc.lambda_cross_contrastive * loss_cross
            + bc.lambda_mmd * loss_mmd
            + self.training_config.lambda_contrast * loss_self
        )

        opt.zero_grad()
        self.manual_backward(total_loss)
        torch.nn.utils.clip_grad_norm_(self.parameters(), self.training_config.gradient_clip_val)
        opt.step()

        self.log("train/loss", total_loss.detach(), prog_bar=True)
        self.log("train/loss_cross", loss_cross.detach())
        self.log("train/loss_fgw", loss_fgw.detach())
        self.log("train/loss_mmd", loss_mmd.detach())
        self.log("train/loss_self", loss_self.detach())

        self._transport_plan = T.detach()

    def _self_supervised_loss(
        self,
        data: Data,
        encoder_out: dict[str, Tensor],
        mask_decoder: nn.Module,
    ) -> Tensor:
        """Compute self-supervised losses for forgetting prevention.

        Combines spatial contrastive loss with masked expression reconstruction
        to prevent catastrophic forgetting of both spatial structure and gene
        expression modeling learned during CHISEL pretraining.
        """
        z = encoder_out["z_spot"]

        # Spatial contrastive loss
        loss_contrast = self.spatial_contrastive_loss(z, data.edge_index)

        # Masked expression reconstruction loss
        x_expr = data.x  # (N, G) original expression
        _masked_x, mask = self.gene_masker(x_expr)  # unpack (masked_x, bool_mask)
        x_pred = mask_decoder(z)  # (N, G) predicted expression from embeddings
        loss_recon = self.masked_expr_loss(x_pred, x_expr, mask)

        return loss_contrast + loss_recon

    def validation_step(self, batch: dict[str, Data], batch_idx: int) -> None:
        """Validation: compute alignment metrics in projected space."""
        data_s = batch["source"]
        data_t = batch["target"]

        out_s = self.encoder_source(data_s.x, data_s.edge_index, data_s.edge_attr)
        out_t = self.encoder_target(data_t.x, data_t.edge_index, data_t.edge_attr)

        # Alignment score must use the same projected space as training
        # Joint BN + L2 normalize (consistent with training_step)
        z_both = self.projection_head(
            torch.cat([out_s["z_spot"], out_t["z_spot"]], dim=0)
        )
        z_both = torch.nn.functional.normalize(z_both, dim=1)
        n_s = out_s["z_spot"].shape[0]
        z_s = z_both[:n_s].detach().cpu().numpy()
        z_t = z_both[n_s:].detach().cpu().numpy()

        # kNN mixing score
        from rosetta.utils.metrics import alignment_score

        z_combined = np.concatenate([z_s, z_t], axis=0)
        species_labels = np.array([0] * z_s.shape[0] + [1] * z_t.shape[0])
        try:
            mix_score = alignment_score(z_combined, species_labels, k=50)
            self.log("val/alignment_score", mix_score, prog_bar=True)
        except Exception:
            pass

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.bridge_config.finetune_lr,
            weight_decay=self.training_config.weight_decay,
        )

    @property
    def transport_plan(self) -> Tensor | None:
        """Current transport plan from last FGW step."""
        return self._transport_plan


class CrossSpeciesDataModule(pl.LightningDataModule):
    """Data module for loading two species datasets for BRIDGE alignment.

    Loads source and target h5ad files, builds spatial graphs, and
    optionally loads ortholog mappings for shared gene space construction.

    Args:
        source_h5ad: Path to source species h5ad file.
        target_h5ad: Path to target species h5ad file.
        source_config: Dataset-specific config dict for source.
        target_config: Dataset-specific config dict for target.
        bridge_config: BRIDGE hyperparameters.
        training_config: Training config.
        ortholog_map: Pre-loaded ortholog mapping (optional).
    """

    def __init__(
        self,
        source_h5ad: str | Path,
        target_h5ad: str | Path,
        source_config: dict,
        target_config: dict,
        bridge_config: BRIDGEConfig,
        training_config: TrainingConfig,
        ortholog_map: OrthologMapping | None = None,
    ):
        super().__init__()
        self.source_h5ad = Path(source_h5ad)
        self.target_h5ad = Path(target_h5ad)
        self.source_config = source_config
        self.target_config = target_config
        self.bridge_config = bridge_config
        self.training_config = training_config
        self.ortholog_map = ortholog_map

        self._source_data: Data | None = None
        self._target_data: Data | None = None
        self._input_dim_source: int | None = None
        self._input_dim_target: int | None = None

    @property
    def input_dim_source(self) -> int:
        if self._input_dim_source is None:
            raise RuntimeError("Call setup() first")
        return self._input_dim_source

    @property
    def input_dim_target(self) -> int:
        if self._input_dim_target is None:
            raise RuntimeError("Call setup() first")
        return self._input_dim_target

    @property
    def n_nodes_source(self) -> int:
        if self._source_data is None:
            raise RuntimeError("Call setup() first")
        return self._source_data.x.shape[0]

    @property
    def n_nodes_target(self) -> int:
        if self._target_data is None:
            raise RuntimeError("Call setup() first")
        return self._target_data.x.shape[0]

    def setup(self, stage: str | None = None) -> None:
        if self._source_data is not None:
            return

        self._source_data = self._load_and_build_graph(
            self.source_h5ad, self.source_config, "source"
        )
        self._target_data = self._load_and_build_graph(
            self.target_h5ad, self.target_config, "target"
        )
        self._input_dim_source = self._source_data.x.shape[1]
        self._input_dim_target = self._target_data.x.shape[1]

        logger.info(
            "Source: %d nodes, %d genes | Target: %d nodes, %d genes",
            self.n_nodes_source, self._input_dim_source,
            self.n_nodes_target, self._input_dim_target,
        )

    def _load_and_build_graph(
        self, h5ad_path: Path, config: dict, label: str
    ) -> Data:
        """Load h5ad, preprocess, subsample, build spatial graph."""
        import anndata as ad
        import scanpy as sc

        from rosetta.utils.config import PreprocessingConfig

        logger.info("Loading %s: %s", label, h5ad_path)
        adata = ad.read_h5ad(h5ad_path)
        adata = adata.copy()
        adata.layers["counts"] = adata.X.copy()

        pp = config.get("preprocessing", PreprocessingConfig())

        sc.pp.filter_cells(adata, min_genes=pp.min_genes)
        sc.pp.filter_genes(adata, min_cells=pp.min_cells)

        if adata.n_obs == 0 or adata.n_vars == 0:
            raise ValueError(f"All cells/genes filtered for {label}")

        if not config.get("skip_normalize", False):
            sc.pp.normalize_total(adata, target_sum=pp.target_sum)
            sc.pp.log1p(adata)

        if not config.get("skip_hvg", False) and adata.n_vars > pp.n_top_genes:
            try:
                adata_hvg = adata.copy()
                adata_hvg.X = adata_hvg.layers["counts"].copy()
                sc.pp.highly_variable_genes(
                    adata_hvg, n_top_genes=pp.n_top_genes, flavor="seurat_v3",
                )
                adata.var["highly_variable"] = adata_hvg.var["highly_variable"]
                adata = adata[:, adata.var["highly_variable"]].copy()
            except Exception as e:
                logger.warning("HVG failed for %s (%s), using top by variance", label, e)
                X = adata.X.toarray() if sp.issparse(adata.X) else np.array(adata.X)
                var = np.var(X, axis=0)
                top_idx = np.argsort(var)[-pp.n_top_genes:]
                adata = adata[:, top_idx].copy()

        # Subsample
        max_n = self.bridge_config.max_nodes_per_species
        if adata.n_obs > max_n:
            logger.info("Subsampling %s: %d -> %d", label, adata.n_obs, max_n)
            rng = np.random.default_rng(42)
            idx = rng.choice(adata.n_obs, size=max_n, replace=False)
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

        data = build_spatial_graph(
            spatial_coords=spatial_coords,
            expression=expression,
            k_spatial=config.get("graph_k_spatial", 6),
            k_expression=config.get("graph_k_expression", 3),
            expression_threshold=0.3,
        )

        return data

    def train_dataloader(self) -> TorchDataLoader:
        return TorchDataLoader(
            [_PairedData(self._source_data, self._target_data)],
            batch_size=1,
            shuffle=False,
            collate_fn=_paired_collate,
        )

    def val_dataloader(self) -> TorchDataLoader:
        return TorchDataLoader(
            [_PairedData(self._source_data, self._target_data)],
            batch_size=1,
            shuffle=False,
            collate_fn=_paired_collate,
        )


class _PairedData:
    """Simple wrapper to hold paired source/target graph data."""

    def __init__(self, source: Data, target: Data):
        self.source = source
        self.target = target


def _paired_collate(batch: list[_PairedData]) -> dict[str, Data]:
    """Collate function for paired data."""
    item = batch[0]
    return {"source": item.source, "target": item.target}
