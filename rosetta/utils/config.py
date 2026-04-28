"""Dataclass-based configuration hierarchy with YAML loading."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from omegaconf import OmegaConf


@dataclass
class GraphConfig:
    k_spatial: int = 6
    k_expression: int = 3
    sigma: Optional[float] = None  # auto-compute from median NN distance
    expression_threshold: float = 0.5


@dataclass
class SpatialGNNConfig:
    input_dim: int = 2000
    hidden_dim: int = 256
    num_layers: int = 3
    num_heads: int = 4
    dropout: float = 0.1
    spatial_dim: int = 2


@dataclass
class MultiScaleConfig:
    pool_ratios: list[float] = field(default_factory=lambda: [0.25, 0.25])
    embed_dim: int = 256
    num_gnn_layers_per_level: int = 2


@dataclass
class CHISELConfig:
    graph: GraphConfig = field(default_factory=GraphConfig)
    spatial_gnn: SpatialGNNConfig = field(default_factory=SpatialGNNConfig)
    multi_scale: MultiScaleConfig = field(default_factory=MultiScaleConfig)


@dataclass
class PreprocessingConfig:
    min_genes: int = 200
    min_cells: int = 3
    n_top_genes: int = 2000
    target_sum: float = 10000.0


@dataclass
class DataConfig:
    species: str = "mouse"
    platform: str = "visium"


@dataclass
class TrainingConfig:
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    max_epochs: int = 100
    batch_size: int = 1
    num_workers: int = 4
    mask_ratio: float = 0.2
    lambda_contrast: float = 0.1
    lambda_link: float = 1.0
    lambda_entropy: float = 1.0
    contrastive_temperature: float = 0.1
    contrastive_n_negatives: int = 256
    max_nodes: int = 5000
    val_fraction: float = 0.1
    lr_scheduler: str = "cosine"
    warmup_epochs: int = 5
    gradient_clip_val: float = 1.0


@dataclass
class BRIDGEConfig:
    alpha: float = 0.3  # FGW trade-off (0=feature only, 1=structure only)
    epsilon: float = 0.005  # Entropic regularization (lowered for sharper plans)
    fgw_max_iter: int = 20  # FGW outer iterations (converges early with log-domain)
    sinkhorn_max_iter: int = 50  # Inner Sinkhorn iterations for log-domain
    lambda_fgw: float = 1.0  # FGW alignment loss weight
    lambda_cross_contrastive: float = 0.5
    cross_contrastive_top_k: int = 5  # Top-k matches from transport plan
    cross_contrastive_temperature: float = 0.1
    n_alternating_iters: int = 5  # Alternating FGW <-> fine-tune iterations
    max_nodes_per_species: int = 3000  # Subsample for FGW memory
    finetune_epochs_per_iter: int = 10
    finetune_lr: float = 1e-4
    projection_hidden_dim: int = 128  # Projection head hidden dim
    freeze_encoder_epochs: int = 10  # Epochs to keep encoders frozen
    lambda_mmd: float = 10.0  # MMD distribution matching loss weight


@dataclass
class COMPASSConfig:
    k_values: list[int] = field(default_factory=lambda: [5, 8, 10, 15, 20])
    nmf_max_iter: int = 500
    nmf_n_runs: int = 10
    conservation_k: int = 10
    min_gene_weight: float = 0.01


@dataclass
class RosettaConfig:
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    chisel: CHISELConfig = field(default_factory=CHISELConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    bridge: BRIDGEConfig = field(default_factory=BRIDGEConfig)
    compass: COMPASSConfig = field(default_factory=COMPASSConfig)


def load_config(path: str | Path) -> RosettaConfig:
    """Load a YAML config file into a RosettaConfig dataclass."""
    raw = OmegaConf.load(path)
    raw_dict = OmegaConf.to_container(raw, resolve=True)

    # Map flat YAML keys into nested CHISELConfig structure
    cfg_dict = {}

    if "preprocessing" in raw_dict:
        cfg_dict["preprocessing"] = raw_dict["preprocessing"]

    chisel_dict = {}
    if "graph" in raw_dict:
        chisel_dict["graph"] = raw_dict["graph"]
    if "spatial_gnn" in raw_dict:
        chisel_dict["spatial_gnn"] = raw_dict["spatial_gnn"]
    if "multi_scale" in raw_dict:
        chisel_dict["multi_scale"] = raw_dict["multi_scale"]
    if chisel_dict:
        cfg_dict["chisel"] = chisel_dict

    if "data" in raw_dict:
        cfg_dict["data"] = raw_dict["data"]
    if "training" in raw_dict:
        cfg_dict["training"] = raw_dict["training"]
    if "bridge" in raw_dict:
        cfg_dict["bridge"] = raw_dict["bridge"]
    if "compass" in raw_dict:
        cfg_dict["compass"] = raw_dict["compass"]

    schema = OmegaConf.structured(RosettaConfig)
    merged = OmegaConf.merge(schema, OmegaConf.create(cfg_dict))
    return OmegaConf.to_object(merged)
