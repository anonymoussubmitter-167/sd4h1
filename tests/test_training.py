"""Integration tests for CHISEL training module."""

import pytest
import torch
from torch_geometric.data import Data

from rosetta.chisel.training import CHISELLitModule
from rosetta.utils.config import (
    CHISELConfig,
    MultiScaleConfig,
    SpatialGNNConfig,
    TrainingConfig,
)


def _make_test_graph(n=100, n_genes=50, k=6):
    """Create a synthetic spatial graph for testing."""
    # Random spatial coordinates
    pos = torch.randn(n, 2) * 10

    # Expression (some sparse pattern)
    x = torch.randn(n, n_genes).abs()

    # KNN-like edges
    src_list = []
    dst_list = []
    for i in range(n):
        dists = torch.norm(pos - pos[i], dim=1)
        _, idx = torch.topk(dists, k + 1, largest=False)
        neighbors = idx[1:]  # exclude self
        src_list.extend([i] * k)
        dst_list.extend(neighbors.tolist())

    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)

    # Edge attributes: [weight, distance, rel_dx, rel_dy, edge_type]
    diff = pos[edge_index[1]] - pos[edge_index[0]]
    distances = torch.norm(diff, dim=1)
    sigma = torch.median(distances).item()
    weights = torch.exp(-distances**2 / (2 * sigma**2))
    edge_type = torch.zeros(edge_index.shape[1])
    edge_attr = torch.stack([weights, distances, diff[:, 0], diff[:, 1], edge_type], dim=1)

    # Train/val masks
    perm = torch.randperm(n)
    n_val = max(1, int(n * 0.1))
    train_mask = torch.zeros(n, dtype=torch.bool)
    val_mask = torch.zeros(n, dtype=torch.bool)
    train_mask[perm[n_val:]] = True
    val_mask[perm[:n_val]] = True

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, pos=pos)
    data.train_mask = train_mask
    data.val_mask = val_mask

    return data


def _make_model(n_genes=50, n_nodes=100, hidden_dim=32, embed_dim=32):
    """Create a small CHISEL model for testing."""
    chisel_config = CHISELConfig(
        spatial_gnn=SpatialGNNConfig(
            input_dim=n_genes,
            hidden_dim=hidden_dim,
            num_layers=2,
            num_heads=4,
            dropout=0.1,
        ),
        multi_scale=MultiScaleConfig(
            pool_ratios=[0.25, 0.25],
            embed_dim=embed_dim,
            num_gnn_layers_per_level=2,
        ),
    )

    training_config = TrainingConfig(
        learning_rate=1e-3,
        max_epochs=10,
        mask_ratio=0.2,
        lambda_contrast=0.1,
        lambda_link=1.0,
        lambda_entropy=1.0,
    )

    model = CHISELLitModule(
        chisel_config=chisel_config,
        training_config=training_config,
        input_dim=n_genes,
        max_nodes=n_nodes,
    )
    return model


class TestCHISELLitModule:
    """Test the Lightning training module."""

    def test_training_step_runs(self):
        """Training step should run without errors."""
        data = _make_test_graph(n=50, n_genes=30, k=4)
        model = _make_model(n_genes=30, n_nodes=50)

        loss = model.training_step(data, 0)
        assert loss.item() > 0
        assert torch.isfinite(loss)

    def test_validation_step_runs(self):
        """Validation step should run without errors."""
        data = _make_test_graph(n=50, n_genes=30, k=4)
        model = _make_model(n_genes=30, n_nodes=50)

        result = model.validation_step(data, 0)
        assert "loss" in result
        assert result["loss"].item() > 0

    def test_loss_decreases_over_steps(self):
        """Loss should decrease over multiple optimization steps."""
        data = _make_test_graph(n=80, n_genes=30, k=4)
        model = _make_model(n_genes=30, n_nodes=80, hidden_dim=32, embed_dim=32)

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        losses = []
        for _ in range(10):
            optimizer.zero_grad()
            loss = model.training_step(data, 0)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        # Loss should decrease from first to last
        assert losses[-1] < losses[0], (
            f"Loss should decrease: first={losses[0]:.4f}, last={losses[-1]:.4f}"
        )

    def test_configure_optimizers(self):
        """Optimizer should be configured correctly."""
        model = _make_model()
        opt_config = model.configure_optimizers()

        assert "optimizer" in opt_config
        assert isinstance(opt_config["optimizer"], torch.optim.AdamW)

        # Should have lr scheduler with cosine config
        assert "lr_scheduler" in opt_config
        assert "scheduler" in opt_config["lr_scheduler"]

    def test_gradient_flow_to_all_components(self):
        """Gradients should flow to encoder, decoder, and all loss paths."""
        data = _make_test_graph(n=50, n_genes=30, k=4)
        model = _make_model(n_genes=30, n_nodes=50)

        loss = model.training_step(data, 0)
        loss.backward()

        # Check encoder has gradients
        for name, param in model.encoder.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for encoder.{name}"
                break

        # Check decoder has gradients
        for name, param in model.mask_decoder.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for mask_decoder.{name}"
                break

    def test_forward_returns_dict(self):
        """Forward pass should return expected dict keys."""
        data = _make_test_graph(n=50, n_genes=30, k=4)
        model = _make_model(n_genes=30, n_nodes=50)

        with torch.no_grad():
            out = model(data)

        assert "z_spot" in out
        assert "z_niche" in out
        assert "z_region" in out
        assert out["z_spot"].shape[0] == 50

    def test_no_nan_in_outputs(self):
        """No NaN values should appear in any output."""
        data = _make_test_graph(n=50, n_genes=30, k=4)
        model = _make_model(n_genes=30, n_nodes=50)

        with torch.no_grad():
            out = model(data)

        for key in ["z_spot", "z_niche", "z_region"]:
            assert not torch.isnan(out[key]).any(), f"NaN in {key}"

    def test_mask_decoder_output_shape(self):
        """Mask decoder should output (N, input_dim)."""
        data = _make_test_graph(n=50, n_genes=30, k=4)
        model = _make_model(n_genes=30, n_nodes=50, embed_dim=32)

        with torch.no_grad():
            out = model(data)
            pred = model.mask_decoder(out["z_spot"])

        assert pred.shape == (50, 30)
