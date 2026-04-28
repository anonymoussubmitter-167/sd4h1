"""Visualization utilities for spatial transcriptomics data."""

from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from sklearn.decomposition import PCA


def plot_spatial(
    coords: NDArray,
    values: NDArray | None = None,
    title: str = "",
    cmap: str = "viridis",
    point_size: float = 5.0,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Scatter plot of spatial coordinates colored by values."""
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(8, 8))

    scatter_kwargs = dict(s=point_size, cmap=cmap)
    if values is not None:
        scatter_kwargs["c"] = values
    else:
        scatter_kwargs["c"] = "steelblue"

    sc = ax.scatter(coords[:, 0], coords[:, 1], **scatter_kwargs)

    if values is not None:
        plt.colorbar(sc, ax=ax, shrink=0.6)

    ax.set_title(title)
    ax.set_aspect("equal")
    ax.invert_yaxis()
    return ax


def plot_embeddings_umap(
    embeddings: NDArray,
    spatial_coords: NDArray,
    labels: NDArray | None = None,
    title: str = "",
    point_size: float = 5.0,
) -> plt.Figure:
    """2-panel UMAP: colored by spatial position and by labels.

    Args:
        embeddings: (N, D) embedding matrix.
        spatial_coords: (N, 2) spatial coordinates (used for coloring).
        labels: (N,) optional labels for coloring second panel.
        title: Figure title.
        point_size: Scatter point size.

    Returns:
        matplotlib Figure.
    """
    from umap import UMAP

    reducer = UMAP(n_components=2, random_state=42)
    umap_coords = reducer.fit_transform(embeddings)

    n_panels = 2 if labels is not None else 1
    fig, axes = plt.subplots(1, n_panels, figsize=(8 * n_panels, 7))
    if n_panels == 1:
        axes = [axes]

    # Panel 1: colored by spatial position (use X coordinate)
    spatial_color = spatial_coords[:, 0]
    sc0 = axes[0].scatter(
        umap_coords[:, 0], umap_coords[:, 1],
        c=spatial_color, cmap="viridis", s=point_size,
    )
    plt.colorbar(sc0, ax=axes[0], shrink=0.6, label="Spatial X")
    axes[0].set_title(f"{title} — UMAP (spatial position)")
    axes[0].set_xlabel("UMAP 1")
    axes[0].set_ylabel("UMAP 2")

    # Panel 2: colored by labels
    if labels is not None:
        unique_labels = np.unique(labels)
        if np.issubdtype(labels.dtype, np.number) and len(unique_labels) > 20:
            sc1 = axes[1].scatter(
                umap_coords[:, 0], umap_coords[:, 1],
                c=labels, cmap="tab20", s=point_size,
            )
            plt.colorbar(sc1, ax=axes[1], shrink=0.6, label="Label")
        else:
            for i, lbl in enumerate(unique_labels):
                mask = labels == lbl
                axes[1].scatter(
                    umap_coords[mask, 0], umap_coords[mask, 1],
                    s=point_size, label=str(lbl), alpha=0.7,
                )
            axes[1].legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=7)
        axes[1].set_title(f"{title} — UMAP (labels)")
        axes[1].set_xlabel("UMAP 1")
        axes[1].set_ylabel("UMAP 2")

    fig.tight_layout()
    return fig


def plot_spatial_embeddings(
    spatial_coords: NDArray,
    embeddings: NDArray,
    n_components: int = 3,
    title: str = "",
    point_size: float = 5.0,
) -> plt.Figure:
    """Spatial map colored by top PCA components of embeddings.

    Args:
        spatial_coords: (N, 2) spatial coordinates.
        embeddings: (N, D) embedding matrix.
        n_components: Number of PCA components to visualize.
        title: Figure title.
        point_size: Scatter point size.

    Returns:
        matplotlib Figure.
    """
    pca = PCA(n_components=n_components)
    pca_emb = pca.fit_transform(embeddings)

    fig, axes = plt.subplots(1, n_components, figsize=(7 * n_components, 6))
    if n_components == 1:
        axes = [axes]

    for i in range(n_components):
        sc = axes[i].scatter(
            spatial_coords[:, 0], spatial_coords[:, 1],
            c=pca_emb[:, i], cmap="RdBu_r", s=point_size,
        )
        plt.colorbar(sc, ax=axes[i], shrink=0.6)
        var_pct = pca.explained_variance_ratio_[i] * 100
        axes[i].set_title(f"PC{i+1} ({var_pct:.1f}% var)")
        axes[i].set_aspect("equal")
        axes[i].invert_yaxis()

    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    return fig


def plot_training_curves(
    metrics: dict[str, list[float]],
    title: str = "Training Curves",
) -> plt.Figure:
    """Plot loss and metric curves over epochs.

    Args:
        metrics: Dict mapping metric name to list of values per epoch.
        title: Figure title.

    Returns:
        matplotlib Figure.
    """
    # Separate losses from other metrics
    loss_keys = [k for k in metrics if "loss" in k.lower()]
    other_keys = [k for k in metrics if "loss" not in k.lower()]

    n_panels = (1 if loss_keys else 0) + (1 if other_keys else 0)
    if n_panels == 0:
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        ax.text(0.5, 0.5, "No metrics to plot", ha="center", va="center")
        return fig

    fig, axes = plt.subplots(1, n_panels, figsize=(7 * n_panels, 5))
    if n_panels == 1:
        axes = [axes]

    panel_idx = 0

    if loss_keys:
        ax = axes[panel_idx]
        for key in loss_keys:
            vals = metrics[key]
            ax.plot(range(len(vals)), vals, label=key)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Losses")
        ax.legend()
        ax.grid(True, alpha=0.3)
        panel_idx += 1

    if other_keys:
        ax = axes[panel_idx]
        for key in other_keys:
            vals = metrics[key]
            ax.plot(range(len(vals)), vals, label=key)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Value")
        ax.set_title("Metrics")
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    return fig


def plot_graph(
    coords: NDArray,
    edge_index: NDArray,
    values: NDArray | None = None,
    title: str = "",
    cmap: str = "viridis",
    point_size: float = 5.0,
    edge_alpha: float = 0.1,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Plot a spatial graph with nodes at coordinates and edges drawn."""
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(8, 8))

    # Draw edges
    src, dst = edge_index[0], edge_index[1]
    for s, d in zip(src, dst):
        ax.plot(
            [coords[s, 0], coords[d, 0]],
            [coords[s, 1], coords[d, 1]],
            c="gray",
            alpha=edge_alpha,
            linewidth=0.5,
        )

    # Draw nodes
    scatter_kwargs = dict(s=point_size, cmap=cmap, zorder=5)
    if values is not None:
        scatter_kwargs["c"] = values
    else:
        scatter_kwargs["c"] = "steelblue"

    sc = ax.scatter(coords[:, 0], coords[:, 1], **scatter_kwargs)

    if values is not None:
        plt.colorbar(sc, ax=ax, shrink=0.6)

    ax.set_title(title)
    ax.set_aspect("equal")
    ax.invert_yaxis()
    return ax
