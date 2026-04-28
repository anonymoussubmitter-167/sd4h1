"""Tests for evaluation metrics."""

import numpy as np
import pytest

from rosetta.utils.metrics import morans_i, reconstruction_accuracy, silhouette_score_spatial


class TestMoransI:
    """Test Moran's I spatial autocorrelation metric."""

    def test_spatially_correlated_data(self):
        """Spatially correlated data should yield high Moran's I."""
        n = 200
        # Create a grid of points
        x = np.linspace(0, 10, 20)
        y = np.linspace(0, 10, 10)
        xx, yy = np.meshgrid(x, y)
        coords = np.column_stack([xx.ravel(), yy.ravel()])

        # Create embeddings that vary smoothly in space
        embeddings = np.column_stack([
            np.sin(coords[:, 0]),
            np.cos(coords[:, 1]),
            coords[:, 0] * coords[:, 1] / 100,
        ])

        mi = morans_i(embeddings, coords, k=6)
        assert mi > 0.3, f"Spatially correlated data should have Moran's I > 0.3, got {mi}"

    def test_random_data_near_zero(self):
        """Random embeddings should have Moran's I near 0."""
        rng = np.random.default_rng(42)
        n = 200
        coords = rng.uniform(0, 100, (n, 2))
        embeddings = rng.standard_normal((n, 10))

        mi = morans_i(embeddings, coords, k=6)
        assert abs(mi) < 0.2, f"Random data should have Moran's I near 0, got {mi}"

    def test_constant_embeddings(self):
        """Constant embeddings should yield Moran's I of 0."""
        n = 50
        coords = np.random.randn(n, 2)
        embeddings = np.ones((n, 5))

        mi = morans_i(embeddings, coords, k=6)
        assert mi == pytest.approx(0.0, abs=1e-6)

    def test_small_dataset(self):
        """Should handle small datasets gracefully."""
        coords = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=np.float32)
        embeddings = np.array([[1, 0], [1, 0], [0, 1], [0, 1]], dtype=np.float32)

        mi = morans_i(embeddings, coords, k=2)
        # Neighbors should have similar embeddings along rows
        assert isinstance(mi, float)


class TestSilhouetteScoreSpatial:
    """Test silhouette score metric."""

    def test_well_separated_clusters(self):
        """Well-separated clusters should have high silhouette score."""
        rng = np.random.default_rng(42)
        # Two well-separated clusters
        cluster_0 = rng.standard_normal((50, 10)) + 10
        cluster_1 = rng.standard_normal((50, 10)) - 10
        embeddings = np.vstack([cluster_0, cluster_1])
        labels = np.array([0] * 50 + [1] * 50)

        score = silhouette_score_spatial(embeddings, labels)
        assert score > 0.5, f"Well-separated clusters should have score > 0.5, got {score}"

    def test_single_cluster_returns_zero(self):
        """Single cluster should return 0."""
        embeddings = np.random.randn(50, 10)
        labels = np.zeros(50, dtype=int)

        score = silhouette_score_spatial(embeddings, labels)
        assert score == 0.0

    def test_returns_float(self):
        """Should return a float."""
        rng = np.random.default_rng(42)
        embeddings = rng.standard_normal((100, 10))
        labels = np.array([0] * 50 + [1] * 50)

        score = silhouette_score_spatial(embeddings, labels)
        assert isinstance(score, float)
        assert -1 <= score <= 1


class TestReconstructionAccuracy:
    """Test reconstruction accuracy metrics."""

    def test_perfect_reconstruction(self):
        """Perfect prediction should yield MSE=0, Pearson=1, Cosine=1."""
        target = np.random.randn(20, 50)
        pred = target.copy()
        mask = np.ones((20, 50), dtype=bool)

        result = reconstruction_accuracy(pred, target, mask)
        assert result["mse"] == pytest.approx(0.0, abs=1e-6)
        assert result["pearson_r"] == pytest.approx(1.0, abs=1e-4)
        assert result["cosine_sim"] == pytest.approx(1.0, abs=1e-4)

    def test_poor_reconstruction(self):
        """Random predictions should have high MSE and low correlation."""
        rng = np.random.default_rng(42)
        target = rng.standard_normal((20, 50))
        pred = rng.standard_normal((20, 50))
        mask = np.ones((20, 50), dtype=bool)

        result = reconstruction_accuracy(pred, target, mask)
        assert result["mse"] > 0.5
        assert abs(result["pearson_r"]) < 0.3

    def test_partial_mask(self):
        """Should only compute metrics on masked positions."""
        pred = np.zeros((20, 50))
        target = np.ones((20, 50))
        mask = np.zeros((20, 50), dtype=bool)
        mask[:5, :10] = True

        result = reconstruction_accuracy(pred, target, mask)
        assert result["mse"] == pytest.approx(1.0, abs=1e-6)

    def test_empty_mask(self):
        """Empty mask should return zeros."""
        pred = np.random.randn(20, 50)
        target = np.random.randn(20, 50)
        mask = np.zeros((20, 50), dtype=bool)

        result = reconstruction_accuracy(pred, target, mask)
        assert result["mse"] == 0.0
        assert result["pearson_r"] == 0.0

    def test_returns_dict(self):
        """Should return dict with expected keys."""
        pred = np.random.randn(10, 20)
        target = np.random.randn(10, 20)
        mask = np.ones((10, 20), dtype=bool)

        result = reconstruction_accuracy(pred, target, mask)
        assert "mse" in result
        assert "pearson_r" in result
        assert "cosine_sim" in result
