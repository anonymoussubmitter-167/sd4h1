"""Tests for cross-species alignment metrics."""

import numpy as np
import pytest

from rosetta.utils.metrics import (
    alignment_score,
    label_transfer_accuracy,
    label_transfer_with_confidence,
    spatial_label_smoothness,
)


class TestAlignmentScore:
    """Test kNN mixing score."""

    def test_perfect_mixing(self):
        """Interleaved species should yield normalized mixing score ~1.0."""
        rng = np.random.default_rng(42)
        n = 200
        # Both species from the same distribution
        embeddings = rng.standard_normal((n, 10))
        species = np.array([0, 1] * (n // 2))

        score = alignment_score(embeddings, species, k=20)
        # Normalized: 1.0 = perfect mixing (raw / expected)
        assert 0.7 < score < 1.3, f"Expected ~1.0 for random mixing, got {score}"

    def test_no_mixing(self):
        """Completely separated species should yield mixing score ~0."""
        rng = np.random.default_rng(42)
        n = 100
        # Species 0 centered at (10, 0, ...), species 1 at (-10, 0, ...)
        emb_0 = rng.standard_normal((n, 10)) + 100
        emb_1 = rng.standard_normal((n, 10)) - 100
        embeddings = np.vstack([emb_0, emb_1])
        species = np.array([0] * n + [1] * n)

        score = alignment_score(embeddings, species, k=20)
        assert score < 0.05, f"Expected ~0 for separated species, got {score}"

    def test_returns_float(self):
        embeddings = np.random.randn(50, 5)
        species = np.array([0] * 25 + [1] * 25)
        score = alignment_score(embeddings, species, k=5)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.5  # Normalized: ~1.0 for random, can exceed 1.0 slightly

    def test_small_dataset(self):
        embeddings = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=np.float32)
        species = np.array([0, 0, 1, 1])
        score = alignment_score(embeddings, species, k=2)
        assert isinstance(score, float)

    def test_single_species(self):
        """All same species should yield 0 mixing."""
        embeddings = np.random.randn(50, 5)
        species = np.zeros(50, dtype=int)
        score = alignment_score(embeddings, species, k=5)
        assert score == pytest.approx(0.0, abs=1e-6)


class TestLabelTransferAccuracy:
    """Test label transfer accuracy using kNN."""

    def test_perfect_transfer(self):
        """Identical source/target with matching labels should give accuracy=1."""
        rng = np.random.default_rng(42)
        # Two well-separated clusters
        z_0 = rng.standard_normal((50, 10)) + 10
        z_1 = rng.standard_normal((50, 10)) - 10
        z = np.vstack([z_0, z_1])
        labels = np.array([0] * 50 + [1] * 50)

        acc = label_transfer_accuracy(z, z.copy(), labels, labels.copy(), k=5)
        assert acc > 0.95, f"Expected ~1.0, got {acc}"

    def test_random_embeddings_chance(self):
        """Random embeddings should give accuracy near chance level."""
        rng = np.random.default_rng(42)
        n = 200
        d = 10
        n_types = 4
        z_source = rng.standard_normal((n, d))
        z_target = rng.standard_normal((n, d))
        labels = np.repeat(np.arange(n_types), n // n_types)

        acc = label_transfer_accuracy(z_source, z_target, labels, labels.copy(), k=10)
        # Chance = 1/4 = 0.25, allow some variance
        assert acc < 0.5, f"Expected near chance, got {acc}"

    def test_returns_float(self):
        z_s = np.random.randn(20, 5)
        z_t = np.random.randn(20, 5)
        labels = np.array([0] * 10 + [1] * 10)
        acc = label_transfer_accuracy(z_s, z_t, labels, labels.copy(), k=3)
        assert isinstance(acc, float)
        assert 0.0 <= acc <= 1.0


class TestSpatialLabelSmoothness:
    """Test spatial label smoothness metric."""

    def test_perfect_smoothness(self):
        """All same label should yield smoothness = 1.0."""
        rng = np.random.default_rng(42)
        coords = rng.standard_normal((50, 2))
        labels = np.zeros(50, dtype=int)

        score = spatial_label_smoothness(coords, labels, k=10)
        assert score == pytest.approx(1.0, abs=1e-6)

    def test_random_labels(self):
        """Random labels should yield smoothness ~ 1/C."""
        rng = np.random.default_rng(42)
        n = 1000
        C = 10
        coords = rng.standard_normal((n, 2))
        labels = rng.integers(0, C, size=n)

        score = spatial_label_smoothness(coords, labels, k=10)
        expected = 1.0 / C
        assert abs(score - expected) < 0.05, f"Expected ~{expected}, got {score}"


class TestLabelTransferWithConfidence:
    """Test label transfer with confidence scores."""

    def test_output_shapes(self):
        """Output arrays should match target size."""
        rng = np.random.default_rng(42)
        n_s, n_t, d = 50, 30, 10
        z_s = rng.standard_normal((n_s, d))
        z_t = rng.standard_normal((n_t, d))
        labels = np.repeat(np.arange(5), 10)

        pred, conf = label_transfer_with_confidence(z_s, z_t, labels, k=5)
        assert pred.shape == (n_t,)
        assert conf.shape == (n_t,)

    def test_confidence_range(self):
        """Confidences should be in (0, 1.0]."""
        rng = np.random.default_rng(42)
        k = 10
        z_s = rng.standard_normal((100, 8))
        z_t = rng.standard_normal((50, 8))
        labels = rng.integers(0, 5, size=100)

        _, conf = label_transfer_with_confidence(z_s, z_t, labels, k=k)
        # With distance-weighted voting, confidence is fraction of total
        # weight for the winner — always positive, at most 1.0
        assert np.all(conf > 0.0)
        assert np.all(conf <= 1.0 + 1e-9)


