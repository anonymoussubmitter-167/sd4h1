"""Tests for ATLAS conformal prediction annotation transfer."""

import numpy as np
import pytest

from rosetta.atlas.conformal import (
    _knn_class_probabilities,
    calibrate_conformal,
    conformal_label_transfer,
    conformal_prediction_sets,
)


class TestKNNProbabilities:
    """Test kNN class probability estimation."""

    def test_probabilities_sum_to_one(self):
        """Probabilities should sum to 1 for each query point."""
        rng = np.random.default_rng(42)
        n_train, n_query, d = 200, 50, 16
        z_train = rng.standard_normal((n_train, d)).astype(np.float32)
        z_query = rng.standard_normal((n_query, d)).astype(np.float32)
        labels = np.array(["A"] * 100 + ["B"] * 60 + ["C"] * 40)
        classes = np.array(["A", "B", "C"])

        probs = _knn_class_probabilities(z_train, z_query, labels, classes, k=10)

        assert probs.shape == (n_query, 3)
        np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-6)
        assert (probs >= 0).all()

    def test_separable_data(self):
        """Well-separated clusters should give high confidence."""
        rng = np.random.default_rng(42)
        # Class A centered at (10, 0), class B centered at (-10, 0)
        z_train = np.vstack([
            rng.standard_normal((50, 2)) + [10, 0],
            rng.standard_normal((50, 2)) + [-10, 0],
        ]).astype(np.float32)
        labels = np.array(["A"] * 50 + ["B"] * 50)
        classes = np.array(["A", "B"])

        # Query near class A
        z_query = np.array([[10.0, 0.0]], dtype=np.float32)
        probs = _knn_class_probabilities(z_train, z_query, labels, classes, k=10)

        assert probs[0, 0] > 0.9  # P(A) should be high


class TestCalibration:
    """Test conformal calibration."""

    def test_scores_bounded(self):
        """Nonconformity scores should be in [0, 1]."""
        rng = np.random.default_rng(42)
        n_train, n_cal, d = 100, 30, 8
        z_train = rng.standard_normal((n_train, d)).astype(np.float32)
        z_cal = rng.standard_normal((n_cal, d)).astype(np.float32)
        labels_train = np.array(["A"] * 50 + ["B"] * 50)
        labels_cal = np.array(["A"] * 15 + ["B"] * 15)
        classes = np.array(["A", "B"])

        scores = calibrate_conformal(z_train, z_cal, labels_train, labels_cal, classes, k=10)

        assert scores.shape == (n_cal,)
        assert (scores >= 0).all()
        assert (scores <= 1).all()


class TestConformalPredictionSets:
    """Test conformal prediction set construction."""

    def test_prediction_sets_nonempty(self):
        """Every prediction set should have at least one class."""
        rng = np.random.default_rng(42)
        n_train, n_test = 100, 30
        d = 8
        z_train = rng.standard_normal((n_train, d)).astype(np.float32)
        z_test = rng.standard_normal((n_test, d)).astype(np.float32)
        labels = np.array(["A"] * 50 + ["B"] * 50)
        classes = np.array(["A", "B"])
        cal_scores = rng.random(20)

        pred_sets, sizes, probs = conformal_prediction_sets(
            z_train, z_test, labels, classes, cal_scores, alpha=0.1, k=10,
        )

        assert len(pred_sets) == n_test
        assert all(len(ps) >= 1 for ps in pred_sets)
        assert (sizes >= 1).all()
        assert probs.shape == (n_test, 2)

    def test_lower_alpha_larger_sets(self):
        """Lower alpha (more coverage) should produce larger prediction sets."""
        rng = np.random.default_rng(42)
        n_train, n_test = 200, 50
        d = 8
        # Create overlapping clusters
        z_train = rng.standard_normal((n_train, d)).astype(np.float32)
        z_test = rng.standard_normal((n_test, d)).astype(np.float32)
        labels = np.array(["A"] * 70 + ["B"] * 70 + ["C"] * 60)
        classes = np.array(["A", "B", "C"])
        cal_scores = rng.random(30)

        _, sizes_10, _ = conformal_prediction_sets(
            z_train, z_test, labels, classes, cal_scores, alpha=0.1, k=10,
        )
        _, sizes_30, _ = conformal_prediction_sets(
            z_train, z_test, labels, classes, cal_scores, alpha=0.3, k=10,
        )

        # Alpha=0.1 (90% coverage) should have >= set sizes vs alpha=0.3 (70%)
        assert np.mean(sizes_10) >= np.mean(sizes_30)


class TestConformalLabelTransfer:
    """Test full conformal label transfer pipeline."""

    def test_full_pipeline(self):
        """Test end-to-end conformal label transfer."""
        rng = np.random.default_rng(42)
        n_lab, n_unlab, d = 200, 100, 16

        # Create well-separated classes
        z_labeled = np.vstack([
            rng.standard_normal((70, d)) + [5] * d,
            rng.standard_normal((70, d)) + [-5] * d,
            rng.standard_normal((60, d)),
        ]).astype(np.float32)
        labels = np.array(["A"] * 70 + ["B"] * 70 + ["C"] * 60)

        z_unlabeled = np.vstack([
            rng.standard_normal((40, d)) + [5] * d,
            rng.standard_normal((40, d)) + [-5] * d,
            rng.standard_normal((20, d)),
        ]).astype(np.float32)

        result = conformal_label_transfer(
            z_labeled, z_unlabeled, labels,
            alpha=0.1, cal_fraction=0.2, k=10,
        )

        assert "predicted_labels" in result
        assert "prediction_sets" in result
        assert "set_sizes" in result
        assert "probabilities" in result
        assert "classes" in result
        assert "coverage_guarantee" in result
        assert "mean_set_size" in result
        assert "singleton_fraction" in result
        assert "calibration_coverage" in result

        assert result["predicted_labels"].shape == (n_unlab,)
        assert len(result["prediction_sets"]) == n_unlab
        assert result["set_sizes"].shape == (n_unlab,)
        assert result["coverage_guarantee"] == 0.9
        assert result["probabilities"].shape == (n_unlab, 3)

        # With well-separated data, calibration coverage should meet target
        assert result["calibration_coverage"] >= 0.8

    def test_singleton_fraction_separable(self):
        """Well-separated classes should produce mostly singleton prediction sets."""
        rng = np.random.default_rng(42)
        d = 8

        # Very well-separated clusters
        z_labeled = np.vstack([
            rng.standard_normal((100, d)) + [20] * d,
            rng.standard_normal((100, d)) + [-20] * d,
        ]).astype(np.float32)
        labels = np.array(["A"] * 100 + ["B"] * 100)

        z_unlabeled = np.vstack([
            rng.standard_normal((30, d)) + [20] * d,
            rng.standard_normal((30, d)) + [-20] * d,
        ]).astype(np.float32)

        result = conformal_label_transfer(
            z_labeled, z_unlabeled, labels,
            alpha=0.1, cal_fraction=0.2, k=10,
        )

        # Most predictions should be singletons for well-separated data
        assert result["singleton_fraction"] > 0.8
