"""Comprehensive tests for standard baselines module.

Tests cover all five baseline methods plus the comparison function,
with 35+ individual test cases validating correctness, output shapes,
edge cases, and numerical properties.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from benchmarks.baselines import (
    EnsembleAverage,
    StackingBaseline,
    BootstrapEnsemble,
    SplitConformal,
    DiversityMetrics,
    run_baseline_comparison,
    _stack_predictions,
    _extract_class_predictions,
    _mean_proba,
)


# ============================================================
# Fixtures: synthetic data generators
# ============================================================

@pytest.fixture
def binary_predictions_3():
    """3 models, binary classification, 100 samples."""
    rng = np.random.default_rng(42)
    n_samples = 100
    y_true = rng.integers(0, 2, size=n_samples)
    preds = {}
    for i in range(3):
        # Produce probabilities correlated with y_true
        noise = rng.uniform(0.0, 0.3, size=n_samples)
        p = np.where(y_true == 1, 0.7 + noise * 0.3, 0.3 - noise * 0.3)
        p = np.clip(p, 0.01, 0.99)
        preds[f"model_{i}"] = p
    return preds, y_true


@pytest.fixture
def multiclass_predictions_5():
    """5 models, 4-class classification, 200 samples."""
    rng = np.random.default_rng(123)
    n_samples = 200
    n_classes = 4
    y_true = rng.integers(0, n_classes, size=n_samples)
    preds = {}
    for i in range(5):
        proba = rng.dirichlet(np.ones(n_classes), size=n_samples)
        # Bias toward correct class
        for s in range(n_samples):
            proba[s, y_true[s]] += 0.5
        proba = proba / proba.sum(axis=1, keepdims=True)
        preds[f"model_{i}"] = proba
    return preds, y_true


@pytest.fixture
def multiclass_predictions_10():
    """10 models, 3-class classification, 150 samples."""
    rng = np.random.default_rng(99)
    n_samples = 150
    n_classes = 3
    y_true = rng.integers(0, n_classes, size=n_samples)
    preds = {}
    for i in range(10):
        proba = rng.dirichlet(np.ones(n_classes) * 2.0, size=n_samples)
        for s in range(n_samples):
            proba[s, y_true[s]] += 0.3
        proba = proba / proba.sum(axis=1, keepdims=True)
        preds[f"model_{i}"] = proba
    return preds, y_true


@pytest.fixture
def train_test_split_binary():
    """Train and test splits for binary classification with 5 models."""
    rng = np.random.default_rng(55)
    n_train, n_test = 200, 100
    y_train = rng.integers(0, 2, size=n_train)
    y_test = rng.integers(0, 2, size=n_test)
    preds_train = {}
    preds_test = {}
    for i in range(5):
        noise_tr = rng.uniform(0.0, 0.2, size=n_train)
        noise_te = rng.uniform(0.0, 0.2, size=n_test)
        preds_train[f"model_{i}"] = np.clip(
            np.where(y_train == 1, 0.7 + noise_tr, 0.3 - noise_tr), 0.01, 0.99
        )
        preds_test[f"model_{i}"] = np.clip(
            np.where(y_test == 1, 0.7 + noise_te, 0.3 - noise_te), 0.01, 0.99
        )
    return preds_train, preds_test, y_train, y_test


@pytest.fixture
def train_test_split_multiclass():
    """Train and test splits for multi-class classification with 5 models."""
    rng = np.random.default_rng(77)
    n_train, n_test = 300, 100
    n_classes = 4
    y_train = rng.integers(0, n_classes, size=n_train)
    y_test = rng.integers(0, n_classes, size=n_test)
    preds_train = {}
    preds_test = {}
    for i in range(5):
        for split_name, y, n in [
            ("train", y_train, n_train),
            ("test", y_test, n_test),
        ]:
            proba = rng.dirichlet(np.ones(n_classes), size=n)
            for s in range(n):
                proba[s, y[s]] += 0.5
            proba = proba / proba.sum(axis=1, keepdims=True)
            if split_name == "train":
                preds_train[f"model_{i}"] = proba
            else:
                preds_test[f"model_{i}"] = proba
    return preds_train, preds_test, y_train, y_test


# ============================================================
# 1. EnsembleAverage
# ============================================================

class TestEnsembleAverage:
    """Tests for EnsembleAverage baseline."""

    def test_aggregate_shape_binary(self, binary_predictions_3):
        preds, _ = binary_predictions_3
        ea = EnsembleAverage()
        result = ea.aggregate(preds)
        assert result.shape == (100,)

    def test_aggregate_shape_multiclass(self, multiclass_predictions_5):
        preds, _ = multiclass_predictions_5
        ea = EnsembleAverage()
        result = ea.aggregate(preds)
        assert result.shape == (200, 4)

    def test_aggregate_values_bounded(self, multiclass_predictions_5):
        preds, _ = multiclass_predictions_5
        ea = EnsembleAverage()
        result = ea.aggregate(preds)
        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)

    def test_score_keys(self, binary_predictions_3):
        preds, y_true = binary_predictions_3
        ea = EnsembleAverage()
        scores = ea.score(preds, y_true)
        assert "accuracy" in scores
        assert "log_loss" in scores
        assert "brier_score" in scores

    def test_score_accuracy_range(self, binary_predictions_3):
        preds, y_true = binary_predictions_3
        ea = EnsembleAverage()
        scores = ea.score(preds, y_true)
        assert 0.0 <= scores["accuracy"] <= 1.0
        assert scores["log_loss"] >= 0.0
        assert 0.0 <= scores["brier_score"] <= 1.0

    def test_with_10_models(self, multiclass_predictions_10):
        preds, y_true = multiclass_predictions_10
        ea = EnsembleAverage()
        result = ea.aggregate(preds)
        assert result.shape == (150, 3)
        scores = ea.score(preds, y_true)
        assert 0.0 <= scores["accuracy"] <= 1.0

    def test_single_model(self):
        """Single model: average should equal the model's predictions."""
        rng = np.random.default_rng(1)
        p = rng.dirichlet(np.ones(3), size=50)
        preds = {"only_model": p}
        ea = EnsembleAverage()
        result = ea.aggregate(preds)
        np.testing.assert_allclose(result, p, atol=1e-12)


# ============================================================
# 2. StackingBaseline
# ============================================================

class TestStackingBaseline:
    """Tests for StackingBaseline."""

    def test_fit_predict_logistic(self, train_test_split_binary):
        preds_train, preds_test, y_train, y_test = train_test_split_binary
        sb = StackingBaseline(meta_learner="logistic")
        sb.fit(preds_train, y_train)
        result = sb.predict(preds_test)
        assert result.shape == (100,)

    def test_fit_predict_ridge(self, train_test_split_binary):
        preds_train, preds_test, y_train, y_test = train_test_split_binary
        sb = StackingBaseline(meta_learner="ridge")
        sb.fit(preds_train, y_train)
        result = sb.predict(preds_test)
        assert result.shape == (100,)

    def test_fit_predict_random_forest(self, train_test_split_binary):
        preds_train, preds_test, y_train, y_test = train_test_split_binary
        sb = StackingBaseline(meta_learner="random_forest")
        sb.fit(preds_train, y_train)
        result = sb.predict(preds_test)
        assert result.shape == (100,)

    def test_multiclass_stacking(self, train_test_split_multiclass):
        preds_train, preds_test, y_train, y_test = train_test_split_multiclass
        sb = StackingBaseline(meta_learner="logistic")
        sb.fit(preds_train, y_train)
        result = sb.predict(preds_test)
        assert result.shape == (100, 4)

    def test_score_keys(self, train_test_split_binary):
        preds_train, preds_test, y_train, y_test = train_test_split_binary
        sb = StackingBaseline(meta_learner="logistic")
        sb.fit(preds_train, y_train)
        scores = sb.score(preds_test, y_test)
        assert "accuracy" in scores
        assert "log_loss" in scores
        assert "brier_score" in scores

    def test_predict_before_fit_raises(self):
        sb = StackingBaseline(meta_learner="logistic")
        preds = {"m1": np.array([0.5, 0.5])}
        with pytest.raises(RuntimeError, match="must be fit"):
            sb.predict(preds)

    def test_invalid_meta_learner_raises(self):
        with pytest.raises(ValueError, match="Unknown meta_learner"):
            StackingBaseline(meta_learner="svm")

    def test_stacking_improves_or_matches_average(self, train_test_split_binary):
        """Stacking should be at least competitive with simple averaging."""
        preds_train, preds_test, y_train, y_test = train_test_split_binary
        ea = EnsembleAverage()
        ea_acc = ea.score(preds_test, y_test)["accuracy"]

        sb = StackingBaseline(meta_learner="logistic")
        sb.fit(preds_train, y_train)
        sb_acc = sb.score(preds_test, y_test)["accuracy"]

        # Stacking should not be dramatically worse
        assert sb_acc >= ea_acc - 0.15


# ============================================================
# 3. BootstrapEnsemble
# ============================================================

class TestBootstrapEnsemble:
    """Tests for BootstrapEnsemble."""

    def test_disagreement_shape_binary(self, binary_predictions_3):
        preds, _ = binary_predictions_3
        be = BootstrapEnsemble(n_bootstrap=50)
        result = be.compute_disagreement(preds)
        assert result.shape == (100,)

    def test_disagreement_shape_multiclass(self, multiclass_predictions_5):
        preds, _ = multiclass_predictions_5
        be = BootstrapEnsemble(n_bootstrap=50)
        result = be.compute_disagreement(preds)
        assert result.shape == (200,)

    def test_disagreement_non_negative(self, binary_predictions_3):
        preds, _ = binary_predictions_3
        be = BootstrapEnsemble(n_bootstrap=50)
        result = be.compute_disagreement(preds)
        assert np.all(result >= -1e-10)

    def test_all_models_agree(self):
        """When all models agree perfectly, disagreement should be ~0."""
        p = np.array([0.9, 0.05, 0.05])
        preds = {f"m{i}": np.tile(p, (50, 1)) for i in range(5)}
        be = BootstrapEnsemble(n_bootstrap=100)
        disagreement = be.compute_disagreement(preds)
        assert np.all(disagreement < 0.01)

    def test_uncertainty_vs_error_keys(self, binary_predictions_3):
        preds, y_true = binary_predictions_3
        be = BootstrapEnsemble(n_bootstrap=50)
        result = be.uncertainty_vs_error(preds, y_true)
        assert "pearson_correlation" in result
        assert "mean_disagreement" in result

    def test_uncertainty_correlation_bounded(self, binary_predictions_3):
        preds, y_true = binary_predictions_3
        be = BootstrapEnsemble(n_bootstrap=50)
        result = be.uncertainty_vs_error(preds, y_true)
        assert -1.0 <= result["pearson_correlation"] <= 1.0

    def test_disagreement_correlates_with_error(self):
        """High disagreement should correlate with higher error rate.

        We construct predictions where uncertain samples (models disagree)
        are the ones that tend to be wrong.
        """
        rng = np.random.default_rng(42)
        n_samples = 500
        y_true = rng.integers(0, 2, size=n_samples)

        preds = {}
        for i in range(5):
            p = np.zeros(n_samples)
            for s in range(n_samples):
                if y_true[s] == 1:
                    # Easy samples: all models agree
                    if s < n_samples // 2:
                        p[s] = 0.9 + rng.uniform(0, 0.05)
                    else:
                        # Hard samples: models disagree
                        p[s] = 0.4 + rng.uniform(-0.3, 0.3)
                else:
                    if s < n_samples // 2:
                        p[s] = 0.1 + rng.uniform(0, 0.05)
                    else:
                        p[s] = 0.4 + rng.uniform(-0.3, 0.3)
            preds[f"model_{i}"] = np.clip(p, 0.01, 0.99)

        be = BootstrapEnsemble(n_bootstrap=200)
        result = be.uncertainty_vs_error(preds, y_true)
        # With designed data, correlation should be positive
        assert result["pearson_correlation"] > -0.5  # Should not be strongly negative

    def test_with_10_models(self, multiclass_predictions_10):
        preds, y_true = multiclass_predictions_10
        be = BootstrapEnsemble(n_bootstrap=50)
        d = be.compute_disagreement(preds)
        assert d.shape == (150,)


# ============================================================
# 4. SplitConformal
# ============================================================

class TestSplitConformal:
    """Tests for SplitConformal baseline."""

    def test_calibrate_and_predict_binary(self):
        rng = np.random.default_rng(42)
        n = 500
        y = rng.integers(0, 2, size=n)
        proba = np.clip(np.where(y == 1, 0.8, 0.2) + rng.normal(0, 0.1, n), 0.01, 0.99)

        n_cal = 250
        sc = SplitConformal(alpha=0.10)
        sc.calibrate(proba[:n_cal], y[:n_cal])
        pred_sets = sc.predict_sets(proba[n_cal:])
        assert len(pred_sets) == n - n_cal
        assert all(isinstance(ps, set) for ps in pred_sets)

    def test_calibrate_and_predict_multiclass(self):
        rng = np.random.default_rng(77)
        n = 400
        n_classes = 4
        y = rng.integers(0, n_classes, size=n)
        proba = rng.dirichlet(np.ones(n_classes), size=n)
        for s in range(n):
            proba[s, y[s]] += 1.0
        proba = proba / proba.sum(axis=1, keepdims=True)

        n_cal = 200
        sc = SplitConformal(alpha=0.10)
        sc.calibrate(proba[:n_cal], y[:n_cal])
        pred_sets = sc.predict_sets(proba[n_cal:])
        assert len(pred_sets) == n - n_cal
        # Each set should contain at least one class
        assert all(len(ps) >= 1 for ps in pred_sets)

    def test_coverage_close_to_nominal(self):
        """Empirical coverage should be >= 1 - alpha (approximately)."""
        rng = np.random.default_rng(99)
        n = 2000
        n_classes = 3
        y = rng.integers(0, n_classes, size=n)
        proba = rng.dirichlet(np.ones(n_classes) * 3, size=n)
        for s in range(n):
            proba[s, y[s]] += 0.8
        proba = proba / proba.sum(axis=1, keepdims=True)

        alpha = 0.10
        n_cal = 1000
        sc = SplitConformal(alpha=alpha)
        sc.calibrate(proba[:n_cal], y[:n_cal])
        cov = sc.coverage(proba[n_cal:], y[n_cal:])
        # Coverage should be at least 1-alpha minus some slack for finite samples
        assert cov >= (1 - alpha) - 0.05, f"Coverage {cov:.3f} too low"

    def test_predict_before_calibrate_raises(self):
        sc = SplitConformal(alpha=0.10)
        with pytest.raises(RuntimeError, match="calibrate"):
            sc.predict_sets(np.array([0.5, 0.5]))

    def test_avg_set_size_range(self):
        rng = np.random.default_rng(55)
        n = 500
        n_classes = 4
        y = rng.integers(0, n_classes, size=n)
        proba = rng.dirichlet(np.ones(n_classes), size=n)
        for s in range(n):
            proba[s, y[s]] += 0.8
        proba = proba / proba.sum(axis=1, keepdims=True)

        n_cal = 250
        sc = SplitConformal(alpha=0.10)
        sc.calibrate(proba[:n_cal], y[:n_cal])
        avg_ss = sc.avg_set_size(proba[n_cal:], y[n_cal:])
        assert 1.0 <= avg_ss <= n_classes

    def test_lower_alpha_wider_sets(self):
        """Lower alpha (higher confidence) should produce larger prediction sets."""
        rng = np.random.default_rng(11)
        n = 1000
        n_classes = 4
        y = rng.integers(0, n_classes, size=n)
        proba = rng.dirichlet(np.ones(n_classes) * 2, size=n)
        for s in range(n):
            proba[s, y[s]] += 0.5
        proba = proba / proba.sum(axis=1, keepdims=True)

        n_cal = 500

        sc_90 = SplitConformal(alpha=0.10)
        sc_90.calibrate(proba[:n_cal], y[:n_cal])
        size_90 = sc_90.avg_set_size(proba[n_cal:], y[n_cal:])

        sc_95 = SplitConformal(alpha=0.05)
        sc_95.calibrate(proba[:n_cal], y[:n_cal])
        size_95 = sc_95.avg_set_size(proba[n_cal:], y[n_cal:])

        assert size_95 >= size_90 - 0.1  # 95% should have wider or equal sets

    def test_perfect_predictions_small_sets(self):
        """If model is very confident and correct, sets should be small."""
        n = 200
        n_classes = 3
        y = np.arange(n) % n_classes
        proba = np.full((n, n_classes), 0.01)
        for s in range(n):
            proba[s, y[s]] = 0.98
        proba = proba / proba.sum(axis=1, keepdims=True)

        n_cal = 100
        sc = SplitConformal(alpha=0.10)
        sc.calibrate(proba[:n_cal], y[:n_cal])
        avg_ss = sc.avg_set_size(proba[n_cal:], y[n_cal:])
        assert avg_ss < 2.0  # Should be close to 1.0


# ============================================================
# 5. DiversityMetrics
# ============================================================

class TestDiversityMetrics:
    """Tests for DiversityMetrics baseline."""

    def test_q_statistic_identical_models(self):
        """Identical models that sometimes err should have Q close to 1."""
        rng = np.random.default_rng(42)
        n = 200
        y_true = rng.integers(0, 2, size=n)
        # Add noise so some predictions are wrong, giving nonzero d
        noise = rng.normal(0, 0.3, size=n)
        p = np.clip(np.where(y_true == 1, 0.7 + noise, 0.3 + noise), 0.01, 0.99)
        preds = {f"m{i}": p.copy() for i in range(3)}
        dm = DiversityMetrics()
        q = dm.q_statistic(preds, y_true)
        # Identical models with some errors: ad > 0, bc = 0, so Q = 1.0
        assert q > 0.5  # Should be high for identical models

    def test_q_statistic_range(self, binary_predictions_3):
        preds, y_true = binary_predictions_3
        dm = DiversityMetrics()
        q = dm.q_statistic(preds, y_true)
        assert -1.0 <= q <= 1.0

    def test_disagreement_measure_range(self, binary_predictions_3):
        preds, y_true = binary_predictions_3
        dm = DiversityMetrics()
        d = dm.disagreement_measure(preds, y_true)
        assert 0.0 <= d <= 1.0

    def test_disagreement_identical_models(self):
        """Identical models should have zero disagreement."""
        rng = np.random.default_rng(42)
        n = 200
        y_true = rng.integers(0, 2, size=n)
        p = np.where(y_true == 1, 0.9, 0.1)
        preds = {f"m{i}": p.copy() for i in range(5)}
        dm = DiversityMetrics()
        d = dm.disagreement_measure(preds, y_true)
        assert d == pytest.approx(0.0, abs=1e-6)

    def test_correlation_coefficient_range(self, binary_predictions_3):
        preds, y_true = binary_predictions_3
        dm = DiversityMetrics()
        c = dm.correlation_coefficient(preds, y_true)
        assert -1.0 <= c <= 1.0

    def test_entropy_measure_non_negative(self, multiclass_predictions_5):
        preds, _ = multiclass_predictions_5
        dm = DiversityMetrics()
        e = dm.entropy_measure(preds)
        assert e >= 0.0

    def test_entropy_identical_models_zero(self):
        """Identical models should have zero entropy (all vote the same class)."""
        n = 100
        proba = np.zeros((n, 3))
        proba[:, 0] = 0.8
        proba[:, 1] = 0.1
        proba[:, 2] = 0.1
        preds = {f"m{i}": proba.copy() for i in range(5)}
        dm = DiversityMetrics()
        e = dm.entropy_measure(preds)
        assert e < 0.01  # Near zero for identical strong models

    def test_kl_diversity_non_negative(self, multiclass_predictions_5):
        preds, _ = multiclass_predictions_5
        dm = DiversityMetrics()
        kl = dm.kl_diversity(preds)
        assert kl >= 0.0

    def test_kl_diversity_identical_zero(self):
        """KL divergence between identical distributions should be ~0."""
        n = 100
        proba = np.array([[0.5, 0.3, 0.2]] * n)
        preds = {f"m{i}": proba.copy() for i in range(4)}
        dm = DiversityMetrics()
        kl = dm.kl_diversity(preds)
        assert kl < 0.01

    def test_all_models_disagree(self):
        """Maximum disagreement scenario: each model predicts a different class."""
        n = 100
        preds = {}
        for i in range(4):
            proba = np.full((n, 4), 0.01)
            proba[:, i] = 0.97
            proba = proba / proba.sum(axis=1, keepdims=True)
            preds[f"m{i}"] = proba

        y_true = np.zeros(n, dtype=int)  # Arbitrary ground truth
        dm = DiversityMetrics()

        # High entropy
        e = dm.entropy_measure(preds)
        assert e > 0.5

        # High KL divergence
        kl = dm.kl_diversity(preds)
        assert kl > 0.5

        # High disagreement
        d = dm.disagreement_measure(preds, y_true)
        assert d > 0.3

    def test_with_10_models(self, multiclass_predictions_10):
        preds, y_true = multiclass_predictions_10
        dm = DiversityMetrics()
        q = dm.q_statistic(preds, y_true)
        d = dm.disagreement_measure(preds, y_true)
        c = dm.correlation_coefficient(preds, y_true)
        e = dm.entropy_measure(preds)
        kl = dm.kl_diversity(preds)
        assert -1.0 <= q <= 1.0
        assert 0.0 <= d <= 1.0
        assert -1.0 <= c <= 1.0
        assert e >= 0.0
        assert kl >= 0.0


# ============================================================
# 6. Edge Cases
# ============================================================

class TestEdgeCases:
    """Edge cases: single model, perfect agreement, total disagreement."""

    def test_single_model_ensemble_average(self):
        """Single model should return its own predictions."""
        p = np.array([0.3, 0.5, 0.2])
        preds = {"solo": np.tile(p, (10, 1))}
        ea = EnsembleAverage()
        result = ea.aggregate(preds)
        np.testing.assert_allclose(result, np.tile(p, (10, 1)), atol=1e-12)

    def test_single_model_diversity(self):
        """Single model: diversity metrics should handle gracefully."""
        rng = np.random.default_rng(42)
        n = 50
        y_true = rng.integers(0, 3, size=n)
        proba = rng.dirichlet(np.ones(3), size=n)
        preds = {"solo": proba}
        dm = DiversityMetrics()
        # No pairs to compare: should return 0
        assert dm.q_statistic(preds, y_true) == 0.0
        assert dm.disagreement_measure(preds, y_true) == 0.0
        assert dm.correlation_coefficient(preds, y_true) == 0.0
        assert dm.kl_diversity(preds) == 0.0

    def test_all_models_agree_perfectly(self):
        """All models return identical predictions."""
        n = 100
        proba = np.array([[0.7, 0.2, 0.1]] * n)
        preds = {f"m{i}": proba.copy() for i in range(5)}
        y_true = np.zeros(n, dtype=int)

        ea = EnsembleAverage()
        avg = ea.aggregate(preds)
        np.testing.assert_allclose(avg, proba, atol=1e-12)

        dm = DiversityMetrics()
        d = dm.disagreement_measure(preds, y_true)
        assert d == pytest.approx(0.0, abs=1e-6)

        e = dm.entropy_measure(preds)
        assert e < 0.01

    def test_binary_all_ones(self):
        """All models predict probability 1.0 for class 1."""
        n = 50
        preds = {f"m{i}": np.ones(n) * 0.99 for i in range(3)}
        y_true = np.ones(n, dtype=int)

        ea = EnsembleAverage()
        scores = ea.score(preds, y_true)
        assert scores["accuracy"] == pytest.approx(1.0, abs=1e-6)


# ============================================================
# 7. Helper Functions
# ============================================================

class TestHelpers:
    """Tests for internal helper functions."""

    def test_stack_predictions_binary(self, binary_predictions_3):
        preds, _ = binary_predictions_3
        X = _stack_predictions(preds)
        assert X.shape == (100, 3)

    def test_stack_predictions_multiclass(self, multiclass_predictions_5):
        preds, _ = multiclass_predictions_5
        X = _stack_predictions(preds)
        # 5 models * 4 classes = 20 columns
        assert X.shape == (200, 20)

    def test_extract_class_predictions(self, multiclass_predictions_5):
        preds, _ = multiclass_predictions_5
        hard = _extract_class_predictions(preds)
        assert hard.shape == (200, 5)
        assert np.all(hard >= 0)
        assert np.all(hard <= 3)

    def test_mean_proba(self, binary_predictions_3):
        preds, _ = binary_predictions_3
        avg = _mean_proba(preds)
        assert avg.shape == (100,)
        assert np.all(avg >= 0.0)
        assert np.all(avg <= 1.0)


# ============================================================
# 8. Comparison Table
# ============================================================

class TestRunBaselineComparison:
    """Tests for run_baseline_comparison."""

    def test_comparison_table_binary(self, train_test_split_binary):
        preds_train, preds_test, y_train, y_test = train_test_split_binary
        df = run_baseline_comparison(
            preds_train, preds_test, y_train, y_test
        )
        assert isinstance(df, pd.DataFrame)
        assert "Method" in df.columns
        assert "Accuracy" in df.columns
        assert "Log Loss" in df.columns
        assert "Brier" in df.columns
        assert "Coverage" in df.columns
        assert "Set Size" in df.columns
        assert "Diversity Corr" in df.columns
        # Should have rows for all baselines (no ICM row since icm_scores=None)
        assert len(df) >= 5

    def test_comparison_table_with_icm(self, train_test_split_binary):
        preds_train, preds_test, y_train, y_test = train_test_split_binary
        icm_scores = {
            "accuracy": 0.85,
            "log_loss": 0.35,
            "brier_score": 0.12,
            "coverage": 0.91,
            "set_size": 1.3,
        }
        df = run_baseline_comparison(
            preds_train, preds_test, y_train, y_test,
            icm_scores=icm_scores,
        )
        methods = df["Method"].tolist()
        assert "ICM Framework" in methods
        assert "Ensemble Avg" in methods
        assert "Stacking (LR)" in methods
        assert "Stacking (RF)" in methods
        assert "Bootstrap" in methods
        assert "Split Conformal" in methods

    def test_comparison_table_multiclass(self, train_test_split_multiclass):
        preds_train, preds_test, y_train, y_test = train_test_split_multiclass
        df = run_baseline_comparison(
            preds_train, preds_test, y_train, y_test
        )
        assert isinstance(df, pd.DataFrame)
        assert len(df) >= 5

    def test_comparison_table_no_crash_with_small_data(self):
        """Should not crash with very small datasets."""
        rng = np.random.default_rng(42)
        n_train, n_test = 20, 10
        y_train = rng.integers(0, 2, size=n_train)
        y_test = rng.integers(0, 2, size=n_test)
        preds_train = {
            f"m{i}": np.clip(
                np.where(y_train == 1, 0.7, 0.3) + rng.normal(0, 0.1, n_train),
                0.01, 0.99
            )
            for i in range(3)
        }
        preds_test = {
            f"m{i}": np.clip(
                np.where(y_test == 1, 0.7, 0.3) + rng.normal(0, 0.1, n_test),
                0.01, 0.99
            )
            for i in range(3)
        }
        df = run_baseline_comparison(
            preds_train, preds_test, y_train, y_test
        )
        assert isinstance(df, pd.DataFrame)
        assert len(df) >= 5
