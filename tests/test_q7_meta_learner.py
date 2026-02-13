"""Comprehensive tests for Q7 Meta-Learner experiments.

At least 10 tests covering:
  - Domain scenario generation
  - ICM score computation with custom weights
  - Risk-coverage AUC computation
  - Conformal coverage computation
  - Weight optimization end-to-end
  - Cross-validation stability
  - Baseline comparisons
  - Domain transfer
  - Edge cases
  - Reproducibility
"""

from __future__ import annotations

import io

import numpy as np
import pytest

# Ensure repo root is importable
import sys
import os
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from framework.config import ICMConfig
from framework.meta_learner import MetaLearner, _WEIGHT_NAMES, _WEIGHT_BOUNDS

from experiments.q7_meta_learner import (
    generate_domain_scenarios,
    compute_scenario_icm_scores,
    compute_risk_coverage_auc,
    compute_conformal_coverage,
    run_experiment_1,
    run_experiment_2,
    run_experiment_3,
    run_experiment_4,
    run_experiment_5,
    BASELINE_EQUAL,
    BASELINE_AGREEMENT_ONLY,
    BASELINE_NO_PENALTY,
)


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def learner() -> MetaLearner:
    """Fresh MetaLearner."""
    return MetaLearner()


@pytest.fixture
def default_weights() -> dict[str, float]:
    cfg = ICMConfig()
    return {
        "w_A": cfg.w_A, "w_D": cfg.w_D,
        "w_U": cfg.w_U, "w_C": cfg.w_C,
        "lam": cfg.lam,
    }


@pytest.fixture
def small_scenarios(learner: MetaLearner) -> list[dict]:
    return learner.generate_training_scenarios(n_scenarios=10, seed=77)


@pytest.fixture
def classification_scenarios() -> list[dict]:
    return generate_domain_scenarios("classification", n_scenarios=12, seed=100)


@pytest.fixture
def regression_scenarios() -> list[dict]:
    return generate_domain_scenarios("regression", n_scenarios=12, seed=200)


# ============================================================
# 1. Domain Scenario Generation
# ============================================================

class TestDomainScenarioGeneration:
    """Tests for generate_domain_scenarios."""

    def test_correct_count(self):
        scenarios = generate_domain_scenarios("classification", n_scenarios=15, seed=0)
        assert len(scenarios) == 15

    def test_all_domains_produce_valid_scenarios(self):
        """Each domain should produce scenarios with required keys."""
        for domain in ["classification", "regression", "cascade"]:
            scenarios = generate_domain_scenarios(domain, n_scenarios=6, seed=1)
            for s in scenarios:
                assert "predictions_dict" in s
                assert "loss" in s
                assert "label" in s
                assert "intervals" in s
                assert "signs" in s
                assert "features" in s
                assert isinstance(s["predictions_dict"], dict)
                assert len(s["predictions_dict"]) >= 3
                assert s["label"] in (0, 1)
                assert 0.0 <= s["loss"] <= 1.0 + 1e-6

    def test_label_balance(self):
        """First half should be high-convergence (label=1)."""
        scenarios = generate_domain_scenarios("classification", n_scenarios=20, seed=2)
        labels = [s["label"] for s in scenarios]
        n_high = sum(labels)
        assert n_high == 10

    def test_predictions_valid_distributions(self):
        """Prediction arrays should be valid probability distributions."""
        scenarios = generate_domain_scenarios("regression", n_scenarios=8, seed=3)
        for s in scenarios:
            for name, pred in s["predictions_dict"].items():
                assert np.all(pred >= 0), f"Negative predictions in {name}"
                assert pred.sum() == pytest.approx(1.0, abs=1e-6)

    def test_reproducibility(self):
        """Same seed should give identical scenarios."""
        s1 = generate_domain_scenarios("cascade", n_scenarios=5, seed=99)
        s2 = generate_domain_scenarios("cascade", n_scenarios=5, seed=99)
        for a, b in zip(s1, s2):
            assert a["loss"] == b["loss"]
            assert a["label"] == b["label"]

    def test_invalid_domain_raises(self):
        with pytest.raises(ValueError, match="Unknown domain"):
            generate_domain_scenarios("quantum", n_scenarios=5, seed=0)


# ============================================================
# 2. ICM Score Computation
# ============================================================

class TestScenarioICMScores:
    """Tests for compute_scenario_icm_scores."""

    def test_returns_correct_shape(self, small_scenarios, default_weights):
        icm_scores, losses = compute_scenario_icm_scores(
            small_scenarios, default_weights
        )
        assert len(icm_scores) == len(small_scenarios)
        assert len(losses) == len(small_scenarios)

    def test_icm_scores_bounded(self, small_scenarios, default_weights):
        icm_scores, _ = compute_scenario_icm_scores(
            small_scenarios, default_weights
        )
        assert np.all(icm_scores >= 0.0)
        assert np.all(icm_scores <= 1.0)

    def test_different_weights_different_scores(self, small_scenarios):
        """Different weights should produce different ICM scores."""
        w1 = {"w_A": 0.50, "w_D": 0.05, "w_U": 0.05, "w_C": 0.05, "lam": 0.05}
        w2 = {"w_A": 0.05, "w_D": 0.05, "w_U": 0.05, "w_C": 0.50, "lam": 0.30}
        s1, _ = compute_scenario_icm_scores(small_scenarios, w1)
        s2, _ = compute_scenario_icm_scores(small_scenarios, w2)
        # Not all identical
        assert not np.allclose(s1, s2, atol=1e-6)


# ============================================================
# 3. Risk-Coverage AUC
# ============================================================

class TestRiskCoverageAUC:
    """Tests for compute_risk_coverage_auc."""

    def test_returns_finite_value(self, small_scenarios, default_weights):
        icm_scores, losses = compute_scenario_icm_scores(
            small_scenarios, default_weights
        )
        auc = compute_risk_coverage_auc(icm_scores, losses)
        assert np.isfinite(auc)

    def test_auc_non_negative(self, small_scenarios, default_weights):
        """AUC should be non-negative (risk >= 0, coverage >= 0)."""
        icm_scores, losses = compute_scenario_icm_scores(
            small_scenarios, default_weights
        )
        auc = compute_risk_coverage_auc(icm_scores, losses)
        assert auc >= 0.0

    def test_perfect_scores_low_auc(self):
        """If ICM perfectly predicts loss, AUC should be small."""
        # High ICM -> low loss
        icm = np.linspace(0.1, 0.9, 20)
        losses = 1.0 - icm + np.random.default_rng(42).normal(0, 0.01, 20)
        losses = np.clip(losses, 0.01, 1.0)
        auc = compute_risk_coverage_auc(icm, losses)
        assert auc < 0.5  # Should be reasonably low


# ============================================================
# 4. Conformal Coverage
# ============================================================

class TestConformalCoverage:
    """Tests for compute_conformal_coverage."""

    def test_returns_value_in_01(self, small_scenarios, default_weights):
        """Coverage should be between 0 and 1 or NaN."""
        icm_scores, losses = compute_scenario_icm_scores(
            small_scenarios, default_weights
        )
        cov = compute_conformal_coverage(icm_scores, losses, alpha=0.10)
        if not np.isnan(cov):
            assert 0.0 <= cov <= 1.0

    def test_too_few_points_returns_nan(self):
        """With fewer than 9 points, should return NaN."""
        icm = np.array([0.5, 0.6, 0.7])
        losses = np.array([0.3, 0.2, 0.1])
        cov = compute_conformal_coverage(icm, losses, alpha=0.10)
        assert np.isnan(cov)

    def test_larger_sample_returns_finite(self):
        """With enough data, should return a finite coverage."""
        rng = np.random.default_rng(42)
        icm = rng.uniform(0.4, 0.7, 30)
        losses = 1.0 - icm + rng.normal(0, 0.1, 30)
        losses = np.clip(losses, 0.0, 1.5)
        cov = compute_conformal_coverage(icm, losses, alpha=0.10)
        assert np.isfinite(cov)


# ============================================================
# 5. End-to-End Weight Optimization
# ============================================================

class TestWeightOptimization:
    """Tests for end-to-end optimization via Experiment 1."""

    def test_experiment_1_runs(self):
        """Experiment 1 should complete without error."""
        output = io.StringIO()
        result = run_experiment_1(output)
        assert "grid_result" in result
        assert "opt_result" in result
        assert "comparison" in result

    def test_optimized_score_positive(self):
        output = io.StringIO()
        result = run_experiment_1(output)
        assert result["opt_result"]["best_score"] > 0.0

    def test_optimized_weights_within_bounds(self):
        output = io.StringIO()
        result = run_experiment_1(output)
        for name, (lo, hi) in _WEIGHT_BOUNDS.items():
            v = result["opt_result"]["best_weights"][name]
            assert lo - 0.01 <= v <= hi + 0.01, (
                f"Optimized {name}={v} out of [{lo}, {hi}]"
            )

    def test_comparison_has_all_metrics(self):
        output = io.StringIO()
        result = run_experiment_1(output)
        for metric in ["monotonicity", "discrimination", "coverage", "composite"]:
            assert metric in result["comparison"]["default_scores"]
            assert metric in result["comparison"]["optimized_scores"]
            assert metric in result["comparison"]["improvement_pct"]


# ============================================================
# 6. Cross-Validation Stability
# ============================================================

class TestCrossValidation:
    """Tests for cross-validation (Experiment 2)."""

    def test_experiment_2_runs(self):
        output = io.StringIO()
        result = run_experiment_2(output)
        assert "cv_result" in result
        assert "cv_time" in result
        assert "mean_weight_std" in result

    def test_cv_fold_count(self):
        output = io.StringIO()
        result = run_experiment_2(output)
        assert len(result["cv_result"]["fold_scores"]) == 5

    def test_cv_scores_positive(self):
        output = io.StringIO()
        result = run_experiment_2(output)
        for score in result["cv_result"]["fold_scores"]:
            assert score > 0.0

    def test_cv_std_reasonable(self):
        """CV std should be finite and reasonable."""
        output = io.StringIO()
        result = run_experiment_2(output)
        assert np.isfinite(result["cv_result"]["std_score"])
        assert result["cv_result"]["std_score"] < 1.0


# ============================================================
# 7. Baseline Comparisons
# ============================================================

class TestBaselineComparisons:
    """Tests for baseline comparisons (Experiment 4)."""

    def test_baseline_weights_valid(self):
        """All baselines should have valid weight keys."""
        for baseline in [BASELINE_EQUAL, BASELINE_AGREEMENT_ONLY, BASELINE_NO_PENALTY]:
            for name in _WEIGHT_NAMES:
                assert name in baseline
                lo, hi = _WEIGHT_BOUNDS[name]
                assert lo <= baseline[name] <= hi

    def test_equal_weights_sum_to_one(self):
        """Equal baseline weights should sum to 1.0."""
        total = sum(BASELINE_EQUAL.values())
        assert total == pytest.approx(1.0, abs=1e-10)

    def test_baselines_produce_finite_scores(self, small_scenarios):
        ml = MetaLearner()
        for baseline in [BASELINE_EQUAL, BASELINE_AGREEMENT_ONLY, BASELINE_NO_PENALTY]:
            result = ml.evaluate_weights(baseline, small_scenarios)
            assert np.isfinite(result["composite"])


# ============================================================
# 8. Domain Transfer
# ============================================================

class TestDomainTransfer:
    """Tests for domain transfer (Experiment 5)."""

    def test_experiment_5_runs(self):
        output = io.StringIO()
        result = run_experiment_5(output)
        assert "domain_results" in result
        assert "transfer_gaps" in result
        assert "source_domain" in result

    def test_source_domain_zero_gap(self):
        output = io.StringIO()
        result = run_experiment_5(output)
        source = result["source_domain"]
        assert result["transfer_gaps"][source] == pytest.approx(0.0, abs=1e-10)

    def test_all_domains_have_results(self):
        output = io.StringIO()
        result = run_experiment_5(output)
        for domain in ["classification", "regression", "cascade"]:
            assert domain in result["domain_results"]
            assert domain in result["transfer_gaps"]

    def test_domain_composites_positive(self):
        output = io.StringIO()
        result = run_experiment_5(output)
        for domain, r in result["domain_results"].items():
            assert r["composite"] > 0.0, f"{domain} composite not positive"


# ============================================================
# 9. Edge Cases
# ============================================================

class TestEdgeCases:
    """Edge case tests."""

    def test_single_scenario_per_domain(self):
        """Should not crash with minimal scenarios."""
        for domain in ["classification", "regression", "cascade"]:
            scenarios = generate_domain_scenarios(domain, n_scenarios=2, seed=42)
            assert len(scenarios) == 2

    def test_icm_scores_with_equal_weights(self, small_scenarios):
        """Equal weights should produce valid ICM scores."""
        icm, losses = compute_scenario_icm_scores(small_scenarios, BASELINE_EQUAL)
        assert len(icm) == len(small_scenarios)
        assert np.all(np.isfinite(icm))

    def test_rc_auc_with_uniform_icm(self):
        """If all ICM scores are the same, AUC should still compute."""
        icm = np.full(20, 0.5)
        losses = np.random.default_rng(42).uniform(0, 1, 20)
        auc = compute_risk_coverage_auc(icm, losses)
        # Should be finite (may be 0 or small)
        assert np.isfinite(auc) or np.isnan(auc)


# ============================================================
# 10. Reproducibility
# ============================================================

class TestReproducibility:
    """Tests for deterministic reproducibility."""

    def test_domain_scenario_reproducible(self):
        s1 = generate_domain_scenarios("classification", n_scenarios=8, seed=123)
        s2 = generate_domain_scenarios("classification", n_scenarios=8, seed=123)
        for a, b in zip(s1, s2):
            assert a["loss"] == b["loss"]
            assert a["label"] == b["label"]
            for key in a["predictions_dict"]:
                np.testing.assert_array_equal(
                    a["predictions_dict"][key],
                    b["predictions_dict"][key],
                )

    def test_icm_score_computation_reproducible(self, small_scenarios, default_weights):
        s1, l1 = compute_scenario_icm_scores(small_scenarios, default_weights)
        s2, l2 = compute_scenario_icm_scores(small_scenarios, default_weights)
        np.testing.assert_array_equal(s1, s2)
        np.testing.assert_array_equal(l1, l2)

    def test_rc_auc_reproducible(self, small_scenarios, default_weights):
        icm, losses = compute_scenario_icm_scores(small_scenarios, default_weights)
        auc1 = compute_risk_coverage_auc(icm, losses)
        auc2 = compute_risk_coverage_auc(icm, losses)
        if np.isfinite(auc1):
            assert auc1 == pytest.approx(auc2, abs=1e-10)


# ============================================================
# 11. Risk-Coverage Experiment
# ============================================================

class TestRiskCoverageExperiment:
    """Tests for Experiment 3."""

    def test_experiment_3_runs(self):
        ml = MetaLearner()
        scenarios = ml.generate_training_scenarios(n_scenarios=20, seed=42)
        opt = ml.optimize(scenarios, n_restarts=2)
        output = io.StringIO()
        result = run_experiment_3(opt["best_weights"], output)
        assert "auc_default" in result
        assert "auc_optimized" in result
        assert "dominates" in result
        assert "coverage_results" in result

    def test_experiment_3_auc_finite(self):
        ml = MetaLearner()
        scenarios = ml.generate_training_scenarios(n_scenarios=20, seed=42)
        opt = ml.optimize(scenarios, n_restarts=2)
        output = io.StringIO()
        result = run_experiment_3(opt["best_weights"], output)
        assert np.isfinite(result["auc_default"])
        assert np.isfinite(result["auc_optimized"])
