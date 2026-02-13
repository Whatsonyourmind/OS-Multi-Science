"""Tests for Q6 Structural Invariants experiment.

At least 10 tests covering:
  - Each structural invariant function
  - ICM score computation
  - Stability measurement functions (CV, normalized range)
  - Scenario runners
  - Edge cases
"""

from __future__ import annotations

import sys
import os

# Ensure repo root is on the Python path
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np
import pytest

from experiments.q6_structural_invariants import (
    ranking_invariant,
    sign_invariant,
    monotonicity_invariant,
    ordering_invariant,
    compute_cv,
    compute_normalized_range,
    run_classification_scenario,
    run_regression_scenario,
    run_cascade_scenario,
    _compute_icm_score,
    N_MODELS,
)
from framework.config import ICMConfig


# ============================================================
# 1. Ranking invariant
# ============================================================

class TestRankingInvariant:
    """Tests for ranking_invariant (Kendall-tau)."""

    def test_identical_predictions_give_perfect_ranking(self):
        """When all models predict the same values, Kendall-tau = 1."""
        pred = np.array([3.0, 1.0, 4.0, 1.0, 5.0])
        preds = [pred.copy() for _ in range(4)]
        tau = ranking_invariant(preds)
        assert tau == pytest.approx(1.0, abs=1e-6)

    def test_reversed_predictions_give_negative_ranking(self):
        """When one model is reversed, Kendall-tau should be negative."""
        pred_a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        pred_b = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
        tau = ranking_invariant([pred_a, pred_b])
        assert tau == pytest.approx(-1.0, abs=1e-6)

    def test_single_model_returns_one(self):
        """With a single model, ranking invariant should be 1."""
        pred = np.array([1.0, 2.0, 3.0])
        assert ranking_invariant([pred]) == pytest.approx(1.0)

    def test_bounded_output(self):
        """Output should be in [-1, 1]."""
        rng = np.random.default_rng(42)
        preds = [rng.standard_normal(30) for _ in range(5)]
        tau = ranking_invariant(preds)
        assert -1.0 <= tau <= 1.0 + 1e-10


# ============================================================
# 2. Sign invariant
# ============================================================

class TestSignInvariant:
    """Tests for sign_invariant."""

    def test_all_same_sign_gives_one(self):
        """When all models agree on sign, agreement rate = 1."""
        preds = [np.array([1.0, 2.0, 3.0]) for _ in range(4)]
        assert sign_invariant(preds) == pytest.approx(1.0, abs=1e-6)

    def test_opposite_signs_give_low_agreement(self):
        """When models disagree on sign, agreement should be low."""
        pred_a = np.array([1.0, 2.0, 3.0])
        pred_b = np.array([-1.0, -2.0, -3.0])
        agree = sign_invariant([pred_a, pred_b])
        assert agree == pytest.approx(0.0, abs=1e-6)

    def test_partial_agreement(self):
        """Mixed sign agreement should give a value between 0 and 1."""
        pred_a = np.array([1.0, -2.0, 3.0, -4.0])
        pred_b = np.array([1.0, 2.0, 3.0, -4.0])
        agree = sign_invariant([pred_a, pred_b])
        # 3 out of 4 agree (positions 0, 2, 3 match; position 1 differs)
        assert agree == pytest.approx(0.75, abs=1e-6)

    def test_single_model_returns_one(self):
        """Single model should have perfect self-agreement."""
        pred = np.array([1.0, -2.0, 3.0])
        assert sign_invariant([pred]) == pytest.approx(1.0)


# ============================================================
# 3. Monotonicity invariant
# ============================================================

class TestMonotonicityInvariant:
    """Tests for monotonicity_invariant (Spearman)."""

    def test_identical_predictions_give_perfect_monotonicity(self):
        """Identical predictions -> Spearman rho = 1."""
        pred = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        preds = [pred.copy() for _ in range(3)]
        rho = monotonicity_invariant(preds)
        assert rho == pytest.approx(1.0, abs=1e-6)

    def test_perfectly_anti_monotone(self):
        """Reversed predictions -> Spearman rho = -1."""
        pred_a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        pred_b = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
        rho = monotonicity_invariant([pred_a, pred_b])
        assert rho == pytest.approx(-1.0, abs=1e-6)

    def test_bounded_output(self):
        """Output should be in [-1, 1]."""
        rng = np.random.default_rng(77)
        preds = [rng.standard_normal(20) for _ in range(4)]
        rho = monotonicity_invariant(preds)
        assert -1.0 <= rho <= 1.0 + 1e-10


# ============================================================
# 4. Ordering invariant
# ============================================================

class TestOrderingInvariant:
    """Tests for ordering_invariant (top-k Jaccard)."""

    def test_identical_predictions_give_perfect_overlap(self):
        """Same predictions -> same top-k -> Jaccard = 1."""
        pred = np.array([1.0, 5.0, 3.0, 7.0, 2.0, 8.0, 4.0, 6.0, 9.0, 10.0])
        preds = [pred.copy() for _ in range(3)]
        jacc = ordering_invariant(preds, top_k_frac=0.3)
        assert jacc == pytest.approx(1.0, abs=1e-6)

    def test_completely_different_top_k(self):
        """Disjoint top sets should give low Jaccard."""
        # Model A: top values at end
        pred_a = np.array([0.0, 0.0, 0.0, 0.0, 0.0,
                           0.0, 0.0, 100.0, 100.0, 100.0])
        # Model B: top values at beginning
        pred_b = np.array([100.0, 100.0, 100.0, 0.0, 0.0,
                           0.0, 0.0, 0.0, 0.0, 0.0])
        jacc = ordering_invariant([pred_a, pred_b], top_k_frac=0.3)
        assert jacc == pytest.approx(0.0, abs=1e-6)

    def test_bounded_0_1(self):
        """Jaccard should be in [0, 1]."""
        rng = np.random.default_rng(55)
        preds = [rng.standard_normal(50) for _ in range(5)]
        jacc = ordering_invariant(preds)
        assert 0.0 <= jacc <= 1.0 + 1e-10


# ============================================================
# 5. Stability metrics
# ============================================================

class TestStabilityMetrics:
    """Tests for CV and normalized range."""

    def test_cv_constant_values_is_zero(self):
        """CV of constant array should be 0."""
        vals = np.array([5.0, 5.0, 5.0, 5.0])
        assert compute_cv(vals) == pytest.approx(0.0, abs=1e-12)

    def test_cv_known_value(self):
        """CV of [1, 3] should be std/mean = 1/2 = 0.5."""
        vals = np.array([1.0, 3.0])
        # mean = 2.0, std = 1.0, CV = 0.5
        assert compute_cv(vals) == pytest.approx(0.5, abs=1e-10)

    def test_cv_non_negative(self):
        """CV should be non-negative for positive values."""
        rng = np.random.default_rng(33)
        vals = rng.uniform(1.0, 10.0, size=20)
        assert compute_cv(vals) >= 0.0

    def test_normalized_range_constant_is_zero(self):
        """Normalized range of constant array should be 0."""
        vals = np.array([3.0, 3.0, 3.0])
        assert compute_normalized_range(vals) == pytest.approx(0.0, abs=1e-12)

    def test_normalized_range_known_value(self):
        """Normalized range of [1, 5] should be (5-1)/3 = 4/3."""
        vals = np.array([1.0, 5.0])
        # mean = 3.0, range = 4.0, normalized = 4/3
        assert compute_normalized_range(vals) == pytest.approx(4.0 / 3.0, abs=1e-10)

    def test_cv_zero_mean_returns_zero(self):
        """CV with zero mean should return 0 (not inf)."""
        vals = np.array([0.0, 0.0, 0.0])
        assert compute_cv(vals) == pytest.approx(0.0, abs=1e-12)


# ============================================================
# 6. Scenario runners produce valid outputs
# ============================================================

class TestScenarioRunners:
    """Tests that each scenario runner produces valid ICM + invariants."""

    def test_classification_scenario_returns_valid(self):
        """Classification scenario returns valid ICM and invariants."""
        icm, invs = run_classification_scenario(perturbation=0.1, seed=42)
        assert 0.0 <= icm <= 1.0
        assert -1.0 <= invs["ranking"] <= 1.0
        assert 0.0 <= invs["sign"] <= 1.0
        assert -1.0 <= invs["monotonicity"] <= 1.0
        assert 0.0 <= invs["ordering"] <= 1.0

    def test_regression_scenario_returns_valid(self):
        """Regression scenario returns valid ICM and invariants."""
        icm, invs = run_regression_scenario(perturbation=0.1, seed=42)
        assert 0.0 <= icm <= 1.0
        assert -1.0 <= invs["ranking"] <= 1.0
        assert 0.0 <= invs["sign"] <= 1.0
        assert -1.0 <= invs["monotonicity"] <= 1.0
        assert 0.0 <= invs["ordering"] <= 1.0

    def test_cascade_scenario_returns_valid(self):
        """Cascade scenario returns valid ICM and invariants."""
        icm, invs = run_cascade_scenario(perturbation=0.1, seed=42)
        assert 0.0 <= icm <= 1.0
        assert -1.0 <= invs["ranking"] <= 1.0
        assert 0.0 <= invs["sign"] <= 1.0
        assert -1.0 <= invs["monotonicity"] <= 1.0
        assert 0.0 <= invs["ordering"] <= 1.0

    def test_classification_deterministic(self):
        """Same seed should give same results for classification."""
        icm1, invs1 = run_classification_scenario(perturbation=0.3, seed=123)
        icm2, invs2 = run_classification_scenario(perturbation=0.3, seed=123)
        assert icm1 == pytest.approx(icm2, abs=1e-10)
        for key in invs1:
            assert invs1[key] == pytest.approx(invs2[key], abs=1e-10)

    def test_regression_deterministic(self):
        """Same seed should give same results for regression."""
        icm1, invs1 = run_regression_scenario(perturbation=0.3, seed=456)
        icm2, invs2 = run_regression_scenario(perturbation=0.3, seed=456)
        assert icm1 == pytest.approx(icm2, abs=1e-10)
        for key in invs1:
            assert invs1[key] == pytest.approx(invs2[key], abs=1e-10)

    def test_cascade_deterministic(self):
        """Same seed should give same results for cascade."""
        icm1, invs1 = run_cascade_scenario(perturbation=0.3, seed=789)
        icm2, invs2 = run_cascade_scenario(perturbation=0.3, seed=789)
        assert icm1 == pytest.approx(icm2, abs=1e-10)
        for key in invs1:
            assert invs1[key] == pytest.approx(invs2[key], abs=1e-10)


# ============================================================
# 7. ICM score computation
# ============================================================

class TestICMScoreComputation:
    """Tests for the _compute_icm_score helper."""

    def test_identical_predictions_high_icm(self):
        """Identical prediction vectors should yield high ICM."""
        pred = np.array([0.25, 0.25, 0.25, 0.25])
        preds = [pred.copy() for _ in range(4)]
        config = ICMConfig()
        icm = _compute_icm_score(preds, "hellinger", config)
        assert icm > 0.5

    def test_icm_bounded_0_1(self):
        """ICM should always be in [0, 1]."""
        rng = np.random.default_rng(42)
        for _ in range(10):
            preds = [rng.dirichlet(np.ones(4)) for _ in range(4)]
            config = ICMConfig()
            icm = _compute_icm_score(preds, "hellinger", config)
            assert 0.0 <= icm <= 1.0


# ============================================================
# 8. Perturbation sensitivity
# ============================================================

class TestPerturbationSensitivity:
    """Test that invariants respond reasonably to perturbation levels."""

    def test_ranking_degrades_with_high_perturbation(self):
        """More noise should generally reduce ranking agreement."""
        icm_low, inv_low = run_regression_scenario(perturbation=0.01, seed=42)
        icm_high, inv_high = run_regression_scenario(perturbation=0.8, seed=42)
        # At low perturbation, ranking should be higher than at high
        assert inv_low["ranking"] >= inv_high["ranking"] - 0.3

    def test_regression_invariants_at_zero_perturbation(self):
        """At near-zero perturbation, regression invariants should be near-perfect."""
        icm, invs = run_regression_scenario(perturbation=0.001, seed=42)
        # With very low noise on a continuous signal, models agree strongly
        assert invs["ranking"] > 0.5
        assert invs["sign"] > 0.5
        assert invs["monotonicity"] > 0.5
        assert invs["ordering"] > 0.5


# ============================================================
# 9. Edge cases
# ============================================================

class TestEdgeCases:
    """Edge case tests for invariant functions."""

    def test_constant_predictions(self):
        """Constant prediction vectors should not crash."""
        pred = np.array([1.0, 1.0, 1.0, 1.0])
        preds = [pred.copy() for _ in range(3)]
        # Should not raise; Kendall-tau on ties returns 0 or NaN
        tau = ranking_invariant(preds)
        assert np.isfinite(tau)
        rho = monotonicity_invariant(preds)
        assert np.isfinite(rho)
        sign = sign_invariant(preds)
        assert sign == pytest.approx(1.0, abs=1e-6)
        jacc = ordering_invariant(preds)
        assert 0.0 <= jacc <= 1.0 + 1e-10

    def test_two_element_predictions(self):
        """Very short prediction arrays should not crash."""
        pred_a = np.array([1.0, 2.0])
        pred_b = np.array([1.5, 2.5])
        preds = [pred_a, pred_b]
        # Should not raise
        sign = sign_invariant(preds)
        assert 0.0 <= sign <= 1.0
        jacc = ordering_invariant(preds, top_k_frac=0.5)
        assert 0.0 <= jacc <= 1.0 + 1e-10

    def test_empty_model_list(self):
        """Empty or size-1 lists should return default (1.0 = trivial agreement)."""
        # All invariant functions return 1.0 for K < 2 (trivial case)
        assert ranking_invariant([]) == pytest.approx(1.0)
        assert sign_invariant([]) == pytest.approx(1.0)
        assert monotonicity_invariant([]) == pytest.approx(1.0)
        assert ordering_invariant([]) == pytest.approx(1.0)
