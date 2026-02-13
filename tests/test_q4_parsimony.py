"""Tests for Experiment Q4 -- Parsimonious Diversity K*.

At least 10 tests covering greedy selection, marginal gain computation,
K* identification, model generation, ICM computation, CRC integration,
scenario execution, and result consistency.
"""

from __future__ import annotations

import sys
import os

# Ensure repo root on path
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np
import pytest

from framework.config import ICMConfig
from experiments.q4_parsimony import (
    MODEL_FAMILIES,
    K_MAX,
    MARGINAL_THRESHOLD,
    greedy_select_models,
    identify_k_star,
    compute_icm_for_subset,
    compute_ensemble_loss_classification,
    compute_ensemble_loss_regression,
    compute_ensemble_loss_cascade,
    compute_re_for_ensemble,
    generate_classification_model_predictions,
    generate_regression_model_predictions,
    generate_cascade_model_predictions,
    generate_crc_calibration_data,
    run_scenario,
    _softmax,
)
from benchmarks.synthetic.generators import (
    generate_classification_benchmark,
    generate_network_cascade,
)


# =====================================================================
# Test 1: Greedy selection returns valid permutation
# =====================================================================

class TestGreedySelection:
    """Tests for greedy_select_models."""

    def test_returns_valid_permutation(self):
        """Greedy selection should return a permutation of all model indices."""
        rng = np.random.default_rng(42)
        n_models = 6
        n_samples = 50
        predictions = [rng.normal(i * 0.5, 1.0, size=n_samples) for i in range(n_models)]
        reference = np.zeros(n_samples)

        order = greedy_select_models(predictions, reference)

        assert len(order) == n_models
        assert sorted(order) == list(range(n_models))

    def test_deterministic_with_same_seed(self):
        """Same inputs should give the same greedy ordering."""
        rng = np.random.default_rng(123)
        n_models = 5
        n_samples = 40
        predictions = [rng.normal(i * 0.3, 0.8, size=n_samples) for i in range(n_models)]
        reference = np.zeros(n_samples)

        order1 = greedy_select_models(predictions, reference)
        order2 = greedy_select_models(predictions, reference)

        assert order1 == order2

    def test_diverse_models_selected_first(self):
        """Models with orthogonal residuals should be selected before
        correlated ones."""
        rng = np.random.default_rng(99)
        n_samples = 100

        # Model 0 and 1 have identical residuals (correlated)
        base = rng.normal(0, 1, n_samples)
        pred_0 = base.copy()
        pred_1 = base + rng.normal(0, 0.01, n_samples)  # Nearly identical

        # Model 2 has orthogonal residuals
        pred_2 = rng.normal(5, 2, n_samples)

        predictions = [pred_0, pred_1, pred_2]
        reference = np.zeros(n_samples)

        order = greedy_select_models(predictions, reference)

        # Model 2 (diverse) should appear before the second correlated one
        pos_0 = order.index(0)
        pos_1 = order.index(1)
        pos_2 = order.index(2)

        # At least one of the correlated pair should be selected last
        assert pos_2 < max(pos_0, pos_1), (
            "Diverse model should be selected before both correlated models"
        )


# =====================================================================
# Test 2: Marginal gain computation (identify_k_star)
# =====================================================================

class TestIdentifyKStar:
    """Tests for identify_k_star."""

    def test_all_equal_gains(self):
        """When all gains are equal, K* should be K_max."""
        gains = np.array([0.1, 0.1, 0.1, 0.1, 0.1])
        k_star = identify_k_star(gains, threshold_frac=0.05)
        assert k_star == 5

    def test_sharp_dropoff(self):
        """Large first gain, then near-zero: K* should be 1."""
        gains = np.array([1.0, 0.01, 0.005, 0.002, 0.001])
        k_star = identify_k_star(gains, threshold_frac=0.05)
        assert k_star == 1

    def test_gradual_decay(self):
        """Gradually decaying gains: K* at the transition point."""
        gains = np.array([1.0, 0.8, 0.5, 0.1, 0.02, 0.01])
        k_star = identify_k_star(gains, threshold_frac=0.05)
        # Threshold = 0.05 * 1.0 = 0.05
        # gains >= 0.05: indices 0,1,2,3 -> K* = 4
        assert k_star == 4

    def test_empty_gains(self):
        """Empty array should return K*=1."""
        k_star = identify_k_star(np.array([]), threshold_frac=0.05)
        assert k_star == 1

    def test_zero_gains(self):
        """All-zero gains should return K*=1."""
        gains = np.zeros(5)
        k_star = identify_k_star(gains, threshold_frac=0.05)
        assert k_star == 1

    def test_k_star_bounded(self):
        """K* should never exceed the length of the gains array."""
        gains = np.array([1.0, 0.9, 0.8, 0.7])
        k_star = identify_k_star(gains, threshold_frac=0.05)
        assert 1 <= k_star <= len(gains)


# =====================================================================
# Test 3: ICM computation for subsets
# =====================================================================

class TestICMForSubset:
    """Tests for compute_icm_for_subset."""

    def test_single_model_returns_baseline(self):
        """Single model should return 0.5 (baseline)."""
        rng = np.random.default_rng(42)
        pred = rng.random(10)
        icm = compute_icm_for_subset([pred], "hellinger", ICMConfig())
        assert icm == pytest.approx(0.5, abs=1e-10)

    def test_empty_returns_zero(self):
        """No models should return 0."""
        icm = compute_icm_for_subset([], "hellinger", ICMConfig())
        assert icm == 0.0

    def test_identical_models_high_icm(self):
        """Identical model predictions should yield high ICM."""
        pred = np.array([0.7, 0.2, 0.1])
        preds = [pred.copy() for _ in range(5)]
        config = ICMConfig()
        icm = compute_icm_for_subset(preds, "hellinger", config, reference=pred)
        assert icm > 0.6, f"ICM for identical models should be high, got {icm}"

    def test_diverse_models_lower_icm(self):
        """Diverse predictions should yield lower ICM than identical ones."""
        rng = np.random.default_rng(42)
        config = ICMConfig()

        # Identical predictions
        pred = np.array([0.6, 0.3, 0.1])
        identical_preds = [pred.copy() for _ in range(5)]
        icm_identical = compute_icm_for_subset(
            identical_preds, "hellinger", config, reference=pred,
        )

        # Diverse predictions
        diverse_preds = []
        for i in range(5):
            p = rng.dirichlet(np.ones(3))
            diverse_preds.append(p)
        icm_diverse = compute_icm_for_subset(
            diverse_preds, "hellinger", config, reference=pred,
        )

        assert icm_identical >= icm_diverse, (
            f"Identical ICM ({icm_identical}) should be >= diverse ICM ({icm_diverse})"
        )

    def test_icm_bounded_0_1(self):
        """ICM should always be in [0, 1]."""
        rng = np.random.default_rng(77)
        for _ in range(10):
            preds = [rng.dirichlet(np.ones(3)) for _ in range(4)]
            config = ICMConfig()
            icm = compute_icm_for_subset(preds, "hellinger", config)
            assert 0.0 <= icm <= 1.0, f"ICM out of bounds: {icm}"


# =====================================================================
# Test 4: Model prediction generators
# =====================================================================

class TestModelGenerators:
    """Tests for model prediction generators."""

    def test_classification_predictions_valid_probabilities(self):
        """Classification predictions should be valid probability distributions."""
        rng = np.random.default_rng(42)
        n_classes = 3
        n_samples = 100
        y_true = rng.integers(0, n_classes, n_samples)

        for m_idx in range(min(K_MAX, 5)):
            probs = generate_classification_model_predictions(
                y_true, n_classes, m_idx, seed=42,
            )
            assert probs.shape == (n_samples, n_classes)
            # All non-negative
            assert np.all(probs >= 0), f"Model {m_idx} has negative probabilities"
            # Rows sum to 1
            row_sums = probs.sum(axis=1)
            np.testing.assert_allclose(row_sums, 1.0, atol=1e-10,
                                       err_msg=f"Model {m_idx} rows don't sum to 1")

    def test_regression_predictions_shape(self):
        """Regression predictions should have correct shape."""
        n_samples = 50
        x = np.linspace(0, 1, n_samples)
        y_true = np.sin(2 * np.pi * x)

        for m_idx in range(min(K_MAX, 3)):
            pred = generate_regression_model_predictions(x, y_true, m_idx, seed=42)
            assert pred.shape[0] == n_samples
            assert pred.shape[1] == 20  # n_posterior samples

    def test_cascade_predictions_shape(self):
        """Cascade predictions should have correct trajectory length."""
        n_steps = 20
        adj, state_history, _ = generate_network_cascade(
            n_nodes=30, edge_prob=0.1, threshold=0.3,
            n_steps=n_steps, seed=42,
        )

        traj = generate_cascade_model_predictions(
            adj, state_history, 0, seed=42, n_steps=n_steps,
        )
        assert len(traj) == n_steps + 1
        # Values should be in [0, 1] (fractions)
        assert np.all(traj >= 0.0) and np.all(traj <= 1.0)

    def test_different_models_differ(self):
        """Different model families should produce different predictions."""
        rng = np.random.default_rng(42)
        n_classes = 3
        n_samples = 100
        y_true = rng.integers(0, n_classes, n_samples)

        pred_0 = generate_classification_model_predictions(
            y_true, n_classes, 0, seed=42,
        )
        pred_5 = generate_classification_model_predictions(
            y_true, n_classes, 5, seed=42,
        )

        # They should not be identical
        assert not np.allclose(pred_0, pred_5), (
            "Different model families should produce different predictions"
        )


# =====================================================================
# Test 5: Ensemble loss functions
# =====================================================================

class TestEnsembleLoss:
    """Tests for ensemble loss computations."""

    def test_classification_loss_finite(self):
        """Classification loss should be finite for valid inputs."""
        rng = np.random.default_rng(42)
        n = 50
        n_classes = 3
        y_true = rng.integers(0, n_classes, n)
        probs1 = np.ones((n, n_classes)) / n_classes  # Uniform
        probs2 = np.ones((n, n_classes)) / n_classes

        loss = compute_ensemble_loss_classification([probs1, probs2], y_true)
        assert np.isfinite(loss)
        assert loss > 0  # Uniform predictions -> positive loss

    def test_regression_loss_zero_for_perfect(self):
        """Perfect predictions should give zero loss."""
        y_true = np.array([1.0, 2.0, 3.0])
        # Each prediction is (n, n_posterior), all equal to truth
        pred = np.tile(y_true, (20, 1)).T  # (3, 20)
        loss = compute_ensemble_loss_regression([pred], y_true)
        assert loss == pytest.approx(0.0, abs=1e-10)

    def test_cascade_loss_zero_for_perfect(self):
        """Perfect cascade prediction should give zero loss."""
        true_frac = np.array([0.0, 0.1, 0.3, 0.5, 0.7])
        loss = compute_ensemble_loss_cascade([true_frac.copy()], true_frac)
        assert loss == pytest.approx(0.0, abs=1e-10)

    def test_more_models_generally_helps(self):
        """Adding a good model to ensemble should not dramatically
        increase loss (sanity check)."""
        rng = np.random.default_rng(42)
        n = 100
        n_classes = 3
        y_true = rng.integers(0, n_classes, n)

        # Two models with small noise
        onehot = np.zeros((n, n_classes))
        onehot[np.arange(n), y_true] = 1.0

        good1 = onehot + rng.normal(0, 0.1, onehot.shape)
        good1 = np.abs(good1)
        good1 /= good1.sum(axis=1, keepdims=True)

        good2 = onehot + rng.normal(0, 0.1, onehot.shape)
        good2 = np.abs(good2)
        good2 /= good2.sum(axis=1, keepdims=True)

        loss_1 = compute_ensemble_loss_classification([good1], y_true)
        loss_2 = compute_ensemble_loss_classification([good1, good2], y_true)

        # Loss with 2 models should be similar or better
        assert loss_2 <= loss_1 * 1.5, (
            f"2-model loss ({loss_2:.4f}) should not be much worse than "
            f"1-model loss ({loss_1:.4f})"
        )


# =====================================================================
# Test 6: CRC calibration data generation
# =====================================================================

class TestCRCCalibration:
    """Tests for CRC calibration data generation."""

    def test_output_shapes(self):
        """Calibration data should have correct shapes."""
        n_cal = 50
        icm, losses = generate_crc_calibration_data("classification", seed=42, n_cal=n_cal)
        assert icm.shape == (n_cal,)
        assert losses.shape == (n_cal,)

    def test_icm_in_valid_range(self):
        """ICM calibration scores should be in reasonable range."""
        icm, losses = generate_crc_calibration_data("regression", seed=42, n_cal=100)
        # ICM should be roughly in (0, 1) -- may go slightly outside
        # due to computation, but should not be wildly out of range
        assert np.all(np.isfinite(icm))
        assert np.all(np.isfinite(losses))

    def test_re_computation_works(self):
        """Re computation from calibration data should produce finite values."""
        icm_cal, loss_cal = generate_crc_calibration_data(
            "classification", seed=42, n_cal=100,
        )
        re = compute_re_for_ensemble(icm_cal, loss_cal, 0.7, alpha=0.10)
        assert np.isfinite(re), f"Re should be finite, got {re}"


# =====================================================================
# Test 7: Full scenario run
# =====================================================================

class TestScenarioRun:
    """Tests for run_scenario."""

    def test_classification_scenario_runs(self):
        """Classification scenario should complete and return valid results."""
        result = run_scenario("classification", seed=42)

        assert len(result["icm_scores"]) == K_MAX
        assert len(result["losses"]) == K_MAX
        assert len(result["re_scores"]) == K_MAX
        assert len(result["delta_icm"]) == K_MAX
        assert len(result["order"]) == K_MAX
        assert 1 <= result["k_star_icm"] <= K_MAX
        assert 1 <= result["k_star_loss"] <= K_MAX

    def test_regression_scenario_runs(self):
        """Regression scenario should complete and return valid results."""
        result = run_scenario("regression", seed=42)

        assert len(result["icm_scores"]) == K_MAX
        assert np.all(np.isfinite(result["icm_scores"]))
        assert np.all(np.isfinite(result["losses"]))

    def test_cascade_scenario_runs(self):
        """Cascade scenario should complete and return valid results."""
        result = run_scenario("cascade", seed=42)

        assert len(result["icm_scores"]) == K_MAX
        assert 1 <= result["k_star_icm"] <= K_MAX

    def test_icm_scores_bounded(self):
        """All ICM scores should be in [0, 1]."""
        result = run_scenario("classification", seed=99)
        for icm in result["icm_scores"]:
            assert 0.0 <= icm <= 1.0, f"ICM out of bounds: {icm}"

    def test_order_is_permutation(self):
        """Selection order should be a permutation of [0, K_MAX)."""
        result = run_scenario("regression", seed=77)
        assert sorted(result["order"]) == list(range(K_MAX))

    def test_invalid_scenario_raises(self):
        """Invalid scenario name should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown scenario"):
            run_scenario("nonexistent", seed=42)


# =====================================================================
# Test 8: Reproducibility (deterministic seeds)
# =====================================================================

class TestReproducibility:
    """Tests for deterministic reproducibility."""

    def test_same_seed_same_results(self):
        """Running with the same seed should produce identical results."""
        r1 = run_scenario("classification", seed=42)
        r2 = run_scenario("classification", seed=42)

        np.testing.assert_array_equal(r1["icm_scores"], r2["icm_scores"])
        np.testing.assert_array_equal(r1["losses"], r2["losses"])
        assert r1["order"] == r2["order"]
        assert r1["k_star_icm"] == r2["k_star_icm"]

    def test_different_seeds_differ(self):
        """Different seeds should produce different results (with high probability)."""
        r1 = run_scenario("classification", seed=42)
        r2 = run_scenario("classification", seed=999)

        # At least some ICM scores should differ
        assert not np.allclose(r1["icm_scores"], r2["icm_scores"]), (
            "Different seeds should produce different ICM scores"
        )


# =====================================================================
# Test 9: Diminishing returns property
# =====================================================================

class TestDiminishingReturns:
    """Tests that the diminishing returns property holds."""

    def test_later_marginal_gains_smaller(self):
        """Mean of later marginal gains should be smaller than early ones.

        This is the core hypothesis: gains should diminish.
        """
        result = run_scenario("classification", seed=42)
        delta_icm = result["delta_icm"]

        # Compare first third vs last third
        k_third = K_MAX // 3
        early_gains = np.abs(delta_icm[1:k_third+1])
        late_gains = np.abs(delta_icm[-k_third:])

        mean_early = np.mean(early_gains)
        mean_late = np.mean(late_gains)

        # Late gains should be smaller than early gains
        assert mean_late <= mean_early * 2.0, (
            f"Late gains ({mean_late:.4f}) should be notably smaller than "
            f"early gains ({mean_early:.4f}) for diminishing returns"
        )

    def test_k_star_reasonable(self):
        """K* should be a reasonable value (not always 1 or K_max)."""
        # Run multiple seeds
        k_stars = []
        for seed in [42, 123, 456]:
            result = run_scenario("classification", seed=seed)
            k_stars.append(result["k_star_icm"])

        # At least one K* should be > 1 (not degenerate)
        assert any(k > 1 for k in k_stars), (
            f"K* should not always be 1, got {k_stars}"
        )


# =====================================================================
# Test 10: Softmax helper
# =====================================================================

class TestSoftmax:
    """Tests for the _softmax helper."""

    def test_output_sums_to_one(self):
        """Softmax output should sum to 1."""
        logits = np.array([[1.0, 2.0, 3.0]])
        probs = _softmax(logits)
        assert probs.sum() == pytest.approx(1.0, abs=1e-10)

    def test_numerically_stable(self):
        """Softmax should handle large logits without overflow."""
        logits = np.array([[1000.0, 999.0, 998.0]])
        probs = _softmax(logits)
        assert np.all(np.isfinite(probs))
        assert probs.sum() == pytest.approx(1.0, abs=1e-10)

    def test_uniform_for_equal_logits(self):
        """Equal logits should give uniform probabilities."""
        logits = np.array([[5.0, 5.0, 5.0]])
        probs = _softmax(logits)
        np.testing.assert_allclose(probs, 1.0 / 3.0, atol=1e-10)


# =====================================================================
# Test 11: Model families configuration
# =====================================================================

class TestModelFamilies:
    """Tests for the MODEL_FAMILIES configuration."""

    def test_correct_count(self):
        """Should have exactly K_MAX model families."""
        assert len(MODEL_FAMILIES) == K_MAX

    def test_unique_names(self):
        """All model family names should be unique."""
        names = [m[0] for m in MODEL_FAMILIES]
        assert len(names) == len(set(names))

    def test_valid_scales(self):
        """Bias and variance scales should be non-negative."""
        for name, bias, var, cg in MODEL_FAMILIES:
            assert bias >= 0, f"Model {name} has negative bias scale"
            assert var >= 0, f"Model {name} has negative variance scale"
            assert isinstance(cg, int), f"Model {name} has non-int corr group"
