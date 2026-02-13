"""Comprehensive tests for MetaLearner (Q8) -- ICM weight optimization."""

from __future__ import annotations

import numpy as np
import pytest

from framework.config import ICMConfig
from framework.meta_learner import (
    MetaLearner,
    _WEIGHT_BOUNDS,
    _WEIGHT_NAMES,
    _weights_to_dict,
    _dict_to_vector,
    _make_config_from_weights,
)


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def learner() -> MetaLearner:
    """Fresh MetaLearner with default config."""
    return MetaLearner()


@pytest.fixture
def small_scenarios(learner: MetaLearner) -> list[dict]:
    """A small but diverse set of scenarios (10)."""
    return learner.generate_training_scenarios(n_scenarios=10, seed=123)


@pytest.fixture
def medium_scenarios(learner: MetaLearner) -> list[dict]:
    """A medium set of scenarios (20) for optimization tests."""
    return learner.generate_training_scenarios(n_scenarios=20, seed=42)


# ============================================================
# 1. Scenario Generation
# ============================================================

class TestScenarioGeneration:
    """Tests for generate_training_scenarios."""

    def test_correct_count(self, learner: MetaLearner):
        scenarios = learner.generate_training_scenarios(n_scenarios=15, seed=0)
        assert len(scenarios) == 15

    def test_scenario_structure(self, learner: MetaLearner):
        """Each scenario must contain required keys."""
        scenarios = learner.generate_training_scenarios(n_scenarios=5, seed=1)
        for s in scenarios:
            assert "predictions_dict" in s
            assert "loss" in s
            assert "label" in s
            assert isinstance(s["predictions_dict"], dict)
            assert len(s["predictions_dict"]) >= 3
            assert s["label"] in (0, 1)
            assert 0.0 <= s["loss"] <= 1.0 + 1e-6

    def test_label_balance(self, learner: MetaLearner):
        """Roughly half should be high-convergence (label=1)."""
        scenarios = learner.generate_training_scenarios(n_scenarios=20, seed=2)
        labels = [s["label"] for s in scenarios]
        n_high = sum(labels)
        # First half should be high convergence
        assert n_high == 10

    def test_predictions_are_valid_distributions(self, learner: MetaLearner):
        """Prediction arrays should sum to 1 and be non-negative."""
        scenarios = learner.generate_training_scenarios(n_scenarios=5, seed=3)
        for s in scenarios:
            for name, pred in s["predictions_dict"].items():
                assert np.all(pred >= 0), f"Negative predictions in {name}"
                assert pred.sum() == pytest.approx(1.0, abs=1e-6)

    def test_reproducibility(self, learner: MetaLearner):
        """Same seed should produce identical scenarios."""
        s1 = learner.generate_training_scenarios(n_scenarios=5, seed=99)
        s2 = learner.generate_training_scenarios(n_scenarios=5, seed=99)
        for a, b in zip(s1, s2):
            assert a["loss"] == b["loss"]
            assert a["label"] == b["label"]
            for key in a["predictions_dict"]:
                np.testing.assert_array_equal(
                    a["predictions_dict"][key],
                    b["predictions_dict"][key],
                )

    def test_has_optional_fields(self, learner: MetaLearner):
        """Scenarios should include intervals, signs, and features."""
        scenarios = learner.generate_training_scenarios(n_scenarios=4, seed=4)
        for s in scenarios:
            assert "intervals" in s
            assert "signs" in s
            assert "features" in s
            assert isinstance(s["intervals"], list)
            assert isinstance(s["signs"], np.ndarray)
            assert isinstance(s["features"], list)


# ============================================================
# 2. Weight Evaluation
# ============================================================

class TestEvaluateWeights:
    """Tests for evaluate_weights."""

    def test_returns_all_metrics(self, learner: MetaLearner, small_scenarios):
        weights = {"w_A": 0.35, "w_D": 0.15, "w_U": 0.25, "w_C": 0.10, "lam": 0.15}
        result = learner.evaluate_weights(weights, small_scenarios)
        assert "monotonicity" in result
        assert "coverage" in result
        assert "discrimination" in result
        assert "composite" in result

    def test_metrics_bounded(self, learner: MetaLearner, small_scenarios):
        """All metrics should be in [0, 1]."""
        weights = {"w_A": 0.30, "w_D": 0.20, "w_U": 0.20, "w_C": 0.15, "lam": 0.10}
        result = learner.evaluate_weights(weights, small_scenarios)
        for key in ["monotonicity", "coverage", "discrimination", "composite"]:
            assert 0.0 <= result[key] <= 1.0 + 1e-6, (
                f"{key} = {result[key]} out of [0, 1]"
            )

    def test_history_recorded(self, learner: MetaLearner, small_scenarios):
        """Each evaluation should append to history."""
        assert len(learner.history) == 0
        weights = {"w_A": 0.35, "w_D": 0.15, "w_U": 0.25, "w_C": 0.10, "lam": 0.15}
        learner.evaluate_weights(weights, small_scenarios)
        assert len(learner.history) == 1
        learner.evaluate_weights(weights, small_scenarios)
        assert len(learner.history) == 2

    def test_different_weights_different_scores(
        self, learner: MetaLearner, medium_scenarios
    ):
        """Substantially different weights should usually yield different composites."""
        w1 = {"w_A": 0.50, "w_D": 0.05, "w_U": 0.05, "w_C": 0.05, "lam": 0.05}
        w2 = {"w_A": 0.05, "w_D": 0.05, "w_U": 0.05, "w_C": 0.50, "lam": 0.30}
        r1 = learner.evaluate_weights(w1, medium_scenarios)
        r2 = learner.evaluate_weights(w2, medium_scenarios)
        # They could coincidentally be equal, but with these extremes, unlikely
        assert r1["composite"] != pytest.approx(r2["composite"], abs=1e-6)

    def test_composite_is_weighted_sum(self, learner: MetaLearner, small_scenarios):
        """composite = 0.4*mono + 0.3*disc + 0.3*cov."""
        weights = {"w_A": 0.25, "w_D": 0.25, "w_U": 0.25, "w_C": 0.10, "lam": 0.15}
        result = learner.evaluate_weights(weights, small_scenarios)
        expected = (
            0.4 * result["monotonicity"]
            + 0.3 * result["discrimination"]
            + 0.3 * result["coverage"]
        )
        assert result["composite"] == pytest.approx(expected, abs=1e-10)


# ============================================================
# 3. Grid Search
# ============================================================

class TestGridSearch:
    """Tests for grid_search."""

    def test_returns_required_keys(self, learner: MetaLearner, small_scenarios):
        result = learner.grid_search(small_scenarios, n_points=10, seed=7)
        assert "best_weights" in result
        assert "best_score" in result
        assert "all_results" in result

    def test_best_score_is_max(self, learner: MetaLearner, small_scenarios):
        """best_score should equal the max composite across all results."""
        result = learner.grid_search(small_scenarios, n_points=15, seed=8)
        all_composites = [r["composite"] for r in result["all_results"]]
        assert result["best_score"] == pytest.approx(max(all_composites), abs=1e-10)

    def test_weights_within_bounds(self, learner: MetaLearner, small_scenarios):
        """All sampled weights should respect bounds."""
        result = learner.grid_search(small_scenarios, n_points=20, seed=9)
        for entry in result["all_results"]:
            for name, (lo, hi) in _WEIGHT_BOUNDS.items():
                v = entry["weights"][name]
                assert lo - 1e-6 <= v <= hi + 1e-6, (
                    f"{name}={v} out of [{lo}, {hi}]"
                )

    def test_correct_number_of_results(self, learner: MetaLearner, small_scenarios):
        n = 12
        result = learner.grid_search(small_scenarios, n_points=n, seed=10)
        assert len(result["all_results"]) == n

    def test_best_weights_are_dict(self, learner: MetaLearner, small_scenarios):
        result = learner.grid_search(small_scenarios, n_points=5, seed=11)
        bw = result["best_weights"]
        assert isinstance(bw, dict)
        for name in _WEIGHT_NAMES:
            assert name in bw


# ============================================================
# 4. Optimization
# ============================================================

class TestOptimize:
    """Tests for scipy-based optimization."""

    def test_returns_required_keys(self, learner: MetaLearner, medium_scenarios):
        result = learner.optimize(medium_scenarios, n_restarts=2)
        assert "best_weights" in result
        assert "best_score" in result
        assert "optimization_result" in result

    def test_optimized_weights_within_bounds(
        self, learner: MetaLearner, medium_scenarios
    ):
        result = learner.optimize(medium_scenarios, n_restarts=2)
        for name, (lo, hi) in _WEIGHT_BOUNDS.items():
            v = result["best_weights"][name]
            assert lo - 1e-3 <= v <= hi + 1e-3, (
                f"Optimized {name}={v} out of [{lo}, {hi}]"
            )

    def test_optimize_score_positive(self, learner: MetaLearner, medium_scenarios):
        result = learner.optimize(medium_scenarios, n_restarts=2)
        assert result["best_score"] > 0.0

    def test_optimize_at_least_default_quality(
        self, learner: MetaLearner, medium_scenarios
    ):
        """Optimized weights should be at least as good as or close to defaults."""
        default_w = {"w_A": 0.35, "w_D": 0.15, "w_U": 0.25, "w_C": 0.10, "lam": 0.15}
        default_result = learner.evaluate_weights(default_w, medium_scenarios)

        opt_result = learner.optimize(medium_scenarios, n_restarts=3)
        # Allow small tolerance; optimizer might not always beat defaults
        assert opt_result["best_score"] >= default_result["composite"] - 0.05


# ============================================================
# 5. Cross-Validation
# ============================================================

class TestCrossValidate:
    """Tests for cross_validate."""

    def test_returns_required_keys(self, learner: MetaLearner, medium_scenarios):
        result = learner.cross_validate(medium_scenarios, n_folds=3)
        assert "mean_score" in result
        assert "std_score" in result
        assert "fold_scores" in result
        assert "best_weights" in result

    def test_fold_scores_count(self, learner: MetaLearner, medium_scenarios):
        n_folds = 4
        result = learner.cross_validate(medium_scenarios, n_folds=n_folds)
        assert len(result["fold_scores"]) == n_folds

    def test_mean_matches_fold_scores(self, learner: MetaLearner, medium_scenarios):
        result = learner.cross_validate(medium_scenarios, n_folds=3)
        expected_mean = np.mean(result["fold_scores"])
        assert result["mean_score"] == pytest.approx(expected_mean, abs=1e-10)

    def test_std_non_negative(self, learner: MetaLearner, medium_scenarios):
        result = learner.cross_validate(medium_scenarios, n_folds=3)
        assert result["std_score"] >= 0.0

    def test_too_few_scenarios_raises(self, learner: MetaLearner):
        tiny = learner.generate_training_scenarios(n_scenarios=1, seed=0)
        with pytest.raises(ValueError, match="at least 2"):
            learner.cross_validate(tiny, n_folds=2)


# ============================================================
# 6. Compare with Default
# ============================================================

class TestCompareWithDefault:
    """Tests for compare_with_default."""

    def test_returns_required_keys(self, learner: MetaLearner, small_scenarios):
        opt_w = {"w_A": 0.40, "w_D": 0.10, "w_U": 0.20, "w_C": 0.15, "lam": 0.10}
        result = learner.compare_with_default(small_scenarios, opt_w)
        assert "default_scores" in result
        assert "optimized_scores" in result
        assert "improvement_pct" in result

    def test_default_scores_match_config(self, learner: MetaLearner, small_scenarios):
        """Default scores should use the learner's config weights."""
        default_w = {
            "w_A": learner.config.w_A,
            "w_D": learner.config.w_D,
            "w_U": learner.config.w_U,
            "w_C": learner.config.w_C,
            "lam": learner.config.lam,
        }
        direct = learner.evaluate_weights(default_w, small_scenarios)
        comparison = learner.compare_with_default(small_scenarios, default_w)

        assert comparison["default_scores"]["composite"] == pytest.approx(
            direct["composite"], abs=1e-10
        )

    def test_same_weights_zero_improvement(self, learner: MetaLearner, small_scenarios):
        """Comparing default weights with themselves should yield 0% improvement."""
        default_w = {
            "w_A": learner.config.w_A,
            "w_D": learner.config.w_D,
            "w_U": learner.config.w_U,
            "w_C": learner.config.w_C,
            "lam": learner.config.lam,
        }
        result = learner.compare_with_default(small_scenarios, default_w)
        assert result["improvement_pct"]["composite"] == pytest.approx(0.0, abs=1e-6)

    def test_improvement_pct_has_all_keys(
        self, learner: MetaLearner, small_scenarios
    ):
        opt_w = {"w_A": 0.40, "w_D": 0.10, "w_U": 0.20, "w_C": 0.15, "lam": 0.10}
        result = learner.compare_with_default(small_scenarios, opt_w)
        for key in ["monotonicity", "coverage", "discrimination", "composite"]:
            assert key in result["improvement_pct"]


# ============================================================
# 7. Edge Cases and Utilities
# ============================================================

class TestEdgeCases:
    """Edge cases and utility function tests."""

    def test_weights_to_dict_roundtrip(self):
        w = {"w_A": 0.35, "w_D": 0.15, "w_U": 0.25, "w_C": 0.10, "lam": 0.15}
        vec = _dict_to_vector(w)
        assert vec.shape == (5,)
        recovered = _weights_to_dict(vec)
        for name in _WEIGHT_NAMES:
            assert recovered[name] == pytest.approx(w[name])

    def test_make_config_from_weights(self):
        w = {"w_A": 0.40, "w_D": 0.10, "w_U": 0.30, "w_C": 0.08, "lam": 0.20}
        cfg = _make_config_from_weights(w)
        assert cfg.w_A == pytest.approx(0.40)
        assert cfg.w_D == pytest.approx(0.10)
        assert cfg.w_U == pytest.approx(0.30)
        assert cfg.w_C == pytest.approx(0.08)
        assert cfg.lam == pytest.approx(0.20)

    def test_custom_config(self):
        """MetaLearner should respect a custom ICMConfig."""
        custom = ICMConfig(w_A=0.40, w_D=0.10, w_U=0.30, w_C=0.10, lam=0.10)
        ml = MetaLearner(config=custom)
        assert ml.config.w_A == 0.40
        assert ml.config.lam == 0.10

    def test_single_model_scenario(self):
        """Scenario with just one model should not crash."""
        ml = MetaLearner()
        scenario = {
            "predictions_dict": {"solo": np.array([0.5, 0.3, 0.2])},
            "loss": 0.5,
            "label": 1,
        }
        weights = {"w_A": 0.35, "w_D": 0.15, "w_U": 0.25, "w_C": 0.10, "lam": 0.15}
        result = ml.evaluate_weights(weights, [scenario, scenario])
        assert "composite" in result

    def test_all_same_label_scenarios(self):
        """When all scenarios have the same label, discrimination = 0.5."""
        ml = MetaLearner()
        rng = np.random.default_rng(55)
        scenarios = []
        for _ in range(6):
            base = rng.dirichlet(np.ones(3))
            preds = {
                f"m{j}": base + rng.normal(0, 0.02, 3)
                for j in range(3)
            }
            for k in preds:
                preds[k] = np.abs(preds[k])
                preds[k] /= preds[k].sum()
            scenarios.append({
                "predictions_dict": preds,
                "loss": 0.1,
                "label": 1,  # all the same label
            })
        weights = {"w_A": 0.35, "w_D": 0.15, "w_U": 0.25, "w_C": 0.10, "lam": 0.15}
        result = ml.evaluate_weights(weights, scenarios)
        assert result["discrimination"] == pytest.approx(0.5, abs=1e-6)

    def test_minimal_scenarios_for_coverage(self):
        """With very few scenarios, coverage should still return a value."""
        ml = MetaLearner()
        rng = np.random.default_rng(66)
        scenarios = []
        for i in range(4):
            base = rng.dirichlet(np.ones(3))
            preds = {f"m{j}": base + rng.normal(0, 0.05, 3) for j in range(3)}
            for k in preds:
                preds[k] = np.abs(preds[k])
                preds[k] /= preds[k].sum()
            scenarios.append({
                "predictions_dict": preds,
                "loss": rng.uniform(0.1, 0.9),
                "label": i % 2,
            })
        weights = {"w_A": 0.35, "w_D": 0.15, "w_U": 0.25, "w_C": 0.10, "lam": 0.15}
        result = ml.evaluate_weights(weights, scenarios)
        assert 0.0 <= result["coverage"] <= 1.0 + 1e-6

    def test_grid_search_reproducibility(self, learner: MetaLearner, small_scenarios):
        """Same seed should produce same grid search results."""
        r1 = learner.grid_search(small_scenarios, n_points=8, seed=77)
        r2 = learner.grid_search(small_scenarios, n_points=8, seed=77)
        assert r1["best_score"] == pytest.approx(r2["best_score"], abs=1e-10)
        for name in _WEIGHT_NAMES:
            assert r1["best_weights"][name] == pytest.approx(
                r2["best_weights"][name], abs=1e-10
            )
