"""Comprehensive tests for early warning system module."""

from __future__ import annotations

import numpy as np
import numpy.testing as npt
import pytest

from framework.early_warning import (
    compute_rolling_icm,
    compute_delta_icm,
    compute_prediction_variance,
    compute_z_signal,
    cusum_detector,
    page_hinkley_detector,
    evaluate_early_warning,
    placebo_test,
)
from framework.config import EarlyWarningConfig


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def config():
    return EarlyWarningConfig(
        window_size=10,
        a1=0.4,
        a2=0.4,
        a3=0.2,
        cusum_threshold=5.0,
        cusum_drift=0.5,
    )


@pytest.fixture
def stationary_signal():
    """Low-variance stationary signal around zero."""
    rng = np.random.default_rng(42)
    return rng.standard_normal(200) * 0.1


@pytest.fixture
def step_signal():
    """Signal with a clear step change at index 100."""
    sig = np.zeros(200)
    sig[100:] = 5.0
    return sig


# ============================================================
# compute_rolling_icm
# ============================================================

class TestRollingICM:

    def test_output_shape_matches_input(self, rng):
        icm = rng.random(50)
        result = compute_rolling_icm(icm, window_size=10)
        assert result.shape == icm.shape

    def test_rolling_values_bounded_when_input_bounded(self):
        """If input in [0,1] then rolling mean stays in [0,1]."""
        rng = np.random.default_rng(42)
        icm = rng.random(100)  # values in [0,1)
        result = compute_rolling_icm(icm, window_size=10)
        assert np.all(result >= 0.0 - 1e-12)
        assert np.all(result <= 1.0 + 1e-12)

    def test_window_1_equals_identity(self):
        icm = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = compute_rolling_icm(icm, window_size=1)
        npt.assert_allclose(result, icm)

    def test_full_window_last_entry_is_mean(self):
        icm = np.array([2.0, 4.0, 6.0, 8.0, 10.0])
        result = compute_rolling_icm(icm, window_size=5)
        # Last entry with window 5: mean of entire array
        assert result[-1] == pytest.approx(6.0)

    def test_growing_window_first_entries(self):
        icm = np.array([1.0, 3.0, 5.0, 7.0])
        result = compute_rolling_icm(icm, window_size=10)
        # Window > data, so all entries use growing window
        assert result[0] == pytest.approx(1.0)
        assert result[1] == pytest.approx(2.0)
        assert result[2] == pytest.approx(3.0)
        assert result[3] == pytest.approx(4.0)

    def test_constant_input_gives_constant_output(self):
        icm = np.ones(50) * 0.7
        result = compute_rolling_icm(icm, window_size=10)
        npt.assert_allclose(result, 0.7, atol=1e-12)

    def test_single_timestep(self):
        icm = np.array([0.5])
        result = compute_rolling_icm(icm, window_size=10)
        assert result[0] == pytest.approx(0.5)


# ============================================================
# compute_delta_icm
# ============================================================

class TestDeltaICM:

    def test_output_shape(self, rng):
        icm = rng.random(50)
        delta = compute_delta_icm(icm, window_size=5)
        assert delta.shape == icm.shape

    def test_first_element_is_zero(self):
        icm = np.array([1.0, 2.0, 3.0])
        delta = compute_delta_icm(icm, window_size=2)
        assert delta[0] == pytest.approx(0.0)

    def test_constant_input_gives_zero_delta(self):
        icm = np.ones(30) * 0.5
        delta = compute_delta_icm(icm, window_size=5)
        npt.assert_allclose(delta, 0.0, atol=1e-12)


# ============================================================
# compute_prediction_variance
# ============================================================

class TestPredictionVariance:

    def test_shape_matches_input(self):
        preds = {
            "m1": np.array([1.0, 2.0, 3.0]),
            "m2": np.array([1.5, 2.5, 3.5]),
        }
        var = compute_prediction_variance(preds)
        assert var.shape == (3,)

    def test_identical_models_zero_variance(self):
        v = np.array([1.0, 2.0, 3.0])
        preds = {"m1": v, "m2": v.copy(), "m3": v.copy()}
        var = compute_prediction_variance(preds)
        npt.assert_allclose(var, 0.0, atol=1e-12)

    def test_known_variance(self):
        preds = {
            "m1": np.array([0.0, 0.0]),
            "m2": np.array([2.0, 2.0]),
        }
        var = compute_prediction_variance(preds)
        # Var([0,2]) = 1.0
        npt.assert_allclose(var, [1.0, 1.0])

    def test_single_model_zero_variance(self):
        preds = {"m1": np.array([1.0, 2.0, 3.0])}
        var = compute_prediction_variance(preds)
        npt.assert_allclose(var, 0.0, atol=1e-12)

    def test_variance_nonnegative(self, rng):
        preds = {f"m{i}": rng.standard_normal(20) for i in range(5)}
        var = compute_prediction_variance(preds)
        assert np.all(var >= 0.0)


# ============================================================
# compute_z_signal
# ============================================================

class TestZSignal:

    def test_output_shape(self, config):
        T = 30
        delta = np.zeros(T)
        var_p = np.ones(T)
        pi = np.zeros(T)
        z = compute_z_signal(delta, var_p, pi, config)
        assert z.shape == (T,)

    def test_weights_sum_correctly(self):
        """Z should equal a1*(-delta) + a2*var + a3*pi."""
        cfg = EarlyWarningConfig(a1=0.3, a2=0.5, a3=0.2)
        delta = np.array([1.0, -1.0, 0.0])
        var_p = np.array([2.0, 2.0, 2.0])
        pi = np.array([0.5, 0.5, 0.5])
        z = compute_z_signal(delta, var_p, pi, cfg)
        expected = 0.3 * (-delta) + 0.5 * var_p + 0.2 * pi
        npt.assert_allclose(z, expected)

    def test_zero_inputs_give_zero_signal(self, config):
        T = 10
        z = compute_z_signal(np.zeros(T), np.zeros(T), np.zeros(T), config)
        npt.assert_allclose(z, 0.0, atol=1e-12)

    def test_negative_delta_increases_z(self, config):
        """A negative delta_icm (declining convergence) should increase Z."""
        T = 5
        delta_neg = np.array([-1.0] * T)
        delta_zero = np.zeros(T)
        var_p = np.zeros(T)
        pi = np.zeros(T)
        z_neg = compute_z_signal(delta_neg, var_p, pi, config)
        z_zero = compute_z_signal(delta_zero, var_p, pi, config)
        assert np.all(z_neg > z_zero)


# ============================================================
# cusum_detector
# ============================================================

class TestCUSUMDetector:

    def test_detects_known_changepoint(self, step_signal):
        cps, cusum_vals = cusum_detector(step_signal, threshold=3.0, drift=0.5)
        assert len(cps) > 0
        # First detection should be near the step at index 100
        assert any(95 <= cp <= 115 for cp in cps)

    def test_no_false_alarms_stationary(self, stationary_signal):
        cps, cusum_vals = cusum_detector(stationary_signal, threshold=10.0, drift=0.5)
        # Should produce very few or no detections on low-variance stationary signal
        assert len(cps) <= 2

    def test_cusum_values_shape(self, step_signal):
        cps, cusum_vals = cusum_detector(step_signal, threshold=3.0, drift=0.5)
        assert cusum_vals.shape == step_signal.shape

    def test_cusum_values_nonnegative(self, step_signal):
        _, cusum_vals = cusum_detector(step_signal, threshold=3.0, drift=0.5)
        assert np.all(cusum_vals >= 0.0 - 1e-12)

    def test_cusum_resets_after_detection(self, step_signal):
        cps, cusum_vals = cusum_detector(step_signal, threshold=3.0, drift=0.5)
        # After each detection, cusum should be reset to 0
        for cp in cps:
            assert cusum_vals[cp] == pytest.approx(0.0)

    def test_empty_signal(self):
        cps, cusum_vals = cusum_detector(np.array([]), threshold=3.0, drift=0.5)
        assert len(cps) == 0
        assert len(cusum_vals) == 0


# ============================================================
# page_hinkley_detector
# ============================================================

class TestPageHinkleyDetector:

    def test_detects_known_changepoint(self, step_signal):
        cps, stats = page_hinkley_detector(step_signal, threshold=10.0, alpha=0.005)
        assert len(cps) > 0
        assert any(95 <= cp <= 120 for cp in cps)

    def test_no_false_alarms_stationary(self, stationary_signal):
        cps, stats = page_hinkley_detector(
            stationary_signal, threshold=50.0, alpha=0.005,
        )
        assert len(cps) <= 2

    def test_stats_shape(self, step_signal):
        _, stats = page_hinkley_detector(step_signal, threshold=10.0, alpha=0.005)
        assert stats.shape == step_signal.shape

    def test_stats_nonnegative(self, step_signal):
        _, stats = page_hinkley_detector(step_signal, threshold=10.0, alpha=0.005)
        # PH statistic (m_t - M_t) is always >= 0 by definition
        assert np.all(stats >= 0.0 - 1e-12)

    def test_empty_signal(self):
        cps, stats = page_hinkley_detector(np.array([]), threshold=10.0, alpha=0.005)
        assert len(cps) == 0
        assert len(stats) == 0


# ============================================================
# evaluate_early_warning
# ============================================================

class TestEvaluateEarlyWarning:

    def test_returns_correct_keys(self):
        result = evaluate_early_warning(
            detected_changes=[5], true_changes=[10],
            max_lead_time=10, total_timesteps=100,
        )
        assert "informedness" in result
        assert "lead_times" in result
        assert "false_positive_rate" in result
        assert "true_positive_rate" in result
        assert "auroc" in result  # backward-compat alias

    def test_perfect_detection_informedness_one(self):
        """Detecting exactly at the right lead time => TPR=1, FPR=0 => informedness=1."""
        result = evaluate_early_warning(
            detected_changes=[8],
            true_changes=[10],
            max_lead_time=5,
            total_timesteps=100,
        )
        assert result["true_positive_rate"] == pytest.approx(1.0)
        assert result["false_positive_rate"] == pytest.approx(0.0)
        assert result["informedness"] == pytest.approx(1.0)

    def test_no_detection_gives_tpr_zero(self):
        result = evaluate_early_warning(
            detected_changes=[],
            true_changes=[50, 100],
            max_lead_time=10,
            total_timesteps=200,
        )
        assert result["true_positive_rate"] == pytest.approx(0.0)

    def test_all_false_positives(self):
        result = evaluate_early_warning(
            detected_changes=[10, 20, 30],
            true_changes=[200],
            max_lead_time=5,
            total_timesteps=300,
        )
        assert result["true_positive_rate"] == pytest.approx(0.0)
        assert result["false_positive_rate"] > 0.0

    def test_lead_times_populated(self):
        result = evaluate_early_warning(
            detected_changes=[7],
            true_changes=[10],
            max_lead_time=5,
            total_timesteps=50,
        )
        assert len(result["lead_times"]) == 1
        assert result["lead_times"][0] == 3  # 10 - 7

    def test_informedness_equals_auroc_alias(self):
        result = evaluate_early_warning(
            detected_changes=[5], true_changes=[10],
            max_lead_time=10, total_timesteps=50,
        )
        assert result["informedness"] == result["auroc"]

    def test_empty_true_changes(self):
        result = evaluate_early_warning(
            detected_changes=[5, 10],
            true_changes=[],
            max_lead_time=5,
            total_timesteps=50,
        )
        assert result["true_positive_rate"] == pytest.approx(0.0)

    def test_multiple_true_changes_partial_detection(self):
        result = evaluate_early_warning(
            detected_changes=[8],
            true_changes=[10, 50],
            max_lead_time=5,
            total_timesteps=100,
        )
        assert result["true_positive_rate"] == pytest.approx(0.5)

    def test_total_timesteps_none(self):
        """Should estimate total from data when total_timesteps=None."""
        result = evaluate_early_warning(
            detected_changes=[8],
            true_changes=[10],
            max_lead_time=5,
        )
        assert "informedness" in result


# ============================================================
# placebo_test
# ============================================================

class TestPlaceboTest:

    def test_no_alarms_on_stable(self):
        signal = np.zeros(100)
        stable = [(0, 100)]
        result = placebo_test(
            signal, stable,
            detector_fn=lambda s: cusum_detector(s, threshold=10.0, drift=0.5)[0],
        )
        assert result["false_alarm_rate"] == pytest.approx(0.0)

    def test_alarms_on_unstable(self):
        """Inject a step in a 'stable' period; detector should fire."""
        signal = np.zeros(100)
        signal[50:] = 10.0
        stable = [(0, 100)]
        result = placebo_test(
            signal, stable,
            detector_fn=lambda s: cusum_detector(s, threshold=3.0, drift=0.5)[0],
        )
        assert result["false_alarm_rate"] > 0.0


# ============================================================
# Edge cases
# ============================================================

class TestEdgeCases:

    def test_all_identical_icm_values(self):
        icm = np.ones(50) * 0.5
        rolling = compute_rolling_icm(icm, window_size=10)
        npt.assert_allclose(rolling, 0.5, atol=1e-12)

    def test_single_timestep_rolling(self):
        icm = np.array([0.8])
        result = compute_rolling_icm(icm, window_size=5)
        assert result[0] == pytest.approx(0.8)

    def test_window_larger_than_data(self):
        icm = np.array([0.1, 0.2, 0.3])
        result = compute_rolling_icm(icm, window_size=100)
        # Growing window: cumulative mean
        assert result[0] == pytest.approx(0.1)
        assert result[1] == pytest.approx(0.15)
        assert result[2] == pytest.approx(0.2)

    def test_cusum_single_timestep(self):
        sig = np.array([5.0])
        cps, vals = cusum_detector(sig, threshold=3.0, drift=0.5)
        assert len(cps) == 0  # only starts from t=1

    def test_page_hinkley_single_timestep(self):
        sig = np.array([5.0])
        cps, stats = page_hinkley_detector(sig, threshold=100.0, alpha=0.005)
        # Single timestep: PH won't exceed high threshold
        assert len(stats) == 1
