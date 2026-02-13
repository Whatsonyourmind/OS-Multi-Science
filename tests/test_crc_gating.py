"""Comprehensive tests for CRC gating module."""

from __future__ import annotations

import numpy as np
import numpy.testing as npt
import pytest

from framework.crc_gating import (
    fit_isotonic,
    conformalize,
    compute_re,
    decision_gate,
    risk_coverage_curve,
    calibrate_thresholds,
)
from framework.config import CRCConfig
from framework.types import DecisionAction


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def calibration_data(rng):
    """ICM scores and losses with clear negative correlation."""
    n = 200
    C = rng.random(n)
    # Higher ICM -> lower loss, with noise
    L = 1.0 - C + rng.standard_normal(n) * 0.1
    L = np.clip(L, 0.0, 2.0)
    return C, L


@pytest.fixture
def config():
    return CRCConfig(alpha=0.10, tau_hi=0.7, tau_lo=0.3)


# ============================================================
# fit_isotonic
# ============================================================

class TestFitIsotonic:

    def test_fitted_values_monotone_decreasing(self, calibration_data):
        C, L = calibration_data
        model = fit_isotonic(C, L)
        # Predict on sorted ICM values
        c_sorted = np.linspace(0.0, 1.0, 100)
        fitted = model.predict(c_sorted)
        # increasing=False => monotone non-increasing
        diffs = np.diff(fitted)
        assert np.all(diffs <= 1e-10)

    def test_fitted_model_is_callable(self, calibration_data):
        C, L = calibration_data
        model = fit_isotonic(C, L)
        result = model.predict(np.array([0.5]))
        assert len(result) == 1

    def test_high_icm_low_loss(self, calibration_data):
        C, L = calibration_data
        model = fit_isotonic(C, L)
        pred_high = model.predict(np.array([0.95]))[0]
        pred_low = model.predict(np.array([0.05]))[0]
        assert pred_high <= pred_low

    def test_single_point(self):
        C = np.array([0.5])
        L = np.array([0.3])
        model = fit_isotonic(C, L)
        result = model.predict(np.array([0.5]))
        assert result[0] == pytest.approx(0.3)

    def test_all_same_icm(self):
        """Edge case: all ICM values identical."""
        C = np.ones(20) * 0.5
        L = np.random.default_rng(42).random(20)
        model = fit_isotonic(C, L)
        result = model.predict(np.array([0.5]))
        assert np.isfinite(result[0])


# ============================================================
# conformalize
# ============================================================

class TestConformalize:

    def test_quantile_is_valid(self, calibration_data):
        C, L = calibration_data
        n = len(C)
        mid = n // 2
        model = fit_isotonic(C[:mid], L[:mid])
        g_alpha = conformalize(model, C[mid:], L[mid:], alpha=0.10)
        # g_alpha should return finite values
        result = g_alpha(0.5)
        assert np.isfinite(result)

    def test_coverage_on_calibration_set(self, calibration_data):
        """Conformalized function should provide >= 1-alpha coverage on cal set."""
        C, L = calibration_data
        n = len(C)
        mid = n // 2
        model = fit_isotonic(C[:mid], L[:mid])
        C_cal, L_cal = C[mid:], L[mid:]
        g_alpha = conformalize(model, C_cal, L_cal, alpha=0.10)
        # Check coverage: L_i <= g_alpha(C_i) for >= 90% of calibration data
        bounds = g_alpha(C_cal)
        covered = np.mean(L_cal <= bounds)
        assert covered >= 0.90 - 0.05  # allow small tolerance

    def test_g_alpha_returns_scalar_for_scalar_input(self, calibration_data):
        C, L = calibration_data
        model = fit_isotonic(C[:100], L[:100])
        g_alpha = conformalize(model, C[100:], L[100:], alpha=0.10)
        result = g_alpha(0.5)
        assert isinstance(result, float)

    def test_g_alpha_returns_array_for_array_input(self, calibration_data):
        C, L = calibration_data
        model = fit_isotonic(C[:100], L[:100])
        g_alpha = conformalize(model, C[100:], L[100:], alpha=0.10)
        result = g_alpha(np.array([0.3, 0.5, 0.7]))
        assert len(result) == 3

    def test_lower_alpha_gives_wider_bounds(self, calibration_data):
        """Smaller alpha (more coverage) should give larger risk bounds."""
        C, L = calibration_data
        model = fit_isotonic(C[:100], L[:100])
        g_05 = conformalize(model, C[100:], L[100:], alpha=0.05)
        g_20 = conformalize(model, C[100:], L[100:], alpha=0.20)
        # At alpha=0.05 the bound should be >= the bound at alpha=0.20
        assert g_05(0.5) >= g_20(0.5) - 1e-6


# ============================================================
# compute_re
# ============================================================

class TestComputeRe:

    def test_returns_float(self):
        g_alpha = lambda c: float(1.0 - c)
        re = compute_re(0.8, g_alpha)
        assert isinstance(re, float)

    def test_value_matches_g_alpha(self):
        g_alpha = lambda c: float(1.0 - c)
        assert compute_re(0.3, g_alpha) == pytest.approx(0.7)


# ============================================================
# decision_gate
# ============================================================

class TestDecisionGate:

    def test_act_when_icm_high(self, config):
        action = decision_gate(icm_score=0.9, re_score=0.1, config=config)
        assert action == DecisionAction.ACT

    def test_act_at_boundary_tau_hi(self, config):
        action = decision_gate(icm_score=0.7, re_score=0.1, config=config)
        assert action == DecisionAction.ACT

    def test_defer_when_icm_mid(self, config):
        action = decision_gate(icm_score=0.5, re_score=0.3, config=config)
        assert action == DecisionAction.DEFER

    def test_defer_at_boundary_tau_lo(self, config):
        action = decision_gate(icm_score=0.3, re_score=0.3, config=config)
        assert action == DecisionAction.DEFER

    def test_audit_when_icm_low(self, config):
        action = decision_gate(icm_score=0.1, re_score=0.5, config=config)
        assert action == DecisionAction.AUDIT

    def test_audit_just_below_tau_lo(self, config):
        action = decision_gate(icm_score=0.29, re_score=0.5, config=config)
        assert action == DecisionAction.AUDIT

    def test_decision_returns_enum(self, config):
        action = decision_gate(0.5, 0.3, config)
        assert isinstance(action, DecisionAction)


# ============================================================
# risk_coverage_curve
# ============================================================

class TestRiskCoverageCurve:

    def test_output_shapes_match(self, calibration_data):
        C, L = calibration_data
        result = risk_coverage_curve(C, L)
        n_thresholds = len(result["thresholds"])
        assert len(result["coverage"]) == n_thresholds
        assert len(result["avg_risk"]) == n_thresholds

    def test_default_50_thresholds(self, calibration_data):
        C, L = calibration_data
        result = risk_coverage_curve(C, L)
        assert len(result["thresholds"]) == 50

    def test_custom_thresholds(self, calibration_data):
        C, L = calibration_data
        thresholds = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        result = risk_coverage_curve(C, L, thresholds=thresholds)
        assert len(result["thresholds"]) == 5

    def test_coverage_bounded_0_1(self, calibration_data):
        C, L = calibration_data
        result = risk_coverage_curve(C, L)
        assert np.all(result["coverage"] >= 0.0)
        assert np.all(result["coverage"] <= 1.0)

    def test_coverage_monotone_decreasing(self, calibration_data):
        C, L = calibration_data
        thresholds = np.linspace(0.0, 1.0, 20)
        result = risk_coverage_curve(C, L, thresholds=thresholds)
        diffs = np.diff(result["coverage"])
        assert np.all(diffs <= 1e-12)

    def test_zero_threshold_full_coverage(self, calibration_data):
        C, L = calibration_data
        result = risk_coverage_curve(C, L, thresholds=np.array([0.0]))
        assert result["coverage"][0] == pytest.approx(1.0)

    def test_auc_bounded(self, calibration_data):
        C, L = calibration_data
        result = risk_coverage_curve(C, L)
        # AUC of coverage curve: integral of coverage over thresholds
        auc = np.trapezoid(result["coverage"], result["thresholds"])
        assert 0.0 <= auc <= 1.0


# ============================================================
# calibrate_thresholds
# ============================================================

class TestCalibrateThresholds:

    def test_tau_lo_less_than_tau_hi(self, calibration_data):
        C, L = calibration_data
        tau_hi, tau_lo = calibrate_thresholds(C, L, target_coverage=0.80, alpha=0.10)
        assert tau_lo <= tau_hi

    def test_reproducible_with_seed(self, calibration_data):
        C, L = calibration_data
        th1 = calibrate_thresholds(C, L, seed=42)
        th2 = calibrate_thresholds(C, L, seed=42)
        assert th1[0] == pytest.approx(th2[0])
        assert th1[1] == pytest.approx(th2[1])

    def test_different_seeds_may_differ(self, calibration_data):
        C, L = calibration_data
        th1 = calibrate_thresholds(C, L, seed=1)
        th2 = calibrate_thresholds(C, L, seed=2)
        # Not guaranteed to differ, but likely with different random splits
        # Just check they are valid
        assert th1[1] <= th1[0]
        assert th2[1] <= th2[0]

    def test_thresholds_in_valid_range(self, calibration_data):
        C, L = calibration_data
        tau_hi, tau_lo = calibrate_thresholds(C, L)
        assert 0.0 <= tau_lo <= 1.0
        assert 0.0 <= tau_hi <= 1.0

    def test_high_target_coverage(self, calibration_data):
        """High target coverage should push tau_lo lower."""
        C, L = calibration_data
        _, tau_lo_high = calibrate_thresholds(C, L, target_coverage=0.95)
        _, tau_lo_low = calibrate_thresholds(C, L, target_coverage=0.50)
        assert tau_lo_high <= tau_lo_low + 1e-6


# ============================================================
# Edge cases
# ============================================================

class TestEdgeCases:

    def test_all_high_icm(self):
        config = CRCConfig(tau_hi=0.7, tau_lo=0.3)
        action = decision_gate(1.0, 0.0, config)
        assert action == DecisionAction.ACT

    def test_all_low_icm(self):
        config = CRCConfig(tau_hi=0.7, tau_lo=0.3)
        action = decision_gate(0.0, 1.0, config)
        assert action == DecisionAction.AUDIT

    def test_single_sample_calibration(self):
        """With a very small calibration set, should not crash."""
        C = np.array([0.5, 0.6])
        L = np.array([0.3, 0.2])
        tau_hi, tau_lo = calibrate_thresholds(C, L, target_coverage=0.80, alpha=0.10)
        assert np.isfinite(tau_hi)
        assert np.isfinite(tau_lo)

    def test_risk_coverage_empty_region(self):
        """When threshold is 1.0, coverage should be 0 or near 0."""
        C = np.array([0.1, 0.2, 0.3])
        L = np.array([0.9, 0.8, 0.7])
        result = risk_coverage_curve(C, L, thresholds=np.array([1.0]))
        assert result["coverage"][0] == pytest.approx(0.0)

    def test_isotonic_with_uniform_losses(self):
        """All losses the same: isotonic should give constant fit."""
        C = np.linspace(0.0, 1.0, 50)
        L = np.ones(50) * 0.5
        model = fit_isotonic(C, L)
        preds = model.predict(np.array([0.0, 0.5, 1.0]))
        npt.assert_allclose(preds, 0.5, atol=1e-6)

    def test_decision_gate_exact_boundaries(self):
        config = CRCConfig(tau_hi=0.5, tau_lo=0.2)
        assert decision_gate(0.5, 0.0, config) == DecisionAction.ACT
        assert decision_gate(0.2, 0.0, config) == DecisionAction.DEFER
        assert decision_gate(0.19999, 0.0, config) == DecisionAction.AUDIT

    def test_conformalize_small_cal_set(self):
        """Conformalize with 2-point calibration set."""
        C_fit = np.array([0.2, 0.4, 0.6, 0.8])
        L_fit = np.array([0.8, 0.6, 0.4, 0.2])
        model = fit_isotonic(C_fit, L_fit)
        C_cal = np.array([0.3, 0.7])
        L_cal = np.array([0.7, 0.3])
        g_alpha = conformalize(model, C_cal, L_cal, alpha=0.10)
        result = g_alpha(0.5)
        assert np.isfinite(result)
