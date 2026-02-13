"""Tests for the statistical validation layer.

Covers:
- CRC: isotonic monotonicity, conformal coverage guarantee
- Anti-spurious: negative controls destroy genuine convergence
- Early warning: CUSUM detects known change-points
"""

from __future__ import annotations

import numpy as np
import pytest

from framework.config import (
    AntiSpuriousConfig,
    CRCConfig,
    EarlyWarningConfig,
)
from framework.types import DecisionAction

# ---------------------------------------------------------------------------
# CRC gating tests
# ---------------------------------------------------------------------------

class TestCRCGating:
    """Tests for framework.crc_gating."""

    def test_isotonic_fit_monotone_decreasing(self):
        """Fitted isotonic regression must be monotone decreasing."""
        from framework.crc_gating import fit_isotonic

        rng = np.random.RandomState(42)
        C = rng.uniform(0, 1, 200)
        # Higher convergence -> lower loss (with noise)
        L = 1.0 - C + rng.normal(0, 0.1, 200)

        model = fit_isotonic(C, L)
        C_grid = np.linspace(0, 1, 100)
        L_pred = model.predict(C_grid)

        # Check monotone decreasing (allowing ties)
        assert np.all(np.diff(L_pred) <= 1e-12), (
            "Isotonic fit should be monotone decreasing"
        )

    def test_conformal_coverage(self):
        """Conformalized bound should cover approximately (1 - alpha)."""
        from framework.crc_gating import conformalize, fit_isotonic

        rng = np.random.RandomState(0)
        n = 2000
        C = rng.uniform(0, 1, n)
        L = 1.0 - C + rng.normal(0, 0.15, n)
        alpha = 0.10

        # Split into fit / calibration / test
        idx = rng.permutation(n)
        fit_idx = idx[: n // 3]
        cal_idx = idx[n // 3 : 2 * n // 3]
        test_idx = idx[2 * n // 3 :]

        g = fit_isotonic(C[fit_idx], L[fit_idx])
        g_alpha = conformalize(g, C[cal_idx], L[cal_idx], alpha)

        # Check coverage on test set
        bounds = np.array([g_alpha(c) for c in C[test_idx]])
        covered = L[test_idx] <= bounds
        empirical_coverage = covered.mean()

        # Should be >= 1 - alpha (with some slack for finite samples)
        assert empirical_coverage >= (1 - alpha) - 0.05, (
            f"Coverage {empirical_coverage:.3f} < {1 - alpha - 0.05:.3f}"
        )

    def test_decision_gate_act(self):
        """ICM above tau_hi should yield ACT."""
        from framework.crc_gating import decision_gate

        config = CRCConfig(tau_hi=0.7, tau_lo=0.3)
        assert decision_gate(0.85, 0.1, config) == DecisionAction.ACT

    def test_decision_gate_defer(self):
        """ICM between tau_lo and tau_hi should yield DEFER."""
        from framework.crc_gating import decision_gate

        config = CRCConfig(tau_hi=0.7, tau_lo=0.3)
        assert decision_gate(0.50, 0.3, config) == DecisionAction.DEFER

    def test_decision_gate_audit(self):
        """ICM below tau_lo should yield AUDIT."""
        from framework.crc_gating import decision_gate

        config = CRCConfig(tau_hi=0.7, tau_lo=0.3)
        assert decision_gate(0.10, 0.8, config) == DecisionAction.AUDIT

    def test_risk_coverage_curve_shape(self):
        """Risk-coverage curve should have decreasing coverage."""
        from framework.crc_gating import risk_coverage_curve

        rng = np.random.RandomState(7)
        C = rng.uniform(0, 1, 500)
        L = 1.0 - C + rng.normal(0, 0.1, 500)

        result = risk_coverage_curve(C, L)
        cov = result["coverage"]

        # Coverage should be non-increasing as threshold rises
        assert np.all(np.diff(cov) <= 1e-12), (
            "Coverage should be non-increasing with rising threshold"
        )

    def test_calibrate_thresholds_ordering(self):
        """tau_lo should be <= tau_hi."""
        from framework.crc_gating import calibrate_thresholds

        rng = np.random.RandomState(99)
        C = rng.uniform(0, 1, 400)
        L = 1.0 - C + rng.normal(0, 0.1, 400)

        tau_hi, tau_lo = calibrate_thresholds(C, L, target_coverage=0.8)
        assert tau_lo <= tau_hi, f"tau_lo={tau_lo} > tau_hi={tau_hi}"


# ---------------------------------------------------------------------------
# Anti-spurious tests
# ---------------------------------------------------------------------------

class TestAntiSpurious:
    """Tests for framework.anti_spurious."""

    def test_negative_controls_destroy_convergence(self):
        """Negative controls should produce larger pairwise distances
        than genuine convergent predictions.
        """
        from framework.anti_spurious import (
            _mean_pairwise_distance,
            generate_negative_controls,
        )

        rng = np.random.RandomState(42)
        n_samples = 200
        labels = rng.uniform(0, 1, n_samples)
        features = rng.randn(n_samples, 5)

        # Models that genuinely converge on the truth
        predictions = np.stack([
            labels + rng.normal(0, 0.05, n_samples),
            labels + rng.normal(0, 0.05, n_samples),
            labels + rng.normal(0, 0.05, n_samples),
        ])

        D_genuine = _mean_pairwise_distance(predictions)

        controls = generate_negative_controls(
            predictions, labels, features, method="label_shuffle", n_controls=50,
        )
        D_controls = [_mean_pairwise_distance(c) for c in controls]
        D_mean_ctrl = np.mean(D_controls)

        # Genuine predictions should be much closer together
        assert D_genuine < D_mean_ctrl, (
            f"Genuine D={D_genuine:.4f} should be < control D={D_mean_ctrl:.4f}"
        )

    def test_normalize_convergence_range(self):
        """Normalized convergence must be in (0, 1]."""
        from framework.anti_spurious import normalize_convergence

        for D, D0 in [(0.5, 1.0), (0.0, 1.0), (2.0, 1.0), (1.0, 1.0)]:
            c = normalize_convergence(D, D0)
            assert 0.0 <= c <= 1.0, f"C({D}, {D0}) = {c} out of range"

    def test_fdr_correction_basic(self):
        """BH correction should reject clearly significant p-values."""
        from framework.anti_spurious import fdr_correction

        p = np.array([0.001, 0.005, 0.01, 0.5, 0.9])
        rejected, corrected = fdr_correction(p, alpha=0.05)

        assert rejected[0], "p=0.001 should be rejected"
        assert not rejected[-1], "p=0.9 should not be rejected"
        assert np.all(corrected >= p), "Corrected p should be >= raw p"

    def test_hsic_independent_signals(self):
        """HSIC p-value should be high for independent signals."""
        from framework.anti_spurious import hsic_test

        rng = np.random.RandomState(123)
        a = rng.randn(80)
        b = rng.randn(80)

        _, pval = hsic_test(a, b, n_permutations=200)
        assert pval > 0.05, f"Independent signals should have high p-value, got {pval}"

    def test_hsic_dependent_signals(self):
        """HSIC p-value should be low for dependent signals."""
        from framework.anti_spurious import hsic_test

        rng = np.random.RandomState(123)
        a = rng.randn(80)
        b = a ** 2 + rng.normal(0, 0.1, 80)  # nonlinear dependence

        _, pval = hsic_test(a, b, n_permutations=500)
        assert pval < 0.10, f"Dependent signals should have low p-value, got {pval}"

    def test_generate_report(self):
        """Full report pipeline should run without error."""
        from framework.anti_spurious import generate_anti_spurious_report

        rng = np.random.RandomState(55)
        n = 100
        labels = rng.randn(n)
        features = rng.randn(n, 3)
        preds = {
            "model_a": labels + rng.normal(0, 0.1, n),
            "model_b": labels + rng.normal(0, 0.1, n),
            "model_c": labels + rng.normal(0, 0.15, n),
        }

        config = AntiSpuriousConfig(
            n_permutations=50, n_negative_controls=20, fdr_level=0.05,
        )
        report = generate_anti_spurious_report(preds, labels, features, config)
        assert report.d0_baseline > 0
        assert 0 <= report.c_normalized <= 1


# ---------------------------------------------------------------------------
# Early warning tests
# ---------------------------------------------------------------------------

class TestEarlyWarning:
    """Tests for framework.early_warning."""

    def test_rolling_icm_smoothness(self):
        """Rolling mean should be smoother than raw signal."""
        from framework.early_warning import compute_rolling_icm

        rng = np.random.RandomState(1)
        raw = np.sin(np.linspace(0, 4 * np.pi, 200)) + rng.normal(0, 0.5, 200)
        smoothed = compute_rolling_icm(raw, window_size=20)

        raw_var = np.var(np.diff(raw))
        smooth_var = np.var(np.diff(smoothed))
        assert smooth_var < raw_var, "Rolling mean should reduce variance"

    def test_cusum_detects_known_changepoint(self):
        """CUSUM should detect a mean shift in a synthetic signal."""
        from framework.early_warning import cusum_detector

        rng = np.random.RandomState(42)
        n = 300
        signal = np.concatenate([
            rng.normal(0, 0.3, 200),   # stable regime
            rng.normal(2, 0.3, 100),    # shifted regime
        ])

        change_points, cusum_vals = cusum_detector(
            signal, threshold=5.0, drift=0.5,
        )

        # Should detect at least one change around t=200
        assert len(change_points) > 0, "CUSUM should detect the shift"
        first_detection = change_points[0]
        assert 180 <= first_detection <= 250, (
            f"First detection at t={first_detection}, expected near 200"
        )

    def test_page_hinkley_detects_changepoint(self):
        """Page-Hinkley test should also detect mean shifts."""
        from framework.early_warning import page_hinkley_detector

        rng = np.random.RandomState(42)
        signal = np.concatenate([
            rng.normal(0, 0.3, 200),
            rng.normal(2, 0.3, 100),
        ])

        change_points, stats = page_hinkley_detector(
            signal, threshold=10.0, alpha=0.005,
        )
        assert len(change_points) > 0, "Page-Hinkley should detect the shift"

    def test_z_signal_weights(self):
        """Z signal should reflect configured weights."""
        from framework.early_warning import compute_z_signal

        config = EarlyWarningConfig(a1=1.0, a2=0.0, a3=0.0)
        delta = np.array([0.5, -0.5, 0.0])
        var = np.array([1.0, 1.0, 1.0])
        pi = np.array([0.5, 0.5, 0.5])

        z = compute_z_signal(delta, var, pi, config)
        expected = np.array([-0.5, 0.5, 0.0])
        np.testing.assert_allclose(z, expected)

    def test_evaluate_early_warning(self):
        """Evaluation metrics should be correct for a simple case."""
        from framework.early_warning import evaluate_early_warning

        detected = [195, 198, 250]
        true_changes = [200]
        result = evaluate_early_warning(detected, true_changes, max_lead_time=10)

        assert result["true_positive_rate"] == 1.0
        assert len(result["lead_times"]) == 1

    def test_placebo_zero_alarms_on_stable(self):
        """Placebo test: detector should have low false alarm rate on flat signal."""
        from framework.early_warning import placebo_test

        signal = np.zeros(500)

        def no_alarm_detector(s):
            # CUSUM with high threshold on zero signal -> no alarms
            from framework.early_warning import cusum_detector
            cps, _ = cusum_detector(s, threshold=100.0, drift=0.5)
            return cps

        result = placebo_test(
            signal,
            stable_periods=[(0, 200), (300, 500)],
            detector_fn=no_alarm_detector,
        )
        assert result["false_alarm_rate"] == 0.0

    def test_prediction_variance_shape(self):
        """Prediction variance should have correct shape."""
        from framework.early_warning import compute_prediction_variance

        rng = np.random.RandomState(10)
        preds = {
            "m1": rng.randn(50),
            "m2": rng.randn(50),
            "m3": rng.randn(50),
        }
        var = compute_prediction_variance(preds)
        assert var.shape == (50,)
        assert np.all(var >= 0)
