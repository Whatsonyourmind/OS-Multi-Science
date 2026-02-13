"""Tests for Q8 ABM+ML Tipping-Point Detection experiment.

Covers: simulator, model families, ICM computation, detection metrics,
and end-to-end pipeline correctness.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pytest

from experiments.q8_tipping_detection import (
    build_erdos_renyi,
    simulate_ising_contagion,
    model_a_cascade_simulator,
    model_b_mean_field,
    model_c_network_logistic,
    model_d_autoregressive,
    model_e_random_forest,
    model_f_exponential_smoothing,
    run_all_models,
    compute_family_icm_timeseries,
    run_early_warning_pipeline,
    ABM_MODELS,
    ML_MODELS,
    ALL_MODELS,
)
from framework.icm import compute_icm_from_predictions
from framework.early_warning import (
    cusum_detector,
    page_hinkley_detector,
    evaluate_early_warning,
)


# ============================================================
# 1. Simulator Tests
# ============================================================

class TestErdosRenyi:
    """Test Erdos-Renyi network generation."""

    def test_symmetry(self):
        adj = build_erdos_renyi(50, 0.1, seed=42)
        np.testing.assert_array_equal(adj, adj.T)

    def test_no_self_loops(self):
        adj = build_erdos_renyi(50, 0.1, seed=42)
        assert np.all(np.diag(adj) == 0)

    def test_correct_shape(self):
        adj = build_erdos_renyi(75, 0.1, seed=42)
        assert adj.shape == (75, 75)

    def test_binary_values(self):
        adj = build_erdos_renyi(50, 0.1, seed=42)
        assert set(np.unique(adj)).issubset({0.0, 1.0})


class TestIsingContagion:
    """Test the Ising-like contagion simulator."""

    def test_returns_required_keys(self):
        sim = simulate_ising_contagion(n_nodes=30, n_steps=50, seed=42)
        required = [
            "adjacency", "state_history", "frac_distressed",
            "h_field", "tipping_step", "clustering",
            "local_order", "n_nodes", "n_steps",
        ]
        for key in required:
            assert key in sim, f"Missing key: {key}"

    def test_frac_distressed_bounded(self):
        sim = simulate_ising_contagion(n_nodes=50, n_steps=100, seed=42)
        assert np.all(sim["frac_distressed"] >= 0.0)
        assert np.all(sim["frac_distressed"] <= 1.0)

    def test_state_history_shape(self):
        sim = simulate_ising_contagion(n_nodes=40, n_steps=80, seed=42)
        assert sim["state_history"].shape == (80, 40)

    def test_h_field_ramps(self):
        sim = simulate_ising_contagion(
            n_nodes=30, n_steps=200,
            h_ramp_start=50, h_ramp_end=150, h_max=0.5,
            seed=42,
        )
        # Before ramp: h = 0
        assert sim["h_field"][0] == 0.0
        assert sim["h_field"][49] == 0.0
        # After ramp: h = h_max
        assert sim["h_field"][150] == pytest.approx(0.5, abs=1e-6)

    def test_deterministic_with_seed(self):
        s1 = simulate_ising_contagion(n_nodes=30, n_steps=50, seed=99)
        s2 = simulate_ising_contagion(n_nodes=30, n_steps=50, seed=99)
        np.testing.assert_array_equal(s1["frac_distressed"], s2["frac_distressed"])

    def test_tipping_step_valid(self):
        """Tipping step should be -1 (no tipping) or within [0, n_steps)."""
        sim = simulate_ising_contagion(n_nodes=50, n_steps=200, seed=42, h_max=0.6)
        tp = sim["tipping_step"]
        assert tp == -1 or (0 <= tp < 200)


# ============================================================
# 2. Model Family Tests
# ============================================================

class TestModelFamilies:
    """Test that all 6 models produce valid predictions."""

    @pytest.fixture
    def sim_data(self):
        return simulate_ising_contagion(
            n_nodes=50, edge_prob=0.08, n_steps=100,
            h_ramp_start=20, h_ramp_end=70, h_max=0.5,
            seed=42,
        )

    def test_model_a_shape_and_range(self, sim_data):
        pred = model_a_cascade_simulator(sim_data, seed=137)
        assert pred.shape == (100,)
        assert np.all(pred >= 0.0) and np.all(pred <= 1.0)

    def test_model_b_shape_and_range(self, sim_data):
        pred = model_b_mean_field(sim_data)
        assert pred.shape == (100,)
        assert np.all(pred >= 0.0) and np.all(pred <= 1.0)

    def test_model_c_shape_and_range(self, sim_data):
        pred = model_c_network_logistic(sim_data)
        assert pred.shape == (100,)
        assert np.all(pred >= 0.0) and np.all(pred <= 1.0)

    def test_model_d_shape_and_range(self, sim_data):
        pred = model_d_autoregressive(sim_data)
        assert pred.shape == (100,)
        assert np.all(pred >= 0.0) and np.all(pred <= 1.0)

    def test_model_e_shape_and_range(self, sim_data):
        pred = model_e_random_forest(sim_data, seed=42)
        assert pred.shape == (100,)
        assert np.all(pred >= 0.0) and np.all(pred <= 1.0)

    def test_model_f_shape_and_range(self, sim_data):
        pred = model_f_exponential_smoothing(sim_data)
        assert pred.shape == (100,)
        assert np.all(pred >= 0.0) and np.all(pred <= 1.0)

    def test_run_all_models_returns_six(self, sim_data):
        preds = run_all_models(sim_data, seed=42)
        assert len(preds) == 6
        for name in ALL_MODELS:
            assert name in preds
            assert preds[name].shape == (100,)

    def test_models_differ(self, sim_data):
        """Different models should produce different predictions."""
        preds = run_all_models(sim_data, seed=42)
        # At least some pairs should differ
        n_different = 0
        names = list(preds.keys())
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                if not np.allclose(preds[names[i]], preds[names[j]], atol=1e-3):
                    n_different += 1
        assert n_different > 0, "All models produced identical predictions"


# ============================================================
# 3. ICM Computation Tests
# ============================================================

class TestICMComputation:
    """Test ICM computation on model predictions."""

    @pytest.fixture
    def sim_and_preds(self):
        sim = simulate_ising_contagion(
            n_nodes=50, edge_prob=0.08, n_steps=100,
            h_ramp_start=20, h_ramp_end=70, h_max=0.5,
            seed=42,
        )
        preds = run_all_models(sim, seed=42)
        return sim, preds

    def test_icm_bounded_0_1(self, sim_and_preds):
        _, preds = sim_and_preds
        icm = compute_family_icm_timeseries(preds, ALL_MODELS, window_size=5)
        assert np.all(icm >= 0.0) and np.all(icm <= 1.0)

    def test_icm_shape(self, sim_and_preds):
        _, preds = sim_and_preds
        icm = compute_family_icm_timeseries(preds, ALL_MODELS, window_size=5)
        assert icm.shape == (100,)

    def test_abm_icm_vs_ml_icm(self, sim_and_preds):
        """ABM family ICM and ML family ICM should be computable independently."""
        _, preds = sim_and_preds
        abm_icm = compute_family_icm_timeseries(preds, ABM_MODELS, window_size=5)
        ml_icm = compute_family_icm_timeseries(preds, ML_MODELS, window_size=5)
        assert abm_icm.shape == (100,)
        assert ml_icm.shape == (100,)
        # They should differ (different models)
        assert not np.allclose(abm_icm, ml_icm, atol=1e-6)

    def test_icm_from_predictions_direct(self):
        """Test compute_icm_from_predictions with simple inputs."""
        preds = {
            "m1": np.array([0.25, 0.25, 0.25, 0.25]),
            "m2": np.array([0.25, 0.25, 0.25, 0.25]),
        }
        result = compute_icm_from_predictions(preds, distance_fn="hellinger")
        assert 0.0 <= result.icm_score <= 1.0
        # Identical distributions should have high agreement
        assert result.components.A == pytest.approx(1.0, abs=0.01)


# ============================================================
# 4. Detection and Metrics Tests
# ============================================================

class TestDetectionMetrics:
    """Test early warning detection and evaluation metrics."""

    def test_cusum_on_step_signal(self):
        """CUSUM should detect a step change."""
        signal = np.zeros(100)
        signal[50:] = 2.0
        changes, _ = cusum_detector(signal, threshold=3.0, drift=0.5)
        assert len(changes) > 0
        # First detection should be near the step
        assert changes[0] >= 50

    def test_page_hinkley_on_step_signal(self):
        """Page-Hinkley should detect a step change."""
        signal = np.zeros(100)
        signal[50:] = 2.0
        changes, _ = page_hinkley_detector(signal, threshold=5.0)
        assert len(changes) > 0

    def test_evaluate_early_warning_perfect(self):
        """Perfect detection: single detection at the right time."""
        result = evaluate_early_warning([45], [50], max_lead_time=10)
        assert result["true_positive_rate"] == 1.0
        assert result["lead_times"] == [5]

    def test_evaluate_early_warning_miss(self):
        """No detection means TPR=0."""
        result = evaluate_early_warning([], [50], max_lead_time=10)
        assert result["true_positive_rate"] == 0.0

    def test_evaluate_early_warning_false_alarm(self):
        """Detection far from true change is a false positive."""
        result = evaluate_early_warning([10], [50], max_lead_time=5)
        assert result["true_positive_rate"] == 0.0


# ============================================================
# 5. Early Warning Pipeline Tests
# ============================================================

class TestEarlyWarningPipeline:
    """Test the end-to-end early warning pipeline."""

    def test_pipeline_returns_required_keys(self):
        sim = simulate_ising_contagion(
            n_nodes=50, edge_prob=0.08, n_steps=100,
            h_ramp_start=20, h_ramp_end=70, h_max=0.5,
            seed=42,
        )
        preds = run_all_models(sim, seed=42)
        ew = run_early_warning_pipeline(
            preds, ALL_MODELS,
            tipping_step=sim["tipping_step"],
            n_steps=sim["n_steps"],
        )
        required = [
            "icm_scores", "z_signal", "z_standardized", "delta_icm",
            "var_preds", "pi_trend", "cusum_changes", "ph_changes",
            "cusum_eval", "ph_eval",
        ]
        for key in required:
            assert key in ew, f"Missing key: {key}"

    def test_pipeline_icm_shape(self):
        sim = simulate_ising_contagion(
            n_nodes=30, edge_prob=0.1, n_steps=80,
            seed=42,
        )
        preds = run_all_models(sim, seed=42)
        ew = run_early_warning_pipeline(
            preds, ABM_MODELS,
            tipping_step=sim["tipping_step"],
            n_steps=sim["n_steps"],
        )
        assert ew["icm_scores"].shape == (80,)
        assert ew["z_signal"].shape == (80,)

    def test_pipeline_subset_models(self):
        """Pipeline should work with ABM-only or ML-only subsets."""
        sim = simulate_ising_contagion(n_nodes=30, n_steps=60, seed=42)
        preds = run_all_models(sim, seed=42)

        ew_abm = run_early_warning_pipeline(
            preds, ABM_MODELS, tipping_step=sim["tipping_step"],
            n_steps=sim["n_steps"],
        )
        ew_ml = run_early_warning_pipeline(
            preds, ML_MODELS, tipping_step=sim["tipping_step"],
            n_steps=sim["n_steps"],
        )
        assert ew_abm["icm_scores"].shape == (60,)
        assert ew_ml["icm_scores"].shape == (60,)


# ============================================================
# 6. Integration / Smoke Tests
# ============================================================

class TestIntegration:
    """Integration tests for the full experiment flow."""

    def test_full_scenario_smoke(self):
        """Smoke test: full scenario should run without errors."""
        sim = simulate_ising_contagion(
            n_nodes=40, edge_prob=0.08, n_steps=100,
            h_ramp_start=20, h_ramp_end=70, h_max=0.6,
            coupling_J=1.5, temperature=0.5, seed=42,
        )
        preds = run_all_models(sim, seed=42)

        for family_name, model_list in [
            ("ABM", ABM_MODELS),
            ("ML", ML_MODELS),
            ("Combined", ALL_MODELS),
        ]:
            ew = run_early_warning_pipeline(
                preds, model_list,
                tipping_step=sim["tipping_step"],
                n_steps=sim["n_steps"],
            )
            # Eval should have valid structure
            assert "true_positive_rate" in ew["cusum_eval"]
            assert "true_positive_rate" in ew["ph_eval"]

    def test_model_lists_correct(self):
        """Verify model list constants are correct."""
        assert len(ABM_MODELS) == 3
        assert len(ML_MODELS) == 3
        assert len(ALL_MODELS) == 6
        assert set(ALL_MODELS) == set(ABM_MODELS + ML_MODELS)
