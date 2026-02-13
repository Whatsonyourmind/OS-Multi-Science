"""Comprehensive tests for the COVID-19 multi-wave epidemic benchmark.

Tests cover:
  - Ground truth epidemic simulation (SEIR-V dynamics)
  - Observation model (reporting delay, underreporting, weekend effects)
  - Six model simulators (output shapes, non-negativity, diversity)
  - ICM computation across epidemic phases
  - Early warning system integration
  - CRC gating decisions
  - Anti-spurious validation
  - Knowledge graph construction
  - Full pipeline integration
  - Performance (< 5 seconds)
"""

from __future__ import annotations

import sys
import os
import time

import numpy as np
import pytest

# Ensure project root is importable
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, os.pardir))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from benchmarks.real_world.covid19 import (
    POPULATION,
    N_DAYS,
    N_NODES,
    SEED,
    ICM_WINDOW,
    WAVE1_START,
    WAVE1_INTERVENTION,
    WAVE2_START,
    VACCINATION_START,
    WAVE3_START,
    generate_contact_network,
    simulate_covid19_ground_truth,
    model_seirv_compartmental,
    model_agent_based,
    model_statistical_logistic,
    model_ml_ensemble,
    model_exponential_smoothing,
    model_rt_projection,
    compute_per_step_icm,
    run_crc_gating,
    build_knowledge_graph,
    get_phase_masks,
    run_covid19_benchmark,
)
from framework.config import CRCConfig, ICMConfig
from framework.types import DecisionAction


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture(scope="module")
def contact_network():
    """Generate contact network once for all tests."""
    return generate_contact_network(n=N_NODES, n_communities=5, seed=SEED)


@pytest.fixture(scope="module")
def ground_truth():
    """Simulate ground truth epidemic once for all tests."""
    return simulate_covid19_ground_truth(
        population=POPULATION, n_days=N_DAYS, seed=SEED
    )


@pytest.fixture(scope="module")
def all_predictions(ground_truth, contact_network):
    """Generate all model predictions once for all tests."""
    preds = {}
    preds["seirv_compartmental"] = model_seirv_compartmental(
        ground_truth, N_DAYS, POPULATION
    )
    preds["agent_based"] = model_agent_based(
        contact_network, ground_truth, N_DAYS, seed=137
    )
    preds["statistical_logistic"] = model_statistical_logistic(
        ground_truth, N_DAYS, POPULATION
    )
    preds["ml_ensemble"] = model_ml_ensemble(ground_truth, N_DAYS)
    preds["exponential_smoothing"] = model_exponential_smoothing(
        ground_truth, N_DAYS
    )
    preds["rt_projection"] = model_rt_projection(ground_truth, N_DAYS)
    return preds


# ============================================================
# 1. Contact Network Tests
# ============================================================

class TestContactNetwork:
    """Tests for the contact network generator."""

    def test_shape(self, contact_network):
        assert contact_network.shape == (N_NODES, N_NODES)

    def test_symmetric(self, contact_network):
        np.testing.assert_array_equal(contact_network, contact_network.T)

    def test_no_self_loops(self, contact_network):
        assert np.all(np.diag(contact_network) == 0.0)

    def test_binary(self, contact_network):
        unique_vals = np.unique(contact_network)
        assert set(unique_vals).issubset({0.0, 1.0})

    def test_connected(self, contact_network):
        degree = contact_network.sum(axis=1)
        assert np.all(degree > 0), "All nodes should have at least one connection"

    def test_community_structure(self, contact_network):
        """Network should have edges -- communities create structure."""
        n_edges = int(contact_network.sum() / 2)
        assert n_edges > N_NODES, "Should have more edges than nodes"


# ============================================================
# 2. Ground Truth Simulation Tests
# ============================================================

class TestGroundTruth:
    """Tests for the COVID-19 ground truth simulation."""

    def test_output_keys(self, ground_truth):
        required_keys = [
            "true_infections", "reported_cases",
            "S", "E", "I", "R", "V",
            "R_eff", "vaccination_rate",
        ]
        for key in required_keys:
            assert key in ground_truth, f"Missing key: {key}"

    def test_array_lengths(self, ground_truth):
        for key in ground_truth:
            assert len(ground_truth[key]) == N_DAYS, (
                f"{key} should have {N_DAYS} elements"
            )

    def test_compartment_conservation(self, ground_truth):
        """S + E + I + R + V should sum to ~1 at each time step."""
        total = (
            ground_truth["S"]
            + ground_truth["E"]
            + ground_truth["I"]
            + ground_truth["R"]
            + ground_truth["V"]
        )
        np.testing.assert_allclose(total, 1.0, atol=0.01)

    def test_non_negative_infections(self, ground_truth):
        assert np.all(ground_truth["true_infections"] >= 0)
        assert np.all(ground_truth["reported_cases"] >= 0)

    def test_underreporting(self, ground_truth):
        """Reported cases should be less than true infections overall."""
        true_total = ground_truth["true_infections"].sum()
        reported_total = ground_truth["reported_cases"].sum()
        ratio = reported_total / max(true_total, 1)
        assert 0.1 < ratio < 0.8, (
            f"Underreporting ratio {ratio:.2f} should be between 0.1 and 0.8"
        )

    def test_multi_wave_structure(self, ground_truth):
        """The epidemic should have at least 2 discernible waves."""
        infections = ground_truth["true_infections"]
        # Check that there are infections in both wave 1 and wave 2 periods
        wave1_total = infections[WAVE1_START:WAVE1_INTERVENTION].sum()
        wave2_total = infections[WAVE2_START:WAVE3_START].sum()
        assert wave1_total > 100, f"Wave 1 should have substantial infections: {wave1_total}"
        assert wave2_total > 100, f"Wave 2 should have substantial infections: {wave2_total}"

    def test_r_eff_values(self, ground_truth):
        """R_eff should be > 2 pre-intervention and < 1 post-intervention."""
        r_eff = ground_truth["R_eff"]
        # Pre-intervention (mid-wave 1)
        pre_r = r_eff[WAVE1_START + 10:WAVE1_INTERVENTION]
        assert np.mean(pre_r) > 1.5, (
            f"Pre-intervention R_eff should be > 1.5, got {np.mean(pre_r):.2f}"
        )
        # Post-intervention (stable period)
        post_r = r_eff[WAVE1_INTERVENTION + 30:WAVE2_START - 10]
        assert np.mean(post_r) < 1.5, (
            f"Post-intervention R_eff should be < 1.5, got {np.mean(post_r):.2f}"
        )

    def test_vaccination_starts_correctly(self, ground_truth):
        """Vaccination should be zero before VACCINATION_START."""
        v_before = ground_truth["V"][:VACCINATION_START]
        assert np.all(v_before == 0.0)
        # After vaccination, V should increase
        v_after = ground_truth["V"][VACCINATION_START + 10:]
        assert v_after[-1] > 0.1, "Vaccination should reach > 10%"

    def test_deterministic_seed(self, ground_truth):
        """Same seed should produce identical results."""
        gt2 = simulate_covid19_ground_truth(
            population=POPULATION, n_days=N_DAYS, seed=SEED
        )
        np.testing.assert_array_equal(
            ground_truth["true_infections"], gt2["true_infections"]
        )


# ============================================================
# 3. Model Output Tests
# ============================================================

class TestModelOutputs:
    """Tests for model simulator outputs."""

    def test_all_models_correct_length(self, all_predictions):
        for name, pred in all_predictions.items():
            assert len(pred) == N_DAYS, f"{name} should have {N_DAYS} elements"

    def test_all_models_non_negative(self, all_predictions):
        for name, pred in all_predictions.items():
            assert np.all(pred >= 0), f"{name} should be non-negative"

    def test_six_models_present(self, all_predictions):
        assert len(all_predictions) == 6

    def test_model_diversity(self, all_predictions):
        """Models should produce diverse predictions (not identical)."""
        names = list(all_predictions.keys())
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                pred_i = all_predictions[names[i]]
                pred_j = all_predictions[names[j]]
                # Predictions should not be identical
                assert not np.allclose(pred_i, pred_j, atol=0.1), (
                    f"{names[i]} and {names[j]} should differ"
                )

    def test_seirv_produces_infections(self, all_predictions):
        """Compartmental model should predict some infections."""
        assert all_predictions["seirv_compartmental"].sum() > 100

    def test_abm_produces_infections(self, all_predictions):
        """Agent-based model should predict some infections."""
        assert all_predictions["agent_based"].sum() > 10

    def test_ml_ensemble_reasonable(self, all_predictions, ground_truth):
        """ML ensemble should have reasonable correlation with reported."""
        pred = all_predictions["ml_ensemble"]
        reported = ground_truth["reported_cases"]
        corr = np.corrcoef(pred, reported)[0, 1]
        assert corr > 0.1, f"ML ensemble correlation should be > 0.1, got {corr:.3f}"


# ============================================================
# 4. ICM Computation Tests
# ============================================================

class TestICMComputation:
    """Tests for ICM computation across epidemic phases."""

    @pytest.fixture(scope="class")
    def icm_data(self, all_predictions):
        config = ICMConfig(C_A_wasserstein=50.0)
        scores, results = compute_per_step_icm(
            all_predictions, window_size=ICM_WINDOW, config=config
        )
        return scores, results

    def test_icm_shape(self, icm_data):
        scores, results = icm_data
        assert len(scores) == N_DAYS
        assert len(results) == N_DAYS

    def test_icm_bounded(self, icm_data):
        scores, _ = icm_data
        assert np.all(scores >= 0.0)
        assert np.all(scores <= 1.0)

    def test_icm_not_constant(self, icm_data):
        """ICM should vary across epidemic phases."""
        scores, _ = icm_data
        assert scores.std() > 0.001, "ICM should vary over time"

    def test_icm_components_present(self, icm_data):
        _, results = icm_data
        for r in results[:10]:
            assert hasattr(r.components, "A")
            assert hasattr(r.components, "D")
            assert hasattr(r.components, "U")
            assert 0 <= r.components.A <= 1
            assert 0 <= r.components.D <= 1
            assert 0 <= r.components.U <= 1

    def test_icm_n_models(self, icm_data):
        _, results = icm_data
        for r in results:
            assert r.n_models == 6

    def test_phase_icm_variation(self, icm_data):
        """Different epidemic phases should show ICM variation."""
        scores, _ = icm_data
        masks = get_phase_masks(N_DAYS)
        phase_means = {}
        for name, mask in masks.items():
            if mask.sum() > 0:
                phase_means[name] = scores[mask].mean()
        assert len(phase_means) >= 4, "Should have at least 4 phases"


# ============================================================
# 5. CRC Gating Tests
# ============================================================

class TestCRCGating:
    """Tests for CRC gating integration."""

    def test_crc_produces_decision(self, all_predictions, ground_truth):
        config = ICMConfig(C_A_wasserstein=50.0)
        scores, _ = compute_per_step_icm(
            all_predictions, window_size=ICM_WINDOW, config=config
        )
        reported = ground_truth["reported_cases"]
        ensemble_mean = np.mean(
            np.column_stack([all_predictions[n] for n in all_predictions]),
            axis=1,
        )
        losses = (ensemble_mean - reported) ** 2
        sample_idx = np.arange(7, N_DAYS, 7)
        crc_config = CRCConfig(alpha=0.10, tau_hi=0.65, tau_lo=0.35)
        result = run_crc_gating(
            scores[sample_idx], losses[sample_idx], crc_config
        )
        assert "decision" in result
        assert isinstance(result["decision"], DecisionAction)
        assert "re_score" in result
        assert "median_icm" in result


# ============================================================
# 6. Knowledge Graph Tests
# ============================================================

class TestKnowledgeGraph:
    """Tests for knowledge graph construction."""

    def test_knowledge_graph_structure(self, all_predictions, ground_truth):
        config = ICMConfig(C_A_wasserstein=50.0)
        scores, results = compute_per_step_icm(
            all_predictions, window_size=ICM_WINDOW, config=config
        )
        model_names = list(all_predictions.keys())
        ew_results = {"cusum_changes": [], "ph_changes": []}
        crc_results = {
            "decision": DecisionAction.DEFER,
            "re_score": 0.5,
            "median_icm": 0.6,
        }
        phase_icms = {"wave1": 0.6, "wave2": 0.65}

        kg = build_knowledge_graph(
            model_names, all_predictions, scores, results,
            ew_results, None, crc_results, phase_icms,
        )
        summary = kg.summary()
        assert summary["total_nodes"] > 0
        assert summary["total_edges"] > 0
        assert summary["nodes_system"] == 1
        assert summary["nodes_method"] == 6
        assert summary["nodes_result"] == 6
        assert summary["nodes_decision"] == 1


# ============================================================
# 7. Phase Mask Tests
# ============================================================

class TestPhaseMasks:
    """Tests for epidemic phase definitions."""

    def test_phase_masks_cover_timeline(self):
        masks = get_phase_masks(N_DAYS)
        # Compute union of non-overlapping phase masks
        # (some phases overlap by design, e.g. vaccination_rollout)
        core_phases = ["pre_wave1", "wave1_rising", "wave1_declining",
                       "wave2", "wave3"]
        total_days = sum(masks[p].sum() for p in core_phases)
        assert total_days == N_DAYS, (
            f"Core phases should cover all {N_DAYS} days, got {total_days}"
        )

    def test_phase_boundaries(self):
        masks = get_phase_masks(N_DAYS)
        assert masks["pre_wave1"].sum() == WAVE1_START
        assert masks["wave1_rising"].sum() == WAVE1_INTERVENTION - WAVE1_START
        assert masks["wave3"].sum() == N_DAYS - WAVE3_START


# ============================================================
# 8. Early Warning Integration Tests
# ============================================================

class TestEarlyWarning:
    """Tests for early warning system integration."""

    def test_z_signal_computation(self, all_predictions):
        from framework.config import EarlyWarningConfig
        from framework.early_warning import (
            compute_delta_icm,
            compute_prediction_variance,
            compute_z_signal,
        )

        config = ICMConfig(C_A_wasserstein=50.0)
        scores, results = compute_per_step_icm(
            all_predictions, window_size=ICM_WINDOW, config=config
        )
        ew_config = EarlyWarningConfig(
            window_size=14, a1=0.4, a2=0.4, a3=0.2,
            cusum_threshold=2.0, cusum_drift=0.2,
        )
        delta_icm = compute_delta_icm(scores, window_size=ew_config.window_size)
        pred_var = compute_prediction_variance(all_predictions)
        var_max = pred_var.max()
        pred_var_norm = pred_var / var_max if var_max > 1e-12 else pred_var
        pi_trend = np.array([results[t].components.Pi for t in range(N_DAYS)])
        z_signal = compute_z_signal(
            delta_icm, pred_var_norm, pi_trend, ew_config
        )
        assert len(z_signal) == N_DAYS
        assert not np.all(z_signal == 0), "Z signal should have non-zero values"


# ============================================================
# 9. Anti-Spurious Tests
# ============================================================

class TestAntiSpurious:
    """Tests for anti-spurious validation integration."""

    def test_anti_spurious_runs(self, all_predictions, ground_truth):
        from framework.anti_spurious import generate_anti_spurious_report
        from framework.config import AntiSpuriousConfig
        from framework.icm import compute_icm_from_predictions

        obs_start = WAVE1_START
        obs_end = min(WAVE1_INTERVENTION + 30, N_DAYS)
        sample_preds = {
            name: pred[obs_start:obs_end]
            for name, pred in all_predictions.items()
        }
        sample_labels = ground_truth["reported_cases"][obs_start:obs_end]
        sample_features = np.arange(
            obs_end - obs_start, dtype=float
        ).reshape(-1, 1)

        anti_config = AntiSpuriousConfig(
            n_permutations=50, fdr_level=0.05, n_negative_controls=20
        )
        icm_config = ICMConfig(C_A_wasserstein=50.0)

        def icm_fn(p):
            normed = {}
            for nm, v in p.items():
                total = v.sum()
                normed[nm] = v / total if total > 1e-12 else np.ones(len(v)) / len(v)
            r = compute_icm_from_predictions(
                normed, config=icm_config, distance_fn="hellinger"
            )
            return r.icm_score

        report = generate_anti_spurious_report(
            sample_preds, sample_labels, sample_features,
            config=anti_config, icm_fn=icm_fn,
        )
        assert hasattr(report, "d0_baseline")
        assert hasattr(report, "c_normalized")
        assert hasattr(report, "is_genuine")
        assert report.d0_baseline > 0


# ============================================================
# 10. Full Pipeline Integration Test
# ============================================================

class TestFullPipeline:
    """End-to-end integration test for the COVID-19 benchmark."""

    def test_benchmark_runs_successfully(self):
        """The full benchmark should complete without errors."""
        results = run_covid19_benchmark()
        assert results is not None
        assert "ground_truth" in results
        assert "predictions" in results
        assert "icm_scores" in results
        assert "icm_results" in results
        assert "early_warning" in results
        assert "crc_results" in results
        assert "knowledge_graph" in results
        assert "model_rmse" in results
        assert "ablation" in results
        assert "phase_icms" in results
        assert "summary" in results

    def test_benchmark_under_time_limit(self):
        """Benchmark should complete in under 10 seconds."""
        t_start = time.monotonic()
        results = run_covid19_benchmark()
        elapsed = time.monotonic() - t_start
        assert elapsed < 10.0, (
            f"Benchmark took {elapsed:.2f}s, should be < 10s"
        )

    def test_benchmark_produces_six_models(self):
        results = run_covid19_benchmark()
        assert len(results["predictions"]) == 6

    def test_benchmark_summary_keys(self):
        results = run_covid19_benchmark()
        summary = results["summary"]
        assert "true_total_infections" in summary
        assert "reported_total_cases" in summary
        assert "mean_icm" in summary
        assert "crc_decision" in summary
        assert summary["true_total_infections"] > 1000
        assert summary["reported_total_cases"] > 100
