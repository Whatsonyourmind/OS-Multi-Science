"""Tests for the SOA benchmark suite.

30+ tests covering smoke tests, per-experiment validation,
output format checks, ICM score ranges, edge cases, and
report generation.
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from benchmarks.datasets import load_dataset, list_classification_datasets, list_regression_datasets
from benchmarks.model_zoo import (
    build_classification_zoo,
    build_regression_zoo,
    train_zoo,
    collect_predictions_classification,
    collect_predictions_regression,
)
from framework.config import ICMConfig, CRCConfig

# Module under test
from benchmarks.soa_benchmark import (
    run_soa_benchmark,
    experiment_prediction_quality,
    experiment_uncertainty_quantification,
    experiment_error_detection,
    experiment_diversity_assessment,
    experiment_score_range,
    _icm_weighted_ensemble_classification,
    _icm_weighted_ensemble_regression,
    _compute_per_model_icm_classification,
    _compute_per_model_icm_regression,
    _compute_per_sample_icm_classification,
    _compute_per_sample_error_classification,
    _ensemble_entropy,
    _icm_prediction_sets_classification,
)

# Suppress warnings during tests for cleaner output
pytestmark = pytest.mark.filterwarnings("ignore")


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture(scope="module")
def iris_data():
    """Load iris dataset (small, 3-class classification)."""
    return load_dataset("iris", seed=42)


@pytest.fixture(scope="module")
def breast_cancer_data():
    """Load breast cancer dataset (binary classification)."""
    return load_dataset("breast_cancer", seed=42)


@pytest.fixture(scope="module")
def diabetes_data():
    """Load diabetes dataset (regression)."""
    return load_dataset("diabetes", seed=42)


@pytest.fixture(scope="module")
def trained_clf_models_iris(iris_data):
    """Train classification zoo on iris."""
    X_train, X_test, y_train, y_test = iris_data
    models = build_classification_zoo(seed=42)
    train_zoo(models, X_train, y_train)
    return models


@pytest.fixture(scope="module")
def iris_preds(trained_clf_models_iris, iris_data):
    """Collect predictions on iris test set."""
    X_train, X_test, y_train, y_test = iris_data
    return collect_predictions_classification(trained_clf_models_iris, X_test)


@pytest.fixture(scope="module")
def wide_config():
    return ICMConfig.wide_range_preset()


# ============================================================
# Smoke tests
# ============================================================

class TestSmokeTests:
    """Basic smoke tests that the benchmark runs without errors."""

    def test_full_benchmark_runs(self):
        """Smoke test: full benchmark completes without error."""
        results = run_soa_benchmark(seed=42, verbose=False)
        assert isinstance(results, dict)
        assert "dataset_results" in results
        assert "aggregate_results" in results
        assert "icm_analysis" in results
        assert "summary" in results
        assert "elapsed_seconds" in results

    def test_benchmark_runtime_under_limit(self):
        """Runtime should be under 120 seconds (generous limit)."""
        results = run_soa_benchmark(seed=42, verbose=False)
        assert results["elapsed_seconds"] < 120.0

    def test_benchmark_produces_all_datasets(self):
        """All datasets should appear in results."""
        results = run_soa_benchmark(seed=42, verbose=False)
        clf = list_classification_datasets()
        reg = list_regression_datasets()
        for ds in clf + reg:
            assert ds in results["dataset_results"], f"Missing dataset: {ds}"

    def test_summary_is_nonempty_string(self):
        """Summary should be a non-empty string."""
        results = run_soa_benchmark(seed=42, verbose=False)
        assert isinstance(results["summary"], str)
        assert len(results["summary"]) > 50


# ============================================================
# Experiment 1: Prediction Quality
# ============================================================

class TestExperiment1:
    """Tests for Experiment 1: Prediction Quality."""

    def test_classification_results_table(self, iris_data):
        """Exp1 on classification produces a valid results table."""
        X_train, X_test, y_train, y_test = iris_data
        exp1 = experiment_prediction_quality(
            "iris", X_train, X_test, y_train, y_test,
            task="classification", seed=42,
        )
        assert isinstance(exp1["results_table"], pd.DataFrame)
        assert len(exp1["results_table"]) >= 4
        assert "method" in exp1["results_table"].columns
        assert "accuracy" in exp1["results_table"].columns

    def test_classification_icm_weighted_row(self, iris_data):
        """ICM-Weighted should appear as a method in exp1 results."""
        X_train, X_test, y_train, y_test = iris_data
        exp1 = experiment_prediction_quality(
            "iris", X_train, X_test, y_train, y_test,
            task="classification", seed=42,
        )
        methods = list(exp1["results_table"]["method"])
        assert "ICM-Weighted" in methods

    def test_classification_accuracy_in_range(self, iris_data):
        """Accuracy values should be in [0, 1]."""
        X_train, X_test, y_train, y_test = iris_data
        exp1 = experiment_prediction_quality(
            "iris", X_train, X_test, y_train, y_test,
            task="classification", seed=42,
        )
        accs = exp1["results_table"]["accuracy"].dropna()
        assert all(0.0 <= a <= 1.0 for a in accs)

    def test_regression_results_table(self, diabetes_data):
        """Exp1 on regression produces valid results."""
        X_train, X_test, y_train, y_test = diabetes_data
        exp1 = experiment_prediction_quality(
            "diabetes", X_train, X_test, y_train, y_test,
            task="regression", seed=42,
        )
        assert isinstance(exp1["results_table"], pd.DataFrame)
        assert "RMSE" in exp1["results_table"].columns
        assert len(exp1["results_table"]) >= 2

    def test_regression_rmse_positive(self, diabetes_data):
        """RMSE should be positive."""
        X_train, X_test, y_train, y_test = diabetes_data
        exp1 = experiment_prediction_quality(
            "diabetes", X_train, X_test, y_train, y_test,
            task="regression", seed=42,
        )
        for rmse in exp1["results_table"]["RMSE"]:
            assert rmse > 0

    def test_per_model_icm_returned(self, iris_data):
        """Exp1 should return per-model ICM scores."""
        X_train, X_test, y_train, y_test = iris_data
        exp1 = experiment_prediction_quality(
            "iris", X_train, X_test, y_train, y_test,
            task="classification", seed=42,
        )
        assert "per_model_icm" in exp1
        assert len(exp1["per_model_icm"]) == 8  # 8 models in zoo


# ============================================================
# Experiment 2: Uncertainty Quantification
# ============================================================

class TestExperiment2:
    """Tests for Experiment 2: Uncertainty Quantification."""

    def test_classification_coverage(self, iris_data):
        """ICM-CRC coverage should be > 0 and <= 1."""
        X_train, X_test, y_train, y_test = iris_data
        exp2 = experiment_uncertainty_quantification(
            "iris", X_train, X_test, y_train, y_test,
            task="classification", seed=42,
        )
        tbl = exp2["results_table"]
        assert not tbl.empty
        icm_row = tbl[tbl["Method"] == "ICM-CRC Gating"]
        assert not icm_row.empty
        cov = float(icm_row["Coverage"].iloc[0])
        assert 0.0 < cov <= 1.0

    def test_conformal_coverage(self, iris_data):
        """Split conformal should appear in results with valid coverage."""
        X_train, X_test, y_train, y_test = iris_data
        exp2 = experiment_uncertainty_quantification(
            "iris", X_train, X_test, y_train, y_test,
            task="classification", seed=42,
        )
        tbl = exp2["results_table"]
        conf_row = tbl[tbl["Method"] == "Split Conformal"]
        assert not conf_row.empty
        cov = float(conf_row["Coverage"].iloc[0])
        assert 0.0 < cov <= 1.0

    def test_set_size_positive(self, iris_data):
        """Average set sizes should be >= 1."""
        X_train, X_test, y_train, y_test = iris_data
        exp2 = experiment_uncertainty_quantification(
            "iris", X_train, X_test, y_train, y_test,
            task="classification", seed=42,
        )
        for _, row in exp2["results_table"].iterrows():
            ss = row.get("Avg Set Size", np.nan)
            if not np.isnan(ss):
                assert ss >= 1.0

    def test_regression_skipped(self, diabetes_data):
        """Exp2 should skip gracefully for regression datasets."""
        X_train, X_test, y_train, y_test = diabetes_data
        exp2 = experiment_uncertainty_quantification(
            "diabetes", X_train, X_test, y_train, y_test,
            task="regression", seed=42,
        )
        assert exp2["results_table"].empty
        assert exp2.get("skip_reason") == "regression dataset"

    def test_icm_sets_returned(self, iris_data):
        """ICM prediction sets should be returned as list of sets."""
        X_train, X_test, y_train, y_test = iris_data
        exp2 = experiment_uncertainty_quantification(
            "iris", X_train, X_test, y_train, y_test,
            task="classification", seed=42,
        )
        assert isinstance(exp2["icm_sets"], list)
        for ps in exp2["icm_sets"]:
            assert isinstance(ps, set)
            assert len(ps) >= 1


# ============================================================
# Experiment 3: Error Detection
# ============================================================

class TestExperiment3:
    """Tests for Experiment 3: Error Detection."""

    def test_error_detection_signals(self, iris_data):
        """Exp3 should produce correlations for multiple signals."""
        X_train, X_test, y_train, y_test = iris_data
        exp3 = experiment_error_detection(
            "iris", X_train, X_test, y_train, y_test,
            task="classification", seed=42,
        )
        tbl = exp3["results_table"]
        assert len(tbl) >= 4
        signals = list(tbl["Signal"])
        assert "ICM (inverted)" in signals
        assert "Ensemble Entropy" in signals
        assert "Bootstrap Disagreement" in signals
        assert "Max-Prob Uncertainty" in signals

    def test_correlations_in_range(self, iris_data):
        """Spearman correlations should be in [-1, 1]."""
        X_train, X_test, y_train, y_test = iris_data
        exp3 = experiment_error_detection(
            "iris", X_train, X_test, y_train, y_test,
            task="classification", seed=42,
        )
        for _, row in exp3["results_table"].iterrows():
            corr = row["Spearman Corr"]
            assert -1.0 <= corr <= 1.0

    def test_p_values_in_range(self, iris_data):
        """p-values should be in [0, 1]."""
        X_train, X_test, y_train, y_test = iris_data
        exp3 = experiment_error_detection(
            "iris", X_train, X_test, y_train, y_test,
            task="classification", seed=42,
        )
        for _, row in exp3["results_table"].iterrows():
            p = row["p-value"]
            assert 0.0 <= p <= 1.0

    def test_regression_skipped(self, diabetes_data):
        """Exp3 should skip gracefully for regression datasets."""
        X_train, X_test, y_train, y_test = diabetes_data
        exp3 = experiment_error_detection(
            "diabetes", X_train, X_test, y_train, y_test,
            task="regression", seed=42,
        )
        assert exp3["results_table"].empty


# ============================================================
# Experiment 4: Diversity Assessment
# ============================================================

class TestExperiment4:
    """Tests for Experiment 4: Diversity Assessment."""

    def test_diversity_table_has_both_sources(self, iris_data):
        """Results should contain both ICM and Standard metrics."""
        X_train, X_test, y_train, y_test = iris_data
        exp4 = experiment_diversity_assessment(
            "iris", X_train, X_test, y_train, y_test,
            task="classification", seed=42,
        )
        tbl = exp4["results_table"]
        sources = set(tbl["Source"])
        assert "ICM" in sources
        assert "Standard" in sources

    def test_icm_components_returned(self, iris_data):
        """ICM components dict should have all 5 components + ICM score."""
        X_train, X_test, y_train, y_test = iris_data
        exp4 = experiment_diversity_assessment(
            "iris", X_train, X_test, y_train, y_test,
            task="classification", seed=42,
        )
        comps = exp4["icm_components"]
        for key in ["A", "D", "U", "C", "Pi", "ICM"]:
            assert key in comps
            assert 0.0 <= comps[key] <= 1.0

    def test_standard_metrics_returned(self, iris_data):
        """Standard diversity metrics should be present."""
        X_train, X_test, y_train, y_test = iris_data
        exp4 = experiment_diversity_assessment(
            "iris", X_train, X_test, y_train, y_test,
            task="classification", seed=42,
        )
        std = exp4["standard_metrics"]
        assert "Q-Statistic" in std
        assert "Disagreement" in std
        assert "Correlation" in std
        assert "Entropy" in std
        assert "KL Diversity" in std


# ============================================================
# Experiment 5: Score Range Analysis
# ============================================================

class TestExperiment5:
    """Tests for Experiment 5: Score Range Analysis."""

    def test_three_configs_compared(self, iris_data):
        """Should compare default, wide_range, and calibrated."""
        X_train, X_test, y_train, y_test = iris_data
        exp5 = experiment_score_range(
            "iris", X_train, X_test, y_train, y_test,
            task="classification", seed=42,
        )
        tbl = exp5["results_table"]
        configs = set(tbl["Config"])
        assert "Default Logistic" in configs
        assert "Wide Range" in configs
        assert "Calibrated Beta" in configs

    def test_wide_range_has_larger_range(self, iris_data):
        """Wide range should have larger score range than default."""
        X_train, X_test, y_train, y_test = iris_data
        exp5 = experiment_score_range(
            "iris", X_train, X_test, y_train, y_test,
            task="classification", seed=42,
        )
        tbl = exp5["results_table"]
        default_range = float(tbl[tbl["Config"] == "Default Logistic"]["Range"].iloc[0])
        wide_range = float(tbl[tbl["Config"] == "Wide Range"]["Range"].iloc[0])
        assert wide_range >= default_range

    def test_crc_thresholds_functional_wide_range(self, iris_data):
        """Wide range preset should produce scores spanning ACT/DEFER/AUDIT zones."""
        X_train, X_test, y_train, y_test = iris_data
        exp5 = experiment_score_range(
            "iris", X_train, X_test, y_train, y_test,
            task="classification", seed=42,
        )
        wide_scores = exp5["wide_scores"]
        # At least some scores should fall in different zones
        assert len(wide_scores) > 0
        has_high = np.any(wide_scores >= 0.7)
        has_low = np.any(wide_scores < 0.3)
        # At least two zones should be populated (wide range is discriminative)
        n_zones = sum([has_high, np.any((wide_scores >= 0.3) & (wide_scores < 0.7)), has_low])
        assert n_zones >= 2, "Wide range should produce scores in at least 2 zones"

    def test_all_scores_in_zero_one(self, iris_data):
        """All ICM scores should be in [0, 1]."""
        X_train, X_test, y_train, y_test = iris_data
        exp5 = experiment_score_range(
            "iris", X_train, X_test, y_train, y_test,
            task="classification", seed=42,
        )
        for key in ["default_scores", "wide_scores", "calibrated_scores"]:
            scores = exp5[key]
            if isinstance(scores, np.ndarray) and len(scores) > 0:
                assert np.all(scores >= 0.0), f"{key} has scores < 0"
                assert np.all(scores <= 1.0), f"{key} has scores > 1"


# ============================================================
# ICM Score Validity
# ============================================================

class TestICMScoreValidity:
    """Tests for ICM score validity and properties."""

    def test_per_sample_icm_in_range(self, iris_data, wide_config):
        """Per-sample ICM scores should be in [0, 1]."""
        X_train, X_test, y_train, y_test = iris_data
        models = build_classification_zoo(seed=42)
        train_zoo(models, X_train, y_train)
        preds = collect_predictions_classification(models, X_test)
        y_test_int = np.asarray(y_test, dtype=int)

        per_sample = _compute_per_sample_icm_classification(
            preds, y_test_int, wide_config,
        )
        assert np.all(per_sample >= 0.0)
        assert np.all(per_sample <= 1.0)

    def test_per_model_icm_in_range(self, iris_preds, iris_data, wide_config):
        """Per-model ICM contribution scores should be in [0.05, 0.95]."""
        _, _, _, y_test = iris_data
        per_model = _compute_per_model_icm_classification(
            iris_preds, y_test, wide_config,
        )
        for score in per_model.values():
            assert 0.05 <= score <= 0.95

    def test_ensemble_entropy_nonneg(self, iris_preds):
        """Ensemble entropy should be non-negative."""
        ent = _ensemble_entropy(iris_preds)
        assert np.all(ent >= 0.0)

    def test_per_sample_error_binary(self, iris_preds, iris_data):
        """Per-sample error should be 0 or 1."""
        _, _, _, y_test = iris_data
        errors = _compute_per_sample_error_classification(iris_preds, y_test)
        assert set(np.unique(errors)).issubset({0.0, 1.0})


# ============================================================
# Prediction Set Validity
# ============================================================

class TestPredictionSets:
    """Tests for ICM-based prediction sets."""

    def test_sets_nonempty(self, iris_preds, iris_data, wide_config):
        """All prediction sets should be non-empty."""
        _, _, _, y_test = iris_data
        y_test_int = np.asarray(y_test, dtype=int)
        per_sample = _compute_per_sample_icm_classification(
            iris_preds, y_test_int, wide_config,
        )
        sets = _icm_prediction_sets_classification(iris_preds, per_sample)
        for ps in sets:
            assert len(ps) >= 1

    def test_sets_contain_valid_classes(self, iris_preds, iris_data, wide_config):
        """Prediction sets should only contain valid class indices."""
        _, _, _, y_test = iris_data
        y_test_int = np.asarray(y_test, dtype=int)
        n_classes = len(np.unique(y_test_int))
        per_sample = _compute_per_sample_icm_classification(
            iris_preds, y_test_int, wide_config,
        )
        sets = _icm_prediction_sets_classification(iris_preds, per_sample)
        for ps in sets:
            for c in ps:
                assert 0 <= c < n_classes


# ============================================================
# Weighted Ensemble Helpers
# ============================================================

class TestWeightedEnsemble:
    """Tests for ICM-weighted ensemble helpers."""

    def test_classification_weighted_shape(self, iris_preds):
        """Weighted predictions should have correct shape."""
        fake_scores = {m: 0.5 for m in iris_preds}
        weighted = _icm_weighted_ensemble_classification(iris_preds, fake_scores)
        first_key = sorted(iris_preds.keys())[0]
        expected_shape = iris_preds[first_key].shape
        assert weighted.shape == expected_shape

    def test_classification_weighted_sums_to_one(self, iris_preds):
        """Weighted probability predictions should sum close to 1."""
        fake_scores = {m: np.random.rand() for m in iris_preds}
        weighted = _icm_weighted_ensemble_classification(iris_preds, fake_scores)
        row_sums = weighted.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-6)

    def test_regression_weighted_shape(self, diabetes_data):
        """Regression weighted output should have shape (n_samples,)."""
        X_train, X_test, y_train, y_test = diabetes_data
        models = build_regression_zoo(seed=42)
        train_zoo(models, X_train, y_train)
        preds = collect_predictions_regression(models, X_test)
        fake_scores = {m: 0.5 for m in preds}
        weighted = _icm_weighted_ensemble_regression(preds, fake_scores)
        assert weighted.shape == (len(X_test),)


# ============================================================
# Report Generation
# ============================================================

class TestReportGeneration:
    """Tests for report file output."""

    def test_report_file_created(self):
        """Running the benchmark should create a markdown report."""
        run_soa_benchmark(seed=42, verbose=False)
        report_path = Path(__file__).resolve().parent.parent / "reports" / "soa_benchmark_results.md"
        assert report_path.exists()

    def test_report_has_header(self):
        """Report should start with the expected header."""
        run_soa_benchmark(seed=42, verbose=False)
        report_path = Path(__file__).resolve().parent.parent / "reports" / "soa_benchmark_results.md"
        content = report_path.read_text(encoding="utf-8")
        assert "# SOA Benchmark" in content

    def test_report_has_executive_summary(self):
        """Report should contain an executive summary section."""
        run_soa_benchmark(seed=42, verbose=False)
        report_path = Path(__file__).resolve().parent.parent / "reports" / "soa_benchmark_results.md"
        content = report_path.read_text(encoding="utf-8")
        assert "## Executive Summary" in content

    def test_report_has_statistical_analysis(self):
        """Report should contain statistical analysis."""
        run_soa_benchmark(seed=42, verbose=False)
        report_path = Path(__file__).resolve().parent.parent / "reports" / "soa_benchmark_results.md"
        content = report_path.read_text(encoding="utf-8")
        assert "## Statistical Analysis" in content


# ============================================================
# Edge Cases
# ============================================================

class TestEdgeCases:
    """Edge case tests."""

    def test_binary_classification_dataset(self, breast_cancer_data):
        """Binary classification should work correctly."""
        X_train, X_test, y_train, y_test = breast_cancer_data
        exp1 = experiment_prediction_quality(
            "breast_cancer", X_train, X_test, y_train, y_test,
            task="classification", seed=42,
        )
        assert not exp1["results_table"].empty
        # Accuracy should be reasonable for breast cancer
        icm_row = exp1["results_table"][exp1["results_table"]["method"] == "ICM-Weighted"]
        assert float(icm_row["accuracy"].iloc[0]) > 0.80

    def test_binary_uncertainty_quantification(self, breast_cancer_data):
        """Uncertainty quantification should work for binary datasets."""
        X_train, X_test, y_train, y_test = breast_cancer_data
        exp2 = experiment_uncertainty_quantification(
            "breast_cancer", X_train, X_test, y_train, y_test,
            task="classification", seed=42,
        )
        assert not exp2["results_table"].empty

    def test_multiclass_all_experiments(self, iris_data):
        """All experiments should run on multi-class dataset (iris)."""
        X_train, X_test, y_train, y_test = iris_data
        for exp_fn in [
            experiment_prediction_quality,
            experiment_uncertainty_quantification,
            experiment_error_detection,
            experiment_diversity_assessment,
            experiment_score_range,
        ]:
            kwargs = dict(
                dataset_name="iris",
                X_train=X_train, X_test=X_test,
                y_train=y_train, y_test=y_test,
                task="classification", seed=42,
            )
            result = exp_fn(**kwargs)
            assert "results_table" in result

    def test_different_seeds_produce_results(self):
        """Benchmark should work with different seeds."""
        for s in [0, 1, 99]:
            results = run_soa_benchmark(seed=s, verbose=False)
            assert len(results["dataset_results"]) > 0

    def test_aggregate_results_structure(self):
        """Aggregate results should have expected keys."""
        results = run_soa_benchmark(seed=42, verbose=False)
        agg = results["aggregate_results"]
        assert "exp1_accuracy_ranking" in agg
        assert "exp2_coverage" in agg
        assert "exp3_corr_icm" in agg
        assert "exp5_wide_range_improvement" in agg

    def test_icm_analysis_has_components(self):
        """ICM analysis should contain mean components."""
        results = run_soa_benchmark(seed=42, verbose=False)
        analysis = results["icm_analysis"]
        assert "mean_icm_components" in analysis
        comps = analysis["mean_icm_components"]
        if comps:  # Should be non-empty with classification datasets
            for key in ["A", "D", "U", "C", "Pi", "ICM"]:
                assert key in comps
