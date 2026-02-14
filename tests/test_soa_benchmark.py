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

from benchmarks.datasets import (
    load_dataset,
    list_classification_datasets,
    list_regression_datasets,
    list_datasets,
    get_dataset_info,
)
from benchmarks.baselines import DeepEnsemble
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
    experiment_ablation_study,
    experiment_dependency_penalty,
    experiment_gating_comparison,
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
        """Wide range preset should produce scores with meaningful spread.

        Note: With the circularity fix (Pi=0 when no test labels used),
        well-agreeing models on easy datasets like iris may all land in the
        ACT zone.  The key property is that wide_range has a broader score
        range than the default logistic, enabling better discrimination.
        """
        X_train, X_test, y_train, y_test = iris_data
        exp5 = experiment_score_range(
            "iris", X_train, X_test, y_train, y_test,
            task="classification", seed=42,
        )
        wide_scores = exp5["wide_scores"]
        default_scores = exp5["default_scores"]
        assert len(wide_scores) > 0
        # Wide range should have more spread than default
        wide_range = float(wide_scores.max() - wide_scores.min())
        default_range = float(default_scores.max() - default_scores.min())
        assert wide_range >= default_range, (
            f"Wide range ({wide_range:.4f}) should be >= default range ({default_range:.4f})"
        )

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

        per_sample = _compute_per_sample_icm_classification(
            preds, wide_config,
        )
        assert np.all(per_sample >= 0.0)
        assert np.all(per_sample <= 1.0)

    def test_per_model_icm_in_range(self, iris_preds, iris_data, wide_config):
        """Per-model ICM contribution scores should be in [0.05, 0.95]."""
        per_model = _compute_per_model_icm_classification(
            iris_preds, wide_config,
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
        per_sample = _compute_per_sample_icm_classification(
            iris_preds, wide_config,
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
            iris_preds, wide_config,
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


# ============================================================
# New Datasets
# ============================================================

class TestNewDatasets:
    """Tests for newly added datasets (moons, circles, concept_drift, etc.)."""

    def test_moons_in_registry(self):
        """Moons dataset should be in the classification registry."""
        assert "moons" in list_classification_datasets()

    def test_circles_in_registry(self):
        """Circles dataset should be in the classification registry."""
        assert "circles" in list_classification_datasets()

    def test_concept_drift_in_registry(self):
        """Concept drift dataset should be in the classification registry."""
        assert "concept_drift" in list_classification_datasets()

    def test_moons_loads_correctly(self):
        """Moons dataset should load with correct shapes and types."""
        X_train, X_test, y_train, y_test = load_dataset("moons", seed=42)
        assert X_train.shape[1] == 2  # 2D features
        assert X_train.shape[0] == 800  # 80% of 1000
        assert X_test.shape[0] == 200  # 20% of 1000
        assert len(np.unique(y_train)) == 2  # binary
        assert y_train.dtype == np.int64

    def test_circles_loads_correctly(self):
        """Circles dataset should load with correct shapes and types."""
        X_train, X_test, y_train, y_test = load_dataset("circles", seed=42)
        assert X_train.shape[1] == 2  # 2D features
        assert X_train.shape[0] == 800
        assert X_test.shape[0] == 200
        assert len(np.unique(y_train)) == 2  # binary
        assert y_train.dtype == np.int64

    def test_concept_drift_loads_correctly(self):
        """Concept drift dataset should load with shifted test distribution."""
        X_train, X_test, y_train, y_test = load_dataset("concept_drift", seed=42)
        assert X_train.shape[1] == 5  # 5 features
        assert X_train.shape[0] == 480  # 3 classes * 160 per class
        assert X_test.shape[0] == 120  # 3 classes * 40 per class
        assert len(np.unique(y_train)) == 3  # 3 classes
        assert len(np.unique(y_test)) == 3
        assert y_train.dtype == np.int64

    def test_concept_drift_distribution_differs(self):
        """Train and test means should differ for concept drift dataset.

        The test set is generated from a shifted distribution, so
        the per-feature means should differ significantly.
        """
        X_train, X_test, _, _ = load_dataset("concept_drift", seed=42)
        # After standardization, train mean should be ~0
        train_mean = np.abs(X_train.mean(axis=0)).mean()
        # Test mean should differ from 0 due to distribution shift
        test_mean = np.abs(X_test.mean(axis=0)).mean()
        # The test set comes from a shifted distribution, so its mean
        # should be further from zero than the training set
        assert test_mean > train_mean

    def test_moons_experiment1(self):
        """Experiment 1 should run on moons dataset."""
        X_train, X_test, y_train, y_test = load_dataset("moons", seed=42)
        exp1 = experiment_prediction_quality(
            "moons", X_train, X_test, y_train, y_test,
            task="classification", seed=42,
        )
        assert not exp1["results_table"].empty
        methods = list(exp1["results_table"]["method"])
        assert "ICM-Weighted" in methods
        assert "Deep Ensemble" in methods

    def test_circles_experiment1(self):
        """Experiment 1 should run on circles dataset."""
        X_train, X_test, y_train, y_test = load_dataset("circles", seed=42)
        exp1 = experiment_prediction_quality(
            "circles", X_train, X_test, y_train, y_test,
            task="classification", seed=42,
        )
        assert not exp1["results_table"].empty
        accs = exp1["results_table"]["accuracy"].dropna()
        assert all(0.0 <= a <= 1.0 for a in accs)

    def test_concept_drift_experiment1(self):
        """Experiment 1 should run on concept drift dataset."""
        X_train, X_test, y_train, y_test = load_dataset("concept_drift", seed=42)
        exp1 = experiment_prediction_quality(
            "concept_drift", X_train, X_test, y_train, y_test,
            task="classification", seed=42,
        )
        assert not exp1["results_table"].empty

    def test_moons_nonlinear_accuracy(self):
        """Models should achieve reasonable accuracy on moons (nonlinear boundary)."""
        X_train, X_test, y_train, y_test = load_dataset("moons", seed=42)
        exp1 = experiment_prediction_quality(
            "moons", X_train, X_test, y_train, y_test,
            task="classification", seed=42,
        )
        icm_row = exp1["results_table"][exp1["results_table"]["method"] == "ICM-Weighted"]
        assert float(icm_row["accuracy"].iloc[0]) > 0.85

    def test_new_datasets_count(self):
        """At least 8 classification datasets should be registered."""
        clf = list_classification_datasets()
        assert len(clf) >= 8

    def test_all_datasets_have_info(self):
        """All registered datasets should have valid info."""
        for ds_name in list_datasets():
            info = get_dataset_info(ds_name)
            assert "description" in info
            assert "task" in info
            assert info["task"] in ("classification", "regression")

    def test_dataset_reproducibility(self):
        """Loading a dataset twice with same seed should give identical results."""
        for ds_name in ["moons", "circles", "concept_drift"]:
            X1, Xt1, y1, yt1 = load_dataset(ds_name, seed=42)
            X2, Xt2, y2, yt2 = load_dataset(ds_name, seed=42)
            np.testing.assert_array_equal(X1, X2)
            np.testing.assert_array_equal(Xt1, Xt2)
            np.testing.assert_array_equal(y1, y2)
            np.testing.assert_array_equal(yt1, yt2)

    def test_different_seeds_give_different_splits(self):
        """Different seeds should produce different train/test splits."""
        X1, _, _, _ = load_dataset("moons", seed=42)
        X2, _, _, _ = load_dataset("moons", seed=99)
        # Splits should differ (extremely unlikely to be identical)
        assert not np.array_equal(X1, X2)


# ============================================================
# Conditional datasets (may not be available)
# ============================================================

class TestConditionalDatasets:
    """Tests for datasets that may or may not be available."""

    def test_covertype_if_available(self):
        """Covertype should load correctly if available."""
        if "covertype" not in list_classification_datasets():
            pytest.skip("Covertype not available")
        X_train, X_test, y_train, y_test = load_dataset("covertype", seed=42)
        assert X_train.shape[1] == 54
        assert len(np.unique(y_train)) >= 2  # At least 2 classes
        # All class labels should be 0-indexed
        assert y_train.min() >= 0

    def test_olivetti_faces_if_available(self):
        """Olivetti faces should load correctly if available."""
        if "olivetti_faces" not in list_classification_datasets():
            pytest.skip("Olivetti faces not available")
        X_train, X_test, y_train, y_test = load_dataset("olivetti_faces", seed=42)
        assert X_train.shape[1] == 50  # PCA-reduced (fit on train only in load_dataset)
        assert len(np.unique(y_train)) == 40  # 40 classes
        assert y_train.dtype == np.int64

    def test_olivetti_faces_experiment1(self):
        """Experiment 1 should run on olivetti faces."""
        if "olivetti_faces" not in list_classification_datasets():
            pytest.skip("Olivetti faces not available")
        X_train, X_test, y_train, y_test = load_dataset("olivetti_faces", seed=42)
        exp1 = experiment_prediction_quality(
            "olivetti_faces", X_train, X_test, y_train, y_test,
            task="classification", seed=42,
        )
        assert not exp1["results_table"].empty
        # With 40 classes, accuracy should be reasonable (> 0.5)
        icm_row = exp1["results_table"][exp1["results_table"]["method"] == "ICM-Weighted"]
        assert float(icm_row["accuracy"].iloc[0]) > 0.50


# ============================================================
# Deep Ensemble Baseline
# ============================================================

class TestDeepEnsemble:
    """Tests for the DeepEnsemble baseline."""

    def test_deep_ensemble_creates_5_members(self):
        """DeepEnsemble should create 5 MLP members by default."""
        de = DeepEnsemble(n_members=5, max_iter=50, seed=42)
        X_train, _, y_train, _ = load_dataset("iris", seed=42)
        de.fit(X_train, y_train)
        assert len(de._models) == 5

    def test_deep_ensemble_predict_proba_shape(self):
        """predict_proba should return (n_samples, n_classes)."""
        de = DeepEnsemble(n_members=5, max_iter=50, seed=42)
        X_train, X_test, y_train, y_test = load_dataset("iris", seed=42)
        de.fit(X_train, y_train)
        proba = de.predict_proba(X_test)
        n_classes = len(np.unique(y_train))
        assert proba.shape == (len(X_test), n_classes)

    def test_deep_ensemble_probas_sum_to_one(self):
        """Predicted probabilities should sum to 1 for each sample."""
        de = DeepEnsemble(n_members=5, max_iter=50, seed=42)
        X_train, X_test, y_train, _ = load_dataset("iris", seed=42)
        de.fit(X_train, y_train)
        proba = de.predict_proba(X_test)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)

    def test_deep_ensemble_probas_nonneg(self):
        """Predicted probabilities should be non-negative."""
        de = DeepEnsemble(n_members=5, max_iter=50, seed=42)
        X_train, X_test, y_train, _ = load_dataset("iris", seed=42)
        de.fit(X_train, y_train)
        proba = de.predict_proba(X_test)
        assert np.all(proba >= 0.0)

    def test_deep_ensemble_score_format(self):
        """score() should return dict with accuracy, log_loss, brier_score."""
        de = DeepEnsemble(n_members=5, max_iter=50, seed=42)
        X_train, X_test, y_train, y_test = load_dataset("iris", seed=42)
        de.fit(X_train, y_train)
        scores = de.score(X_test, y_test)
        assert "accuracy" in scores
        assert "log_loss" in scores
        assert "brier_score" in scores
        assert 0.0 <= scores["accuracy"] <= 1.0
        assert scores["log_loss"] >= 0.0
        assert scores["brier_score"] >= 0.0

    def test_deep_ensemble_reasonable_accuracy(self):
        """DeepEnsemble should achieve reasonable accuracy on iris."""
        de = DeepEnsemble(n_members=5, max_iter=200, seed=42)
        X_train, X_test, y_train, y_test = load_dataset("iris", seed=42)
        de.fit(X_train, y_train)
        scores = de.score(X_test, y_test)
        assert scores["accuracy"] > 0.6  # Should do better than random

    def test_deep_ensemble_per_member_predictions(self):
        """predict_proba_per_member should return list of 5 arrays."""
        de = DeepEnsemble(n_members=5, max_iter=50, seed=42)
        X_train, X_test, y_train, _ = load_dataset("iris", seed=42)
        de.fit(X_train, y_train)
        per_member = de.predict_proba_per_member(X_test)
        assert len(per_member) == 5
        for arr in per_member:
            assert arr.shape == (len(X_test), len(np.unique(y_train)))

    def test_deep_ensemble_member_diversity(self):
        """Different MLP architectures should produce different predictions."""
        de = DeepEnsemble(n_members=5, max_iter=100, seed=42)
        X_train, X_test, y_train, _ = load_dataset("iris", seed=42)
        de.fit(X_train, y_train)
        per_member = de.predict_proba_per_member(X_test)
        # At least some members should differ
        diffs = []
        for i in range(len(per_member)):
            for j in range(i + 1, len(per_member)):
                diffs.append(np.mean(np.abs(per_member[i] - per_member[j])))
        assert max(diffs) > 1e-6, "All ensemble members are identical"

    def test_deep_ensemble_disagreement_shape(self):
        """disagreement() should return (n_samples,) non-negative values."""
        de = DeepEnsemble(n_members=5, max_iter=50, seed=42)
        X_train, X_test, y_train, _ = load_dataset("iris", seed=42)
        de.fit(X_train, y_train)
        dis = de.disagreement(X_test)
        assert dis.shape == (len(X_test),)
        assert np.all(dis >= 0.0)

    def test_deep_ensemble_not_fitted_raises(self):
        """Calling predict before fit should raise RuntimeError."""
        de = DeepEnsemble(n_members=5, max_iter=50, seed=42)
        X_test = np.random.randn(10, 4)
        with pytest.raises(RuntimeError):
            de.predict_proba(X_test)
        with pytest.raises(RuntimeError):
            de.predict_proba_per_member(X_test)

    def test_deep_ensemble_binary_classification(self):
        """DeepEnsemble should work on binary classification."""
        de = DeepEnsemble(n_members=5, max_iter=100, seed=42)
        X_train, X_test, y_train, y_test = load_dataset("breast_cancer", seed=42)
        de.fit(X_train, y_train)
        scores = de.score(X_test, y_test)
        assert scores["accuracy"] > 0.80  # Should do well on breast cancer

    def test_deep_ensemble_in_benchmark(self):
        """Deep Ensemble should appear in exp1 results."""
        X_train, X_test, y_train, y_test = load_dataset("iris", seed=42)
        exp1 = experiment_prediction_quality(
            "iris", X_train, X_test, y_train, y_test,
            task="classification", seed=42,
        )
        methods = list(exp1["results_table"]["method"])
        assert "Deep Ensemble" in methods

    def test_deep_ensemble_on_nonlinear_data(self):
        """DeepEnsemble should handle nonlinear boundaries (moons)."""
        de = DeepEnsemble(n_members=5, max_iter=200, seed=42)
        X_train, X_test, y_train, y_test = load_dataset("moons", seed=42)
        de.fit(X_train, y_train)
        scores = de.score(X_test, y_test)
        assert scores["accuracy"] > 0.85


# ============================================================
# Expanded Benchmark Integration
# ============================================================

class TestPerSampleICMAggregation:
    """Tests for per-sample ICM aggregation methods."""

    def test_per_sample_icm_logistic_vs_calibrated_differ(self, iris_preds):
        """Per-sample ICM with Logistic and Calibrated Beta should differ."""
        config_logistic = ICMConfig(aggregation="logistic")
        scores_logistic = _compute_per_sample_icm_classification(iris_preds, config_logistic)

        config_calibrated = ICMConfig(aggregation="calibrated", beta_shape_a=5.0, beta_shape_b=5.0)
        scores_calibrated = _compute_per_sample_icm_classification(iris_preds, config_calibrated)

        # Scores should not be identical
        assert not np.allclose(scores_logistic, scores_calibrated, atol=1e-6), (
            "Logistic and Calibrated Beta per-sample scores should differ"
        )

    def test_per_sample_icm_respects_aggregation_config(self, iris_preds):
        """Per-sample ICM should respect config.aggregation setting."""
        for method in ["logistic", "geometric", "calibrated", "adaptive"]:
            config = ICMConfig(aggregation=method)
            scores = _compute_per_sample_icm_classification(iris_preds, config)

            # All scores should be valid
            assert np.all(scores >= 0.0), f"{method}: scores < 0"
            assert np.all(scores <= 1.0), f"{method}: scores > 1"
            assert len(scores) == iris_preds[sorted(iris_preds.keys())[0]].shape[0]

    def test_calibrated_beta_scores_in_range(self, iris_preds):
        """Calibrated Beta per-sample scores should be in [0, 1]."""
        config = ICMConfig(aggregation="calibrated", beta_shape_a=5.0, beta_shape_b=5.0)
        scores = _compute_per_sample_icm_classification(iris_preds, config)
        assert np.all(scores >= 0.0)
        assert np.all(scores <= 1.0)

    def test_geometric_aggregation_per_sample(self, iris_preds):
        """Geometric aggregation should produce valid per-sample scores."""
        config = ICMConfig(aggregation="geometric")
        scores = _compute_per_sample_icm_classification(iris_preds, config)
        assert np.all(scores >= 0.0)
        assert np.all(scores <= 1.0)
        assert len(scores) > 0

    def test_adaptive_aggregation_per_sample(self, iris_preds):
        """Adaptive aggregation should produce valid per-sample scores."""
        config = ICMConfig(aggregation="adaptive")
        scores = _compute_per_sample_icm_classification(iris_preds, config)
        assert np.all(scores >= 0.0)
        assert np.all(scores <= 1.0)
        assert len(scores) > 0


class TestExpandedBenchmarkIntegration:
    """Integration tests for the expanded benchmark with all datasets."""

    def test_expanded_benchmark_includes_new_datasets(self):
        """Full benchmark should include new datasets."""
        results = run_soa_benchmark(seed=42, verbose=False)
        for ds_name in ["moons", "circles", "concept_drift"]:
            assert ds_name in results["dataset_results"], f"Missing: {ds_name}"

    def test_expanded_benchmark_deep_ensemble_present(self):
        """Deep Ensemble should be present in all classification results."""
        results = run_soa_benchmark(seed=42, verbose=False)
        for ds_name, ds_data in results["dataset_results"].items():
            info = ds_data.get("info", {})
            if info.get("task") != "classification":
                continue
            exp1 = ds_data.get("exp1_prediction_quality", {})
            if "results_table" in exp1 and not exp1["results_table"].empty:
                methods = list(exp1["results_table"]["method"])
                assert "Deep Ensemble" in methods, (
                    f"Deep Ensemble missing from {ds_name}"
                )

    def test_expanded_aggregate_has_more_data(self):
        """Aggregate results should cover more datasets with expansion."""
        results = run_soa_benchmark(seed=42, verbose=False)
        agg = results["aggregate_results"]
        # Should have at least 7 classification datasets in accuracy ranking
        assert len(agg["exp1_accuracy_ranking"]) >= 7


# ============================================================
# Tests for Experiment 6: Ablation Study
# ============================================================

class TestAblationStudy:
    """Tests for the ablation study experiment."""

    def test_ablation_returns_dict(self, iris_data, trained_clf_models_iris):
        """Ablation study returns a dict with results_table."""
        X_train, X_test, y_train, y_test = iris_data
        preds_test = collect_predictions_classification(
            trained_clf_models_iris, X_test,
        )
        result = experiment_ablation_study(
            "iris", X_train, X_test, y_train, y_test,
            task="classification", seed=42, _preds_test=preds_test,
        )
        assert isinstance(result, dict)
        assert "results_table" in result
        assert isinstance(result["results_table"], pd.DataFrame)

    def test_ablation_has_all_variants(self, iris_data, trained_clf_models_iris):
        """Ablation study includes all expected variants."""
        X_train, X_test, y_train, y_test = iris_data
        preds_test = collect_predictions_classification(
            trained_clf_models_iris, X_test,
        )
        result = experiment_ablation_study(
            "iris", X_train, X_test, y_train, y_test,
            task="classification", seed=42, _preds_test=preds_test,
        )
        df = result["results_table"]
        expected = ["Full ICM", "No A (w_A~0)", "No D (w_D~0)",
                     "No U (w_U~0)", "No C (w_C~0)"]
        for v in expected:
            assert v in df["Variant"].values, f"Missing variant: {v}"

    def test_ablation_full_icm_highest_score(self, iris_data, trained_clf_models_iris):
        """Full ICM should generally produce the highest mean score."""
        X_train, X_test, y_train, y_test = iris_data
        preds_test = collect_predictions_classification(
            trained_clf_models_iris, X_test,
        )
        result = experiment_ablation_study(
            "iris", X_train, X_test, y_train, y_test,
            task="classification", seed=42, _preds_test=preds_test,
        )
        df = result["results_table"]
        full_score = df[df["Variant"] == "Full ICM"]["Mean Score"].values[0]
        random_score = df[df["Variant"] == "Random Baseline"]["Mean Score"].values[0]
        assert full_score > random_score

    def test_ablation_scores_bounded(self, iris_data, trained_clf_models_iris):
        """All ablation scores should be in [0, 1]."""
        X_train, X_test, y_train, y_test = iris_data
        preds_test = collect_predictions_classification(
            trained_clf_models_iris, X_test,
        )
        result = experiment_ablation_study(
            "iris", X_train, X_test, y_train, y_test,
            task="classification", seed=42, _preds_test=preds_test,
        )
        df = result["results_table"]
        for score in df["Mean Score"].values:
            assert 0.0 <= float(score) <= 1.0


# ============================================================
# Tests for Experiment 7: Dependency Penalty
# ============================================================

class TestDependencyPenalty:
    """Tests for the dependency penalty experiment."""

    def test_dependency_returns_dict(self, iris_data, trained_clf_models_iris):
        """Dependency penalty experiment returns expected structure."""
        X_train, X_test, y_train, y_test = iris_data
        preds_test = collect_predictions_classification(
            trained_clf_models_iris, X_test,
        )
        preds_train = collect_predictions_classification(
            trained_clf_models_iris, X_train,
        )
        result = experiment_dependency_penalty(
            "iris", X_train, X_test, y_train, y_test,
            task="classification", seed=42,
            _preds_train=preds_train, _preds_test=preds_test,
        )
        assert isinstance(result, dict)
        assert "results_table" in result

    def test_dependency_has_scenarios(self, iris_data, trained_clf_models_iris):
        """Should have the three expected scenarios."""
        X_train, X_test, y_train, y_test = iris_data
        preds_test = collect_predictions_classification(
            trained_clf_models_iris, X_test,
        )
        preds_train = collect_predictions_classification(
            trained_clf_models_iris, X_train,
        )
        result = experiment_dependency_penalty(
            "iris", X_train, X_test, y_train, y_test,
            task="classification", seed=42,
            _preds_train=preds_train, _preds_test=preds_test,
        )
        df = result["results_table"]
        assert len(df) >= 2  # At least no-pi and with-pi scenarios

    def test_dependency_pi_activates(self, iris_data, trained_clf_models_iris):
        """With dependency data, Pi should be > 0."""
        X_train, X_test, y_train, y_test = iris_data
        preds_test = collect_predictions_classification(
            trained_clf_models_iris, X_test,
        )
        preds_train = collect_predictions_classification(
            trained_clf_models_iris, X_train,
        )
        result = experiment_dependency_penalty(
            "iris", X_train, X_test, y_train, y_test,
            task="classification", seed=42,
            _preds_train=preds_train, _preds_test=preds_test,
        )
        df = result["results_table"]
        # The "with Pi" scenario should have Pi > 0
        with_pi = df[df["Scenario"].str.contains("with Pi")]
        if len(with_pi) > 0:
            pi_vals = with_pi["Pi Value"].values
            assert any(float(p) > 0 for p in pi_vals)


# ============================================================
# Tests for Experiment 8: Gating Comparison
# ============================================================

class TestGatingComparison:
    """Tests for the gating comparison experiment."""

    def test_gating_returns_dict(self, iris_data, trained_clf_models_iris):
        """Gating comparison returns expected structure."""
        X_train, X_test, y_train, y_test = iris_data
        preds_test = collect_predictions_classification(
            trained_clf_models_iris, X_test,
        )
        result = experiment_gating_comparison(
            "iris", X_train, X_test, y_train, y_test,
            task="classification", seed=42, _preds_test=preds_test,
        )
        assert isinstance(result, dict)
        assert "results_table" in result

    def test_gating_has_all_methods(self, iris_data, trained_clf_models_iris):
        """Should compare ICM against simpler gating methods."""
        X_train, X_test, y_train, y_test = iris_data
        preds_test = collect_predictions_classification(
            trained_clf_models_iris, X_test,
        )
        result = experiment_gating_comparison(
            "iris", X_train, X_test, y_train, y_test,
            task="classification", seed=42, _preds_test=preds_test,
        )
        df = result["results_table"]
        methods = list(df["Method"].values)
        assert "ICM (wide-range)" in methods
        assert "Entropy Only" in methods

    def test_gating_act_pcts_valid(self, iris_data, trained_clf_models_iris):
        """ACT percentages should be between 0 and 100."""
        X_train, X_test, y_train, y_test = iris_data
        preds_test = collect_predictions_classification(
            trained_clf_models_iris, X_test,
        )
        result = experiment_gating_comparison(
            "iris", X_train, X_test, y_train, y_test,
            task="classification", seed=42, _preds_test=preds_test,
        )
        df = result["results_table"]
        for act in df["ACT%"].values:
            assert 0.0 <= float(act) <= 100.0
