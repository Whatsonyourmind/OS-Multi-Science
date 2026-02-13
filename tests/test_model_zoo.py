"""Comprehensive tests for the model zoo and dataset loader.

Tests cover:
- Each model family trains and produces valid outputs
- Predictions have correct shapes
- Probabilities sum to 1 for classification
- Residuals are computed correctly
- All datasets load correctly
- collect_predictions returns ICM-compatible format
- Different model families produce genuinely different predictions
"""

from __future__ import annotations

import numpy as np
import pytest

from benchmarks.model_zoo import (
    ClassificationWrapper,
    RegressionWrapper,
    build_classification_zoo,
    build_regression_zoo,
    train_zoo,
    collect_predictions_classification,
    collect_predictions_regression,
    collect_residuals,
)
from benchmarks.datasets import (
    load_dataset,
    list_datasets,
    list_classification_datasets,
    list_regression_datasets,
    get_dataset_info,
)


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture(scope="module")
def iris_data():
    """Load Iris dataset once for the module."""
    return load_dataset("iris", seed=42)


@pytest.fixture(scope="module")
def diabetes_data():
    """Load Diabetes dataset once for the module."""
    return load_dataset("diabetes", seed=42)


@pytest.fixture(scope="module")
def classification_zoo():
    """Build the classification zoo once."""
    return build_classification_zoo(seed=42)


@pytest.fixture(scope="module")
def regression_zoo():
    """Build the regression zoo once."""
    return build_regression_zoo(seed=42)


@pytest.fixture(scope="module")
def trained_classification_zoo(iris_data, classification_zoo):
    """Classification zoo trained on Iris."""
    X_train, X_test, y_train, y_test = iris_data
    train_zoo(classification_zoo, X_train, y_train)
    return classification_zoo


@pytest.fixture(scope="module")
def trained_regression_zoo(diabetes_data, regression_zoo):
    """Regression zoo trained on Diabetes."""
    X_train, X_test, y_train, y_test = diabetes_data
    train_zoo(regression_zoo, X_train, y_train)
    return regression_zoo


# ============================================================
# 1. Dataset loading tests
# ============================================================

class TestListDatasets:
    """Tests for dataset listing functions."""

    def test_list_datasets_returns_all(self):
        names = list_datasets()
        assert len(names) >= 6
        assert "iris" in names
        assert "wine" in names
        assert "breast_cancer" in names
        assert "digits" in names
        assert "california_housing" in names
        assert "diabetes" in names

    def test_list_classification_datasets(self):
        names = list_classification_datasets()
        assert len(names) >= 4
        assert "iris" in names
        assert "wine" in names
        assert "breast_cancer" in names
        assert "digits" in names

    def test_list_regression_datasets(self):
        names = list_regression_datasets()
        assert len(names) >= 2
        assert "california_housing" in names
        assert "diabetes" in names

    def test_datasets_sorted(self):
        names = list_datasets()
        assert names == sorted(names)


class TestGetDatasetInfo:
    """Tests for dataset metadata."""

    def test_iris_info(self):
        info = get_dataset_info("iris")
        assert info["task"] == "classification"
        assert "Iris" in info["description"]

    def test_diabetes_info(self):
        info = get_dataset_info("diabetes")
        assert info["task"] == "regression"

    def test_unknown_dataset_raises(self):
        with pytest.raises(ValueError, match="Unknown dataset"):
            get_dataset_info("nonexistent_dataset")


class TestLoadDataset:
    """Tests for loading individual datasets."""

    @pytest.mark.parametrize("name", ["iris", "wine", "breast_cancer", "digits"])
    def test_classification_dataset_shapes(self, name):
        X_train, X_test, y_train, y_test = load_dataset(name, seed=42)
        total = len(y_train) + len(y_test)
        assert X_train.shape[0] == len(y_train)
        assert X_test.shape[0] == len(y_test)
        assert X_train.shape[1] == X_test.shape[1]
        # Approximately 80/20 split
        assert abs(len(y_test) / total - 0.2) < 0.05

    @pytest.mark.parametrize("name", ["california_housing", "diabetes"])
    def test_regression_dataset_shapes(self, name):
        X_train, X_test, y_train, y_test = load_dataset(name, seed=42)
        total = len(y_train) + len(y_test)
        assert X_train.shape[0] == len(y_train)
        assert X_test.shape[0] == len(y_test)
        assert X_train.shape[1] == X_test.shape[1]
        assert abs(len(y_test) / total - 0.2) < 0.05

    def test_features_standardized_train(self):
        """Training features should have approximately zero mean, unit variance."""
        X_train, _, _, _ = load_dataset("iris", seed=42)
        col_means = X_train.mean(axis=0)
        col_stds = X_train.std(axis=0)
        np.testing.assert_allclose(col_means, 0.0, atol=1e-10)
        np.testing.assert_allclose(col_stds, 1.0, atol=0.05)

    def test_classification_target_dtype(self):
        _, _, y_train, y_test = load_dataset("iris", seed=42)
        assert y_train.dtype == np.int64
        assert y_test.dtype == np.int64

    def test_regression_target_dtype(self):
        _, _, y_train, y_test = load_dataset("diabetes", seed=42)
        assert y_train.dtype == np.float64
        assert y_test.dtype == np.float64

    def test_unknown_dataset_raises(self):
        with pytest.raises(ValueError, match="Unknown dataset"):
            load_dataset("fake_dataset_xyz")

    def test_reproducibility(self):
        """Same seed should produce identical splits."""
        d1 = load_dataset("iris", seed=42)
        d2 = load_dataset("iris", seed=42)
        for a, b in zip(d1, d2):
            np.testing.assert_array_equal(a, b)

    def test_different_seeds_differ(self):
        """Different seeds should produce different splits."""
        X1, _, _, _ = load_dataset("iris", seed=42)
        X2, _, _, _ = load_dataset("iris", seed=99)
        # The rows should be in a different order
        assert not np.array_equal(X1, X2)


# ============================================================
# 2. Classification model zoo tests
# ============================================================

class TestBuildClassificationZoo:
    """Tests for building the classification zoo."""

    def test_zoo_has_at_least_8_models(self, classification_zoo):
        assert len(classification_zoo) >= 8

    def test_all_have_name(self, classification_zoo):
        for model in classification_zoo:
            assert isinstance(model.name, str)
            assert len(model.name) > 0

    def test_all_have_family(self, classification_zoo):
        for model in classification_zoo:
            assert isinstance(model.family, str)
            assert len(model.family) > 0

    def test_families_are_diverse(self, classification_zoo):
        """Models should span multiple distinct epistemic families."""
        families = {m.family for m in classification_zoo}
        assert len(families) >= 6, (
            f"Only {len(families)} unique families: {families}. "
            f"Expected at least 6 genuinely diverse families."
        )

    def test_names_are_unique(self, classification_zoo):
        names = [m.name for m in classification_zoo]
        assert len(names) == len(set(names))

    def test_is_classification_wrapper(self, classification_zoo):
        for model in classification_zoo:
            assert isinstance(model, ClassificationWrapper)


class TestClassificationModelTraining:
    """Tests for training and predicting with each classification model."""

    def test_all_models_train(self, trained_classification_zoo):
        """All models should train without error (handled by fixture)."""
        assert len(trained_classification_zoo) >= 8

    def test_predict_proba_shape(self, trained_classification_zoo, iris_data):
        X_train, X_test, y_train, y_test = iris_data
        n_classes = len(np.unique(y_train))
        for model in trained_classification_zoo:
            proba = model.predict_proba(X_test)
            assert proba.shape == (len(X_test), n_classes), (
                f"{model.name}: expected shape ({len(X_test)}, {n_classes}), "
                f"got {proba.shape}"
            )

    def test_probabilities_sum_to_one(self, trained_classification_zoo, iris_data):
        _, X_test, _, _ = iris_data
        for model in trained_classification_zoo:
            proba = model.predict_proba(X_test)
            row_sums = proba.sum(axis=1)
            np.testing.assert_allclose(
                row_sums, 1.0, atol=1e-5,
                err_msg=f"{model.name}: probabilities do not sum to 1",
            )

    def test_probabilities_non_negative(self, trained_classification_zoo, iris_data):
        _, X_test, _, _ = iris_data
        for model in trained_classification_zoo:
            proba = model.predict_proba(X_test)
            assert np.all(proba >= -1e-10), (
                f"{model.name}: negative probabilities detected"
            )

    def test_probabilities_bounded(self, trained_classification_zoo, iris_data):
        _, X_test, _, _ = iris_data
        for model in trained_classification_zoo:
            proba = model.predict_proba(X_test)
            assert np.all(proba <= 1.0 + 1e-10), (
                f"{model.name}: probabilities > 1 detected"
            )


# ============================================================
# 3. Regression model zoo tests
# ============================================================

class TestBuildRegressionZoo:
    """Tests for building the regression zoo."""

    def test_zoo_has_at_least_8_models(self, regression_zoo):
        assert len(regression_zoo) >= 8

    def test_all_have_name(self, regression_zoo):
        for model in regression_zoo:
            assert isinstance(model.name, str)
            assert len(model.name) > 0

    def test_all_have_family(self, regression_zoo):
        for model in regression_zoo:
            assert isinstance(model.family, str)
            assert len(model.family) > 0

    def test_families_are_diverse(self, regression_zoo):
        families = {m.family for m in regression_zoo}
        assert len(families) >= 5, (
            f"Only {len(families)} unique families: {families}. "
            f"Expected at least 5 genuinely diverse families."
        )

    def test_names_are_unique(self, regression_zoo):
        names = [m.name for m in regression_zoo]
        assert len(names) == len(set(names))

    def test_is_regression_wrapper(self, regression_zoo):
        for model in regression_zoo:
            assert isinstance(model, RegressionWrapper)


class TestRegressionModelTraining:
    """Tests for training and predicting with each regression model."""

    def test_all_models_train(self, trained_regression_zoo):
        assert len(trained_regression_zoo) >= 8

    def test_predict_quantiles_shape(self, trained_regression_zoo, diabetes_data):
        _, X_test, _, _ = diabetes_data
        for model in trained_regression_zoo:
            quantiles = model.predict_quantiles(X_test)
            assert quantiles.shape == (len(X_test), 3), (
                f"{model.name}: expected shape ({len(X_test)}, 3), "
                f"got {quantiles.shape}"
            )

    def test_quantile_ordering(self, trained_regression_zoo, diabetes_data):
        """For most samples, lower_10 <= mean <= upper_90 should hold."""
        _, X_test, _, _ = diabetes_data
        for model in trained_regression_zoo:
            quantiles = model.predict_quantiles(X_test)
            mean = quantiles[:, 0]
            lower = quantiles[:, 1]
            upper = quantiles[:, 2]
            # Allow some tolerance; at least 80% of samples should satisfy ordering
            valid_lower = np.sum(lower <= mean + 1e-6) / len(mean)
            valid_upper = np.sum(upper >= mean - 1e-6) / len(mean)
            assert valid_lower > 0.8, (
                f"{model.name}: lower_10 > mean for too many samples"
            )
            assert valid_upper > 0.8, (
                f"{model.name}: upper_90 < mean for too many samples"
            )

    def test_predictions_are_finite(self, trained_regression_zoo, diabetes_data):
        _, X_test, _, _ = diabetes_data
        for model in trained_regression_zoo:
            quantiles = model.predict_quantiles(X_test)
            assert np.all(np.isfinite(quantiles)), (
                f"{model.name}: non-finite predictions detected"
            )


# ============================================================
# 4. Prediction collection tests (ICM compatibility)
# ============================================================

class TestCollectPredictionsClassification:
    """Tests for collect_predictions_classification."""

    def test_returns_dict(self, trained_classification_zoo, iris_data):
        _, X_test, _, _ = iris_data
        preds = collect_predictions_classification(trained_classification_zoo, X_test)
        assert isinstance(preds, dict)

    def test_keys_are_model_names(self, trained_classification_zoo, iris_data):
        _, X_test, _, _ = iris_data
        preds = collect_predictions_classification(trained_classification_zoo, X_test)
        expected_names = {m.name for m in trained_classification_zoo}
        assert set(preds.keys()) == expected_names

    def test_values_are_ndarrays(self, trained_classification_zoo, iris_data):
        _, X_test, _, _ = iris_data
        preds = collect_predictions_classification(trained_classification_zoo, X_test)
        for name, arr in preds.items():
            assert isinstance(arr, np.ndarray), (
                f"{name}: expected ndarray, got {type(arr)}"
            )

    def test_icm_compatible_shape(self, trained_classification_zoo, iris_data):
        """Each prediction array should be (n_samples, n_classes)."""
        X_train, X_test, y_train, _ = iris_data
        n_classes = len(np.unique(y_train))
        preds = collect_predictions_classification(trained_classification_zoo, X_test)
        for name, arr in preds.items():
            assert arr.shape == (len(X_test), n_classes), (
                f"{name}: wrong shape {arr.shape}"
            )


class TestCollectPredictionsRegression:
    """Tests for collect_predictions_regression."""

    def test_returns_dict(self, trained_regression_zoo, diabetes_data):
        _, X_test, _, _ = diabetes_data
        preds = collect_predictions_regression(trained_regression_zoo, X_test)
        assert isinstance(preds, dict)

    def test_keys_are_model_names(self, trained_regression_zoo, diabetes_data):
        _, X_test, _, _ = diabetes_data
        preds = collect_predictions_regression(trained_regression_zoo, X_test)
        expected_names = {m.name for m in trained_regression_zoo}
        assert set(preds.keys()) == expected_names

    def test_values_shape(self, trained_regression_zoo, diabetes_data):
        _, X_test, _, _ = diabetes_data
        preds = collect_predictions_regression(trained_regression_zoo, X_test)
        for name, arr in preds.items():
            assert arr.shape == (len(X_test), 3), (
                f"{name}: expected shape ({len(X_test)}, 3), got {arr.shape}"
            )


# ============================================================
# 5. Residuals tests
# ============================================================

class TestCollectResiduals:
    """Tests for collect_residuals."""

    def test_residuals_shape(self, trained_regression_zoo, diabetes_data):
        _, X_test, _, y_test = diabetes_data
        residuals = collect_residuals(trained_regression_zoo, X_test, y_test)
        K = len(trained_regression_zoo)
        N = len(y_test)
        assert residuals.shape == (K, N)

    def test_residuals_are_finite(self, trained_regression_zoo, diabetes_data):
        _, X_test, _, y_test = diabetes_data
        residuals = collect_residuals(trained_regression_zoo, X_test, y_test)
        assert np.all(np.isfinite(residuals))

    def test_residuals_definition(self, trained_regression_zoo, diabetes_data):
        """Residuals should equal y_true - y_pred for each model."""
        _, X_test, _, y_test = diabetes_data
        residuals = collect_residuals(trained_regression_zoo, X_test, y_test)
        for i, model in enumerate(trained_regression_zoo):
            y_pred = model.predict_quantiles(X_test)[:, 0]
            expected = y_test - y_pred
            np.testing.assert_allclose(
                residuals[i], expected, atol=1e-10,
                err_msg=f"{model.name}: residuals mismatch",
            )

    def test_residuals_not_all_zero(self, trained_regression_zoo, diabetes_data):
        """Residuals should not all be zero (models are not perfect)."""
        _, X_test, _, y_test = diabetes_data
        residuals = collect_residuals(trained_regression_zoo, X_test, y_test)
        for i, model in enumerate(trained_regression_zoo):
            assert np.std(residuals[i]) > 1e-6, (
                f"{model.name}: residuals are essentially zero"
            )


# ============================================================
# 6. Epistemic diversity tests (the critical ones!)
# ============================================================

class TestEpistemicDiversity:
    """Verify that different model families produce genuinely different predictions.

    This is the core fix: the old approach used noise-perturbed variants
    of the same prediction.  Here we assert that models from different
    epistemic families produce non-trivially different outputs.
    """

    def test_classification_predictions_differ(self, trained_classification_zoo, iris_data):
        """Different model families should NOT produce identical probability vectors."""
        _, X_test, _, _ = iris_data
        preds = collect_predictions_classification(trained_classification_zoo, X_test)
        names = list(preds.keys())

        n_pairs = 0
        n_different = 0
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                p_i = preds[names[i]]
                p_j = preds[names[j]]
                max_diff = np.max(np.abs(p_i - p_j))
                n_pairs += 1
                if max_diff > 0.01:
                    n_different += 1

        # At least 50% of pairs should have non-trivial differences
        fraction_different = n_different / n_pairs
        assert fraction_different >= 0.5, (
            f"Only {fraction_different:.1%} of model pairs differ meaningfully. "
            f"Models may not be genuinely epistemically diverse."
        )

    def test_regression_predictions_differ(self, trained_regression_zoo, diabetes_data):
        """Different regression model families should produce different predictions."""
        _, X_test, _, _ = diabetes_data
        preds = collect_predictions_regression(trained_regression_zoo, X_test)
        names = list(preds.keys())

        n_pairs = 0
        n_different = 0
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                mean_i = preds[names[i]][:, 0]
                mean_j = preds[names[j]][:, 0]
                rel_diff = np.mean(np.abs(mean_i - mean_j)) / (
                    np.mean(np.abs(mean_i)) + 1e-10
                )
                n_pairs += 1
                if rel_diff > 0.01:
                    n_different += 1

        fraction_different = n_different / n_pairs
        assert fraction_different >= 0.5, (
            f"Only {fraction_different:.1%} of regression model pairs differ. "
            f"Models may not be genuinely epistemically diverse."
        )

    def test_classification_disagreement_on_hard_samples(
        self, trained_classification_zoo, iris_data
    ):
        """Models should disagree on at least some samples' top predicted class."""
        _, X_test, _, _ = iris_data
        preds = collect_predictions_classification(trained_classification_zoo, X_test)

        # Get predicted class for each model on each sample
        predicted_classes = {}
        for name, proba in preds.items():
            predicted_classes[name] = np.argmax(proba, axis=1)

        names = list(predicted_classes.keys())
        n_samples = len(X_test)

        # Count samples where at least two models disagree on the predicted class
        disagreement_count = 0
        for s in range(n_samples):
            classes_for_sample = {predicted_classes[n][s] for n in names}
            if len(classes_for_sample) > 1:
                disagreement_count += 1

        # At least some samples should show genuine disagreement
        assert disagreement_count > 0, (
            "All models agree on every sample -- unlikely with genuinely diverse models."
        )

    def test_residuals_not_perfectly_correlated(self, trained_regression_zoo, diabetes_data):
        """Residual correlation between model families should not be perfect."""
        _, X_test, _, y_test = diabetes_data
        residuals = collect_residuals(trained_regression_zoo, X_test, y_test)
        K = residuals.shape[0]

        if K < 2:
            pytest.skip("Need at least 2 models")

        # Compute correlation matrix
        corr = np.corrcoef(residuals)

        # Check off-diagonal elements: at least some should be < 0.99
        off_diag = []
        for i in range(K):
            for j in range(i + 1, K):
                off_diag.append(abs(corr[i, j]))

        mean_corr = np.mean(off_diag)
        min_corr = np.min(off_diag)

        assert mean_corr < 0.99, (
            f"Mean absolute residual correlation is {mean_corr:.4f}, "
            f"too high for genuinely diverse models."
        )
        assert min_corr < 0.98, (
            f"Minimum absolute residual correlation is {min_corr:.4f}, "
            f"suggesting models are clones."
        )

    def test_different_families_different_error_patterns(
        self, trained_classification_zoo, iris_data
    ):
        """Models from different families should make errors on different samples."""
        _, X_test, _, y_test = iris_data
        preds = collect_predictions_classification(trained_classification_zoo, X_test)

        # For each model, find which samples it gets wrong
        error_sets = {}
        for name, proba in preds.items():
            predicted = np.argmax(proba, axis=1)
            errors = set(np.where(predicted != y_test)[0])
            error_sets[name] = errors

        names = list(error_sets.keys())

        # Compute pairwise Jaccard similarity of error sets
        jaccards = []
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                s_i = error_sets[names[i]]
                s_j = error_sets[names[j]]
                if len(s_i) == 0 and len(s_j) == 0:
                    continue  # Both perfect, skip
                union = len(s_i | s_j)
                if union == 0:
                    continue
                jaccard = len(s_i & s_j) / union
                jaccards.append(jaccard)

        if len(jaccards) > 0:
            mean_jaccard = np.mean(jaccards)
            # If models were clones, Jaccard would be ~1.0
            # Diverse models should have substantially different error patterns
            assert mean_jaccard < 0.95, (
                f"Mean Jaccard similarity of error sets is {mean_jaccard:.4f}, "
                f"suggesting models make the same errors (not diverse)."
            )


# ============================================================
# 7. Integration with ICM framework
# ============================================================

class TestICMIntegration:
    """Test that model zoo outputs work with the ICM framework."""

    def test_classification_preds_with_compute_agreement(
        self, trained_classification_zoo, iris_data
    ):
        """Collected predictions should work with compute_agreement."""
        from framework.icm import compute_agreement

        _, X_test, _, _ = iris_data
        preds = collect_predictions_classification(trained_classification_zoo, X_test)

        # Average probabilities across samples for each model to get a
        # per-model distribution (how compute_agreement is commonly used)
        avg_preds = [proba.mean(axis=0) for proba in preds.values()]

        # Normalize to valid distributions
        avg_preds_norm = []
        for p in avg_preds:
            p = np.abs(p)
            p = p / (p.sum() + 1e-12)
            avg_preds_norm.append(p)

        A = compute_agreement(avg_preds_norm, distance_fn="hellinger")
        assert 0.0 <= A <= 1.0

    def test_residuals_with_dependency_penalty(
        self, trained_regression_zoo, diabetes_data
    ):
        """Collected residuals should work with compute_dependency_penalty."""
        from framework.icm import compute_dependency_penalty

        _, X_test, _, y_test = diabetes_data
        residuals = collect_residuals(trained_regression_zoo, X_test, y_test)
        Pi = compute_dependency_penalty(residuals=residuals)
        assert 0.0 <= Pi <= 1.0

    def test_full_icm_pipeline_classification(
        self, trained_classification_zoo, iris_data
    ):
        """Full ICM computation from classification zoo predictions."""
        from framework.icm import compute_agreement, compute_direction

        _, X_test, _, _ = iris_data
        preds = collect_predictions_classification(trained_classification_zoo, X_test)

        # Average per-model distributions
        avg_preds = []
        for proba in preds.values():
            p = proba.mean(axis=0)
            p = np.abs(p)
            p = p / (p.sum() + 1e-12)
            avg_preds.append(p)

        A = compute_agreement(avg_preds, distance_fn="hellinger")
        assert 0.0 <= A <= 1.0

        # Direction: sign of max-class probability - 0.5
        signs = np.array([np.sign(p.max() - 1.0 / len(p)) for p in avg_preds])
        D = compute_direction(signs)
        assert 0.0 <= D <= 1.0

    def test_full_icm_pipeline_regression(
        self, trained_regression_zoo, diabetes_data
    ):
        """Full ICM computation from regression zoo predictions."""
        from framework.icm import (
            compute_agreement,
            compute_uncertainty_overlap,
            compute_dependency_penalty,
        )

        _, X_test, _, y_test = diabetes_data
        preds = collect_predictions_regression(trained_regression_zoo, X_test)

        # Use mean predictions for agreement
        mean_preds = [arr[:, 0] for arr in preds.values()]
        A = compute_agreement(mean_preds, distance_fn="wasserstein")
        assert 0.0 <= A <= 1.0

        # Use quantile bounds for uncertainty overlap
        intervals = []
        for arr in preds.values():
            lower = float(np.percentile(arr[:, 1], 10))
            upper = float(np.percentile(arr[:, 2], 90))
            intervals.append((lower, upper))
        U = compute_uncertainty_overlap(intervals)
        assert 0.0 <= U <= 1.0

        # Use residuals for dependency penalty
        residuals = collect_residuals(trained_regression_zoo, X_test, y_test)
        Pi = compute_dependency_penalty(residuals=residuals)
        assert 0.0 <= Pi <= 1.0


# ============================================================
# 8. Seed determinism tests
# ============================================================

class TestSeedDeterminism:
    """Test that the same seed produces the same zoo."""

    def test_classification_zoo_deterministic(self):
        zoo1 = build_classification_zoo(seed=123)
        zoo2 = build_classification_zoo(seed=123)
        assert len(zoo1) == len(zoo2)
        for m1, m2 in zip(zoo1, zoo2):
            assert m1.name == m2.name
            assert m1.family == m2.family

    def test_regression_zoo_deterministic(self):
        zoo1 = build_regression_zoo(seed=123)
        zoo2 = build_regression_zoo(seed=123)
        assert len(zoo1) == len(zoo2)
        for m1, m2 in zip(zoo1, zoo2):
            assert m1.name == m2.name
            assert m1.family == m2.family

    def test_different_seeds_same_structure(self):
        """Different seeds should produce same model structure, different weights."""
        zoo1 = build_classification_zoo(seed=1)
        zoo2 = build_classification_zoo(seed=2)
        assert len(zoo1) == len(zoo2)
        for m1, m2 in zip(zoo1, zoo2):
            assert m1.name == m2.name  # Same names


# ============================================================
# 9. Edge cases
# ============================================================

class TestEdgeCases:
    """Edge case tests for model zoo and datasets."""

    def test_single_model_classification_collection(self, iris_data):
        """Collecting predictions from a single model should work."""
        X_train, X_test, y_train, _ = iris_data
        zoo = build_classification_zoo(seed=42)[:1]
        train_zoo(zoo, X_train, y_train)
        preds = collect_predictions_classification(zoo, X_test)
        assert len(preds) == 1

    def test_single_model_regression_collection(self, diabetes_data):
        """Collecting predictions from a single model should work."""
        X_train, X_test, y_train, _ = diabetes_data
        zoo = build_regression_zoo(seed=42)[:1]
        train_zoo(zoo, X_train, y_train)
        preds = collect_predictions_regression(zoo, X_test)
        assert len(preds) == 1

    def test_breast_cancer_binary(self):
        """Binary classification should work correctly."""
        X_train, X_test, y_train, y_test = load_dataset("breast_cancer", seed=42)
        zoo = build_classification_zoo(seed=42)
        train_zoo(zoo, X_train, y_train)
        preds = collect_predictions_classification(zoo, X_test)
        for name, proba in preds.items():
            assert proba.shape[1] == 2, (
                f"{name}: binary problem should have 2 columns, got {proba.shape[1]}"
            )

    def test_digits_multiclass(self):
        """10-class classification should work correctly."""
        X_train, X_test, y_train, y_test = load_dataset("digits", seed=42)
        zoo = build_classification_zoo(seed=42)
        train_zoo(zoo, X_train, y_train)
        preds = collect_predictions_classification(zoo, X_test)
        for name, proba in preds.items():
            assert proba.shape[1] == 10, (
                f"{name}: digits should have 10 columns, got {proba.shape[1]}"
            )
            row_sums = proba.sum(axis=1)
            np.testing.assert_allclose(
                row_sums, 1.0, atol=1e-5,
                err_msg=f"{name}: probabilities do not sum to 1 on digits",
            )

    def test_repr_classification(self):
        zoo = build_classification_zoo(seed=42)
        for model in zoo:
            r = repr(model)
            assert "ClassificationWrapper" in r
            assert model.name in r

    def test_repr_regression(self):
        zoo = build_regression_zoo(seed=42)
        for model in zoo:
            r = repr(model)
            assert "RegressionWrapper" in r
            assert model.name in r
