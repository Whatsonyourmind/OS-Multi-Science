"""Model Zoo -- Genuinely epistemically diverse model families.

Provides classification and regression model wrappers around sklearn
estimators, each representing a fundamentally different epistemic
approach to learning from data.  These are NOT noise-perturbed variants
of a single prediction; each family encodes qualitatively different
inductive biases (linear boundaries, axis-aligned splits, kernel
margins, instance-based reasoning, conditional independence, etc.).

All public convenience functions return data in ICM-compatible formats
(see ``framework.icm``).
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, ElasticNet
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
)
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


# ============================================================
# Wrapper base classes
# ============================================================

@runtime_checkable
class ClassificationModel(Protocol):
    """Protocol for a classification model wrapper."""

    name: str
    family: str

    def fit(self, X: NDArray, y: NDArray) -> None: ...
    def predict_proba(self, X: NDArray) -> NDArray: ...


@runtime_checkable
class RegressionModel(Protocol):
    """Protocol for a regression model wrapper."""

    name: str
    family: str

    def fit(self, X: NDArray, y: NDArray) -> None: ...
    def predict_quantiles(self, X: NDArray) -> NDArray: ...


class ClassificationWrapper:
    """Thin wrapper around an sklearn classifier.

    Attributes
    ----------
    name : str
        Human-readable name of the model instance.
    family : str
        Epistemic family this model belongs to (e.g. "linear",
        "tree", "kernel", etc.).
    """

    def __init__(self, estimator, name: str, family: str) -> None:
        self._estimator = estimator
        self.name = name
        self.family = family
        self._classes: NDArray | None = None

    def fit(self, X: NDArray, y: NDArray) -> None:
        """Train the underlying estimator."""
        self._estimator.fit(X, y)
        self._classes = self._estimator.classes_

    def predict_proba(self, X: NDArray) -> NDArray:
        """Return class probabilities of shape (n_samples, n_classes).

        For estimators that do not natively support ``predict_proba``
        (e.g. SVM with ``probability=False``), the decision function
        is converted to probabilities via a softmax.
        """
        if hasattr(self._estimator, "predict_proba"):
            return self._estimator.predict_proba(X)

        # Fallback: decision_function -> softmax
        decision = self._estimator.decision_function(X)
        if decision.ndim == 1:
            # Binary case: convert to 2-column
            decision = np.column_stack([-decision, decision])
        # Softmax
        exp_d = np.exp(decision - decision.max(axis=1, keepdims=True))
        return exp_d / exp_d.sum(axis=1, keepdims=True)

    def __repr__(self) -> str:
        return f"ClassificationWrapper(name={self.name!r}, family={self.family!r})"


class RegressionWrapper:
    """Thin wrapper around an sklearn regressor.

    Provides ``predict_quantiles`` which returns a matrix of shape
    (n_samples, 3) with columns [mean, lower_10, upper_90].

    For non-ensemble regressors, quantile estimates are derived from
    the training residual distribution.

    Attributes
    ----------
    name : str
        Human-readable name of the model instance.
    family : str
        Epistemic family this model belongs to.
    """

    def __init__(self, estimator, name: str, family: str) -> None:
        self._estimator = estimator
        self.name = name
        self.family = family
        self._residual_q10: float = 0.0
        self._residual_q90: float = 0.0

    def fit(self, X: NDArray, y: NDArray) -> None:
        """Train the underlying estimator and cache residual quantiles."""
        self._estimator.fit(X, y)
        # Compute training residuals for quantile estimation
        y_pred_train = self._estimator.predict(X)
        residuals = y - y_pred_train
        self._residual_q10 = float(np.percentile(residuals, 10))
        self._residual_q90 = float(np.percentile(residuals, 90))

    def predict_quantiles(self, X: NDArray) -> NDArray:
        """Return (n_samples, 3) array with [mean, lower_10, upper_90].

        The lower and upper bounds are estimated by adding the 10th and
        90th percentile of the training residuals to the point prediction.
        """
        mean = self._estimator.predict(X)
        lower = mean + self._residual_q10
        upper = mean + self._residual_q90
        return np.column_stack([mean, lower, upper])

    def __repr__(self) -> str:
        return f"RegressionWrapper(name={self.name!r}, family={self.family!r})"


# ============================================================
# Zoo builders
# ============================================================

def build_classification_zoo(seed: int = 42) -> list[ClassificationWrapper]:
    """Return a list of 8+ genuinely diverse classification models.

    Each model represents a fundamentally different inductive bias:

    1. **Logistic Regression** -- linear decision boundary in feature space.
    2. **Decision Tree** -- axis-aligned recursive splits.
    3. **Random Forest** -- bagged ensemble of decorrelated trees.
    4. **Gradient Boosting** -- sequentially boosted shallow trees.
    5. **K-Nearest Neighbors** -- instance-based / lazy learning.
    6. **MLP Neural Network** -- nonlinear function approximation
       via layered transformations.
    7. **Gaussian Naive Bayes** -- assumes conditional independence
       of features given the class.
    8. **SVM (RBF kernel)** -- kernel-based maximum-margin classifier.

    Parameters
    ----------
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    list[ClassificationWrapper]
    """
    return [
        ClassificationWrapper(
            LogisticRegression(
                max_iter=1000,
                random_state=seed,
                solver="lbfgs",
            ),
            name="LogisticRegression",
            family="linear",
        ),
        ClassificationWrapper(
            DecisionTreeClassifier(
                random_state=seed,
                max_depth=10,
            ),
            name="DecisionTree",
            family="tree",
        ),
        ClassificationWrapper(
            RandomForestClassifier(
                n_estimators=100,
                random_state=seed,
                max_depth=10,
            ),
            name="RandomForest",
            family="ensemble_bagging",
        ),
        ClassificationWrapper(
            GradientBoostingClassifier(
                n_estimators=100,
                random_state=seed,
                max_depth=3,
                learning_rate=0.1,
            ),
            name="GradientBoosting",
            family="ensemble_boosting",
        ),
        ClassificationWrapper(
            KNeighborsClassifier(
                n_neighbors=5,
            ),
            name="KNearestNeighbors",
            family="instance_based",
        ),
        ClassificationWrapper(
            MLPClassifier(
                hidden_layer_sizes=(64, 32),
                max_iter=500,
                random_state=seed,
                early_stopping=True,
                validation_fraction=0.15,
            ),
            name="MLPNeuralNetwork",
            family="neural_network",
        ),
        ClassificationWrapper(
            GaussianNB(),
            name="GaussianNaiveBayes",
            family="probabilistic",
        ),
        ClassificationWrapper(
            SVC(
                kernel="rbf",
                probability=True,
                random_state=seed,
                gamma="scale",
            ),
            name="SVM_RBF",
            family="kernel",
        ),
    ]


def build_regression_zoo(seed: int = 42) -> list[RegressionWrapper]:
    """Return a list of 8+ genuinely diverse regression models.

    Each model represents a fundamentally different inductive bias:

    1. **Linear Regression** -- ordinary least squares.
    2. **Decision Tree Regressor** -- axis-aligned recursive splits.
    3. **Random Forest Regressor** -- bagged ensemble of trees.
    4. **Gradient Boosting Regressor** -- sequentially boosted trees.
    5. **KNN Regressor** -- instance-based / lazy learning.
    6. **MLP Regressor** -- nonlinear function approximation.
    7. **Ridge Regression** -- L2-regularized linear model.
    8. **ElasticNet** -- L1 + L2 combined regularization.

    Parameters
    ----------
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    list[RegressionWrapper]
    """
    return [
        RegressionWrapper(
            LinearRegression(),
            name="LinearRegression",
            family="linear",
        ),
        RegressionWrapper(
            DecisionTreeRegressor(
                random_state=seed,
                max_depth=10,
            ),
            name="DecisionTreeRegressor",
            family="tree",
        ),
        RegressionWrapper(
            RandomForestRegressor(
                n_estimators=100,
                random_state=seed,
                max_depth=10,
            ),
            name="RandomForestRegressor",
            family="ensemble_bagging",
        ),
        RegressionWrapper(
            GradientBoostingRegressor(
                n_estimators=100,
                random_state=seed,
                max_depth=3,
                learning_rate=0.1,
            ),
            name="GradientBoostingRegressor",
            family="ensemble_boosting",
        ),
        RegressionWrapper(
            KNeighborsRegressor(
                n_neighbors=5,
            ),
            name="KNNRegressor",
            family="instance_based",
        ),
        RegressionWrapper(
            MLPRegressor(
                hidden_layer_sizes=(64, 32),
                max_iter=500,
                random_state=seed,
                early_stopping=True,
                validation_fraction=0.15,
            ),
            name="MLPRegressor",
            family="neural_network",
        ),
        RegressionWrapper(
            Ridge(
                alpha=1.0,
                random_state=seed,
            ),
            name="RidgeRegression",
            family="regularized_linear",
        ),
        RegressionWrapper(
            ElasticNet(
                alpha=0.1,
                l1_ratio=0.5,
                random_state=seed,
                max_iter=2000,
            ),
            name="ElasticNet",
            family="regularized_linear_l1l2",
        ),
    ]


# ============================================================
# Training and prediction collection
# ============================================================

def train_zoo(
    models: list[ClassificationWrapper] | list[RegressionWrapper],
    X_train: NDArray,
    y_train: NDArray,
) -> None:
    """Train all models in the zoo on the given data.

    Parameters
    ----------
    models : list of model wrappers (classification or regression).
    X_train : feature matrix (n_samples, n_features).
    y_train : target vector (n_samples,).
    """
    for model in models:
        model.fit(X_train, y_train)


def collect_predictions_classification(
    models: list[ClassificationWrapper],
    X: NDArray,
) -> dict[str, NDArray]:
    """Collect probability predictions from all classification models.

    Returns a dict compatible with ``framework.icm.compute_icm_from_predictions``
    and ``framework.icm.compute_agreement``.

    Parameters
    ----------
    models : list of trained ClassificationWrapper instances.
    X : feature matrix (n_samples, n_features).

    Returns
    -------
    dict[str, np.ndarray]
        Mapping ``model.name -> probabilities`` where probabilities
        has shape ``(n_samples, n_classes)``.
    """
    result: dict[str, NDArray] = {}
    for model in models:
        proba = model.predict_proba(X)
        result[model.name] = proba
    return result


def collect_predictions_regression(
    models: list[RegressionWrapper],
    X: NDArray,
) -> dict[str, NDArray]:
    """Collect point + interval predictions from all regression models.

    Parameters
    ----------
    models : list of trained RegressionWrapper instances.
    X : feature matrix (n_samples, n_features).

    Returns
    -------
    dict[str, np.ndarray]
        Mapping ``model.name -> quantiles`` where quantiles has shape
        ``(n_samples, 3)`` with columns [mean, lower_10, upper_90].
    """
    result: dict[str, NDArray] = {}
    for model in models:
        quantiles = model.predict_quantiles(X)
        result[model.name] = quantiles
    return result


def collect_residuals(
    models: list[RegressionWrapper],
    X: NDArray,
    y_true: NDArray,
) -> NDArray:
    """Compute residuals matrix [K x N] for the Pi (dependency penalty) component.

    Parameters
    ----------
    models : list of K trained RegressionWrapper instances.
    X : feature matrix (n_samples, n_features).
    y_true : true target values (n_samples,).

    Returns
    -------
    np.ndarray
        Shape ``(K, N)`` where ``K = len(models)`` and ``N = len(y_true)``.
        Each row is ``y_true - model.predict(X)``.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    K = len(models)
    N = len(y_true)
    residuals = np.empty((K, N), dtype=np.float64)
    for i, model in enumerate(models):
        y_pred = model.predict_quantiles(X)[:, 0]  # mean column
        residuals[i] = y_true - y_pred
    return residuals
