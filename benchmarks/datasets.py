"""Dataset Loader -- Real public datasets from sklearn.

Provides standardized, reproducible train/test splits for benchmarking
epistemically diverse model ensembles against the ICM framework.

All features are standardized (zero mean, unit variance) using only the
training set statistics (to prevent data leakage).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from sklearn.datasets import (
    load_iris,
    load_wine,
    load_breast_cancer,
    load_digits,
    fetch_california_housing,
    load_diabetes,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ============================================================
# Registry of available datasets
# ============================================================

_CLASSIFICATION_DATASETS = {
    "iris": {
        "loader": load_iris,
        "description": "Iris (4 features, 3 classes, 150 samples)",
        "task": "classification",
    },
    "wine": {
        "loader": load_wine,
        "description": "Wine (13 features, 3 classes, 178 samples)",
        "task": "classification",
    },
    "breast_cancer": {
        "loader": load_breast_cancer,
        "description": "Breast Cancer (30 features, 2 classes, 569 samples)",
        "task": "classification",
    },
    "digits": {
        "loader": load_digits,
        "description": "Digits (64 features, 10 classes, 1797 samples)",
        "task": "classification",
    },
}

_REGRESSION_DATASETS = {
    "california_housing": {
        "loader": fetch_california_housing,
        "description": "California Housing (8 features, 20640 samples)",
        "task": "regression",
    },
    "diabetes": {
        "loader": load_diabetes,
        "description": "Diabetes (10 features, 442 samples)",
        "task": "regression",
    },
}

_ALL_DATASETS = {**_CLASSIFICATION_DATASETS, **_REGRESSION_DATASETS}


# ============================================================
# Public API
# ============================================================

def list_datasets() -> list[str]:
    """Return sorted list of all available dataset names.

    Returns
    -------
    list[str]
        Names that can be passed to ``load_dataset``.
    """
    return sorted(_ALL_DATASETS.keys())


def list_classification_datasets() -> list[str]:
    """Return sorted list of classification dataset names."""
    return sorted(_CLASSIFICATION_DATASETS.keys())


def list_regression_datasets() -> list[str]:
    """Return sorted list of regression dataset names."""
    return sorted(_REGRESSION_DATASETS.keys())


def get_dataset_info(name: str) -> dict[str, str]:
    """Return metadata dict for a dataset.

    Parameters
    ----------
    name : str
        Dataset name (one of ``list_datasets()``).

    Returns
    -------
    dict with keys ``"description"`` and ``"task"``.

    Raises
    ------
    ValueError
        If *name* is not a recognized dataset.
    """
    if name not in _ALL_DATASETS:
        raise ValueError(
            f"Unknown dataset: {name!r}. "
            f"Available: {list_datasets()}"
        )
    entry = _ALL_DATASETS[name]
    return {"description": entry["description"], "task": entry["task"]}


def load_dataset(
    name: str,
    seed: int = 42,
    test_size: float = 0.2,
) -> tuple[NDArray, NDArray, NDArray, NDArray]:
    """Load and split a real dataset with standardized features.

    Features are standardized (zero mean, unit variance) using
    statistics computed **only** on the training split to prevent
    data leakage.

    Parameters
    ----------
    name : str
        Dataset name (one of ``list_datasets()``).
    seed : int
        Random seed for the train/test split.
    test_size : float
        Fraction of data reserved for testing (default 0.2).

    Returns
    -------
    tuple of (X_train, X_test, y_train, y_test)
        X arrays are np.float64 with shape (n_samples, n_features).
        y arrays are np.int64 (classification) or np.float64 (regression).

    Raises
    ------
    ValueError
        If *name* is not a recognized dataset.
    """
    if name not in _ALL_DATASETS:
        raise ValueError(
            f"Unknown dataset: {name!r}. "
            f"Available: {list_datasets()}"
        )

    entry = _ALL_DATASETS[name]
    data = entry["loader"]()
    X = np.asarray(data.data, dtype=np.float64)
    y = np.asarray(data.target)

    # Ensure correct dtype for target
    if entry["task"] == "classification":
        y = y.astype(np.int64)
    else:
        y = y.astype(np.float64)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y if entry["task"] == "classification" else None,
    )

    # Standardize features (fit on train only)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test
