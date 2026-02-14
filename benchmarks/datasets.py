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
    make_moons,
    make_circles,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ============================================================
# Synthetic dataset generators (return Bunch-like objects)
# ============================================================

class _BunchLike:
    """Minimal Bunch-like container compatible with existing loader pattern."""
    def __init__(self, data: NDArray, target: NDArray) -> None:
        self.data = data
        self.target = target


def _make_moons_loader():
    """Generate make_moons dataset: 2D, 2 classes, 1000 samples with noise."""
    X, y = make_moons(n_samples=1000, noise=0.2, random_state=42)
    return _BunchLike(X, y)


def _make_circles_loader():
    """Generate make_circles dataset: 2D, 2 classes, 1000 samples with noise."""
    X, y = make_circles(n_samples=1000, noise=0.1, factor=0.5, random_state=42)
    return _BunchLike(X, y)


def _olivetti_faces_loader():
    """Load Olivetti Faces with PCA to 50 dims: 40 classes, 400 samples.

    Uses PCA to reduce from 4096 to 50 features for tractable model
    training (SVM with 40 classes on 4096 features is very slow).
    """
    from sklearn.datasets import fetch_olivetti_faces
    from sklearn.decomposition import PCA

    data = fetch_olivetti_faces()
    # Reduce dimensionality for speed: 4096 -> 50 via PCA
    pca = PCA(n_components=50, random_state=42)
    X_reduced = pca.fit_transform(data.data)
    return _BunchLike(X_reduced, data.target)


def _covertype_loader():
    """Load Covertype dataset (first 2000 samples): 54 features, 7 classes."""
    try:
        from sklearn.datasets import fetch_covtype
        data = fetch_covtype()
        # Subsample to keep benchmark fast: first 2000 samples
        X = data.data[:2000]
        y = data.target[:2000]
        # Covertype target is 1-indexed (1..7), convert to 0-indexed for consistency
        y = y - 1
        return _BunchLike(X, y)
    except Exception:
        # If fetch_covtype is unavailable or download fails, return None
        return None


def _concept_drift_loader():
    """Generate synthetic concept drift dataset.

    Training data: clean 3-class Gaussian clusters.
    Test data: shifted distribution (means translated by 1.5 std).

    This simulates covariate shift / concept drift to stress-test
    model robustness and ICM's ability to detect degraded convergence.
    """
    rng = np.random.default_rng(42)

    n_per_class = 200
    n_features = 5
    n_classes = 3

    # Training distribution: well-separated Gaussian clusters
    centers_train = np.array([
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [3.0, 3.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 3.0, 3.0, 0.0],
    ])

    X_parts, y_parts = [], []
    for c in range(n_classes):
        X_c = rng.normal(loc=centers_train[c], scale=0.8, size=(n_per_class, n_features))
        y_c = np.full(n_per_class, c, dtype=np.int64)
        X_parts.append(X_c)
        y_parts.append(y_c)

    X = np.vstack(X_parts)
    y = np.concatenate(y_parts)

    return _BunchLike(X, y)


# ============================================================
# Special loader for concept drift (overrides normal load_dataset)
# ============================================================

def _load_concept_drift(seed: int = 42, test_size: float = 0.2):
    """Load concept drift dataset with shifted test distribution.

    Unlike normal datasets, the test set is drawn from a shifted
    distribution to simulate concept drift.
    """
    rng = np.random.default_rng(seed)

    n_per_class_train = 160
    n_per_class_test = 40
    n_features = 5
    n_classes = 3

    centers_train = np.array([
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [3.0, 3.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 3.0, 3.0, 0.0],
    ])

    # Shift for test distribution: translate means by 1.5 units
    drift_vector = np.array([1.5, -1.0, 0.5, -0.5, 1.0])
    centers_test = centers_train + drift_vector

    # Generate training data (clean distribution)
    X_train_parts, y_train_parts = [], []
    for c in range(n_classes):
        X_c = rng.normal(loc=centers_train[c], scale=0.8,
                         size=(n_per_class_train, n_features))
        y_c = np.full(n_per_class_train, c, dtype=np.int64)
        X_train_parts.append(X_c)
        y_train_parts.append(y_c)

    X_train = np.vstack(X_train_parts)
    y_train = np.concatenate(y_train_parts)

    # Generate test data (shifted distribution)
    X_test_parts, y_test_parts = [], []
    for c in range(n_classes):
        X_c = rng.normal(loc=centers_test[c], scale=0.8,
                         size=(n_per_class_test, n_features))
        y_c = np.full(n_per_class_test, c, dtype=np.int64)
        X_test_parts.append(X_c)
        y_test_parts.append(y_c)

    X_test = np.vstack(X_test_parts)
    y_test = np.concatenate(y_test_parts)

    # Shuffle
    train_idx = rng.permutation(len(y_train))
    test_idx = rng.permutation(len(y_test))
    X_train, y_train = X_train[train_idx], y_train[train_idx]
    X_test, y_test = X_test[test_idx], y_test[test_idx]

    # Standardize (fit on train only)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test


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
    "moons": {
        "loader": _make_moons_loader,
        "description": "Moons (2 features, 2 classes, 1000 samples, nonlinear boundary)",
        "task": "classification",
    },
    "circles": {
        "loader": _make_circles_loader,
        "description": "Circles (2 features, 2 classes, 1000 samples, circular boundary)",
        "task": "classification",
    },
    "concept_drift": {
        "loader": _concept_drift_loader,
        "description": "Concept Drift (5 features, 3 classes, 600 samples, shifted test distribution)",
        "task": "classification",
        "custom_loader": True,
    },
}

# Conditionally add covertype and olivetti_faces
# Covertype: try importing to check availability
try:
    from sklearn.datasets import fetch_covtype  # noqa: F401
    _CLASSIFICATION_DATASETS["covertype"] = {
        "loader": _covertype_loader,
        "description": "Covertype (54 features, 7 classes, 2000 samples subset)",
        "task": "classification",
    }
except ImportError:
    pass

# Olivetti Faces: always available in sklearn
try:
    from sklearn.datasets import fetch_olivetti_faces  # noqa: F401
    _CLASSIFICATION_DATASETS["olivetti_faces"] = {
        "loader": _olivetti_faces_loader,
        "description": "Olivetti Faces (50 PCA features, 40 classes, 400 samples)",
        "task": "classification",
    }
except ImportError:
    pass


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

    # Concept drift dataset has a custom loader that generates
    # train/test from different distributions
    if entry.get("custom_loader"):
        if name == "concept_drift":
            return _load_concept_drift(seed=seed, test_size=test_size)

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
