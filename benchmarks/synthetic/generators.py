"""Synthetic data generators for OS Multi-Science benchmarks.

Four generators that produce controlled datasets for validating the ICM
engine, early-warning system, and full pipeline.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


# ============================================================
# 1. Change-point time series
# ============================================================

def generate_change_point_series(
    n_samples: int = 500,
    n_features: int = 5,
    change_at: int = 250,
    seed: int = 42,
) -> tuple[NDArray, int]:
    """Generate multivariate time series with a single structural change-point.

    Before ``change_at``: features are drawn from N(0, 1).
    After  ``change_at``: features shift by a per-feature amount in [0.5, 2.0].

    Parameters
    ----------
    n_samples : int
        Total number of time steps.
    n_features : int
        Dimensionality of the feature vector at each step.
    change_at : int
        Index where the distribution shift occurs.
    seed : int
        Random seed.

    Returns
    -------
    (X, change_point_index)
        X has shape (n_samples, n_features).
    """
    rng = np.random.default_rng(seed)

    X = rng.standard_normal((n_samples, n_features))

    # Per-feature shift magnitudes: linspace from 0.5 to 2.0
    shifts = np.linspace(0.5, 2.0, n_features)
    X[change_at:] += shifts[np.newaxis, :]

    return X, change_at


# ============================================================
# 2. Network cascade (contagion / default cascade)
# ============================================================

def generate_network_cascade(
    n_nodes: int = 100,
    edge_prob: float = 0.05,
    threshold: float = 0.3,
    n_steps: int = 50,
    seed: int = 42,
) -> tuple[NDArray, NDArray, int]:
    """Generate an Erdos-Renyi network with a cascade/contagion process.

    Binary state: 0 = healthy, 1 = defaulted.  At each step a healthy
    node defaults if the fraction of its defaulted neighbours exceeds
    ``threshold``.  5 % of nodes are seeded as initially defaulted.

    Parameters
    ----------
    n_nodes : int
        Number of nodes.
    edge_prob : float
        Erdos-Renyi edge probability.
    threshold : float
        Fraction of defaulted neighbours required to trigger default.
    n_steps : int
        Number of discrete time steps to simulate.
    seed : int
        Random seed.

    Returns
    -------
    (adjacency_matrix, state_history, tipping_step)
        adjacency_matrix : (n_nodes, n_nodes) binary symmetric matrix.
        state_history    : (n_steps + 1, n_nodes) binary matrix (includes t=0).
        tipping_step     : first time step where > 50 % of nodes are defaulted,
                           or -1 if tipping never occurs.
    """
    rng = np.random.default_rng(seed)

    # Erdos-Renyi adjacency (symmetric, no self-loops)
    upper = rng.random((n_nodes, n_nodes)) < edge_prob
    adj = np.triu(upper, k=1)
    adj = adj | adj.T
    adj = adj.astype(np.float64)

    # Degree vector (avoid division by zero for isolated nodes)
    degree = adj.sum(axis=1)

    # Initial state: 5 % seeded defaults
    n_initial = max(1, int(0.05 * n_nodes))
    state = np.zeros(n_nodes, dtype=np.float64)
    initial_defaults = rng.choice(n_nodes, size=n_initial, replace=False)
    state[initial_defaults] = 1.0

    state_history = np.zeros((n_steps + 1, n_nodes), dtype=np.float64)
    state_history[0] = state.copy()

    tipping_step = -1

    for t in range(1, n_steps + 1):
        # Fraction of defaulted neighbours for each node
        n_defaulted_neighbours = adj @ state
        safe_degree = np.where(degree > 0, degree, 1.0)
        frac_defaulted = n_defaulted_neighbours / safe_degree

        # Healthy nodes that exceed threshold default
        new_defaults = (state == 0) & (frac_defaulted > threshold)
        state = np.where(new_defaults, 1.0, state)
        state_history[t] = state.copy()

        if tipping_step == -1 and state.mean() > 0.5:
            tipping_step = t

    return adj, state_history, tipping_step


# ============================================================
# 3. Classification benchmark (Gaussian mixture)
# ============================================================

def generate_classification_benchmark(
    n_samples: int = 1000,
    n_classes: int = 3,
    noise: float = 0.1,
    seed: int = 42,
) -> tuple[NDArray, NDArray, NDArray]:
    """Generate a Gaussian-mixture classification dataset.

    Class centres are placed at the vertices of a regular simplex in
    2-D (for n_classes <= 4) or on the unit sphere in n_classes - 1
    dimensions.

    Parameters
    ----------
    n_samples : int
        Total number of samples (split roughly equally across classes).
    n_classes : int
        Number of classes (>= 2).
    noise : float
        Standard deviation of isotropic Gaussian noise around centres.
    seed : int
        Random seed.

    Returns
    -------
    (X, y_true, class_centers)
        X            : (n_samples, d) feature matrix.
        y_true       : (n_samples,) integer class labels.
        class_centers : (n_classes, d) centre coordinates.
    """
    rng = np.random.default_rng(seed)

    # Place centres on a 2-D circle (works well for visualization)
    d = 2
    angles = np.linspace(0, 2 * np.pi, n_classes, endpoint=False)
    class_centers = np.column_stack([np.cos(angles), np.sin(angles)])
    # Scale so inter-class distance is comfortably larger than noise
    class_centers *= 3.0

    samples_per_class = n_samples // n_classes
    remainder = n_samples - samples_per_class * n_classes

    X_parts: list[NDArray] = []
    y_parts: list[NDArray] = []

    for c in range(n_classes):
        n_c = samples_per_class + (1 if c < remainder else 0)
        X_c = rng.normal(loc=class_centers[c], scale=noise, size=(n_c, d))
        X_parts.append(X_c)
        y_parts.append(np.full(n_c, c, dtype=int))

    X = np.vstack(X_parts)
    y_true = np.concatenate(y_parts)

    # Shuffle
    perm = rng.permutation(len(y_true))
    X = X[perm]
    y_true = y_true[perm]

    return X, y_true, class_centers


# ============================================================
# 4. Multi-model predictions
# ============================================================

def generate_multi_model_predictions(
    X: NDArray,
    y_true: NDArray,
    n_agreeing: int = 3,
    n_disagreeing: int = 1,
    noise_agree: float = 0.05,
    noise_disagree: float = 0.5,
    seed: int = 42,
) -> dict[str, NDArray]:
    """Create synthetic class-probability predictions from multiple models.

    Agreeing models produce predictions close to the one-hot truth with
    small noise.  Disagreeing models add large noise or bias.

    Parameters
    ----------
    X : (n_samples, d)
        Feature matrix (used only for its shape here).
    y_true : (n_samples,) integer class labels.
    n_agreeing : int
        Number of models that closely agree with the truth.
    n_disagreeing : int
        Number of models that diverge substantially.
    noise_agree : float
        Noise magnitude for agreeing models.
    noise_disagree : float
        Noise magnitude for disagreeing models.
    seed : int
        Random seed.

    Returns
    -------
    dict of {model_name: predictions}
        Each predictions array has shape (n_samples, n_classes) and rows
        sum to 1 (valid probability distributions).
    """
    rng = np.random.default_rng(seed)
    n_samples = len(y_true)
    n_classes = int(y_true.max()) + 1

    # One-hot ground truth
    onehot = np.zeros((n_samples, n_classes), dtype=np.float64)
    onehot[np.arange(n_samples), y_true] = 1.0

    predictions: dict[str, NDArray] = {}

    for i in range(n_agreeing):
        noisy = onehot + rng.normal(0, noise_agree, onehot.shape)
        noisy = np.abs(noisy)  # keep non-negative
        noisy /= noisy.sum(axis=1, keepdims=True)
        predictions[f"agree_{i}"] = noisy

    for i in range(n_disagreeing):
        noisy = onehot + rng.normal(0, noise_disagree, onehot.shape)
        # Add a systematic bias toward a random class
        bias_class = rng.integers(0, n_classes)
        noisy[:, bias_class] += 0.3
        noisy = np.abs(noisy)
        noisy /= noisy.sum(axis=1, keepdims=True)
        predictions[f"disagree_{i}"] = noisy

    return predictions
