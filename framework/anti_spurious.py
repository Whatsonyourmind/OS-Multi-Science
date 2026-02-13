"""Anti-spurious convergence tests for OS Multi-Science.

Guards against convergence that arises from shared biases, data leakage,
or correlated errors rather than genuine epistemic agreement.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.stats import rankdata

from framework.config import AntiSpuriousConfig
from framework.types import AntiSpuriousReport


# ---------------------------------------------------------------------------
# Negative controls
# ---------------------------------------------------------------------------

def generate_negative_controls(
    predictions: np.ndarray,
    labels: np.ndarray,
    features: np.ndarray,
    method: str = "label_shuffle",
    n_controls: int = 100,
    seed: int = 42,
) -> list[np.ndarray]:
    """Generate negative-control prediction sets that destroy genuine signal.

    Parameters
    ----------
    predictions : array of shape (n_models, n_samples)
        Model predictions.
    labels : array of shape (n_samples,)
        True labels / targets.
    features : array of shape (n_samples, d)
        Input features.
    method : {'label_shuffle', 'feature_shuffle', 'target_delay'}
        How to destroy signal.
    n_controls : int
        Number of negative-control sets to generate.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    list of arrays, each of shape (n_models, n_samples)
        Permuted / disrupted prediction sets.
    """
    predictions = np.asarray(predictions, dtype=float)
    labels = np.asarray(labels, dtype=float)
    features = np.asarray(features, dtype=float)
    n_models, n_samples = predictions.shape
    controls: list[np.ndarray] = []
    rng = np.random.default_rng(seed)

    for _ in range(n_controls):
        if method == "label_shuffle":
            # Shuffle each model independently to break cross-model alignment
            ctrl = np.empty_like(predictions)
            for m in range(n_models):
                ctrl[m] = predictions[m, rng.permutation(n_samples)]
        elif method == "feature_shuffle":
            # Shuffle feature indices (shared permutation across models).
            # This destroys the feature-prediction mapping while preserving
            # inter-model correlation structure.
            perm = rng.permutation(n_samples)
            ctrl = np.empty_like(predictions)
            for m in range(n_models):
                ctrl[m] = predictions[m, perm]
        elif method == "target_delay":
            # Apply different shifts per model
            ctrl = np.empty_like(predictions)
            for m in range(n_models):
                shift = rng.integers(1, max(2, n_samples // 2))
                ctrl[m] = np.roll(predictions[m], shift)
        else:
            raise ValueError(f"Unknown method: {method}")
        controls.append(ctrl)

    return controls


# ---------------------------------------------------------------------------
# Baseline distance D_0
# ---------------------------------------------------------------------------

def _mean_pairwise_distance(predictions: np.ndarray) -> float:
    """Mean L2 pairwise distance across models for a prediction matrix."""
    # predictions: (n_models, n_samples)
    dists = pdist(predictions, metric="euclidean")
    return float(np.mean(dists)) if len(dists) > 0 else 0.0


def estimate_d0(negative_control_distances: np.ndarray) -> float:
    """Estimate baseline D_0 from negative-control distances.

    Parameters
    ----------
    negative_control_distances : array of shape (n_controls,)
        Mean pairwise distances under each negative control.

    Returns
    -------
    float
        D_0 = mean of negative-control distances.
    """
    return float(np.mean(negative_control_distances))


# ---------------------------------------------------------------------------
# Normalized convergence
# ---------------------------------------------------------------------------

def normalize_convergence(D_observed: float, D_0: float) -> float:
    """Normalized convergence C(x) = exp(-D / D_0).

    Parameters
    ----------
    D_observed : float
        Observed mean pairwise distance.
    D_0 : float
        Baseline distance from negative controls.

    Returns
    -------
    float
        Convergence score in (0, 1].
    """
    if D_0 <= 0:
        return 0.0
    return float(np.exp(-D_observed / D_0))


# ---------------------------------------------------------------------------
# HSIC independence test
# ---------------------------------------------------------------------------

def _rbf_kernel(X: np.ndarray, bandwidth: float | None = None) -> np.ndarray:
    """Compute RBF (Gaussian) kernel matrix."""
    sq = squareform(pdist(X.reshape(-1, 1) if X.ndim == 1 else X, "sqeuclidean"))
    if bandwidth is None:
        bandwidth = float(np.median(sq[sq > 0])) if np.any(sq > 0) else 1.0
    return np.exp(-sq / (2.0 * bandwidth))


def _center_kernel(K: np.ndarray) -> np.ndarray:
    """Center a kernel matrix in feature space."""
    n = K.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    return H @ K @ H


def hsic_test(
    residuals_1: np.ndarray,
    residuals_2: np.ndarray,
    n_permutations: int = 1000,
    seed: int = 42,
) -> tuple[float, float]:
    """HSIC independence test between two residual vectors.

    Uses an RBF kernel with median-heuristic bandwidth and a
    permutation-based null distribution.

    Parameters
    ----------
    residuals_1 : array of shape (n,)
    residuals_2 : array of shape (n,)
    n_permutations : int
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    (statistic, p_value)
    """
    residuals_1 = np.asarray(residuals_1, dtype=float)
    residuals_2 = np.asarray(residuals_2, dtype=float)
    n = len(residuals_1)

    Kx = _center_kernel(_rbf_kernel(residuals_1))
    Ky = _center_kernel(_rbf_kernel(residuals_2))

    hsic_observed = float(np.sum(Kx * Ky)) / (n * n)

    rng = np.random.default_rng(seed)
    null_distribution = np.empty(n_permutations)
    for i in range(n_permutations):
        perm = rng.permutation(n)
        Ky_perm = Ky[np.ix_(perm, perm)]
        null_distribution[i] = float(np.sum(Kx * Ky_perm)) / (n * n)

    p_value = float(np.mean(null_distribution >= hsic_observed))
    return hsic_observed, p_value


# ---------------------------------------------------------------------------
# Ablation analysis
# ---------------------------------------------------------------------------

def ablation_analysis(
    predictions_dict: dict[str, np.ndarray],
    icm_fn: Callable[[dict[str, np.ndarray]], float],
    config: AntiSpuriousConfig,
) -> dict[str, float]:
    """Leave-one-family-out ablation: remove each model, recompute ICM.

    Parameters
    ----------
    predictions_dict : dict mapping model name -> predictions array
        Each value has shape (n_samples,).
    icm_fn : callable
        Function that takes a predictions_dict and returns an ICM score.
    config : AntiSpuriousConfig

    Returns
    -------
    dict mapping model name -> change in ICM when that model is removed.
    """
    full_icm = icm_fn(predictions_dict)
    results: dict[str, float] = {}
    for key in predictions_dict:
        reduced = {k: v for k, v in predictions_dict.items() if k != key}
        if len(reduced) < 2:
            results[key] = 0.0
            continue
        reduced_icm = icm_fn(reduced)
        results[key] = full_icm - reduced_icm  # positive = model was helping
    return results


# ---------------------------------------------------------------------------
# FDR correction (Benjamini-Hochberg)
# ---------------------------------------------------------------------------

def fdr_correction(
    p_values: np.ndarray,
    alpha: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """Benjamini-Hochberg FDR correction.

    Parameters
    ----------
    p_values : array of shape (m,)
        Raw p-values.
    alpha : float
        Target FDR level.

    Returns
    -------
    (rejected, corrected_pvalues)
        ``rejected`` is a boolean array; ``corrected_pvalues`` are adjusted
        p-values (capped at 1.0).
    """
    p_values = np.asarray(p_values, dtype=float)
    m = len(p_values)
    sorted_idx = np.argsort(p_values)
    sorted_p = p_values[sorted_idx]
    ranks = np.arange(1, m + 1)

    # Adjusted p-values (step-up)
    adjusted = np.minimum(1.0, sorted_p * m / ranks)
    # Enforce monotonicity from right to left
    for i in range(m - 2, -1, -1):
        adjusted[i] = min(adjusted[i], adjusted[i + 1])

    # Place back in original order
    corrected = np.empty(m)
    corrected[sorted_idx] = adjusted

    rejected = corrected <= alpha
    return rejected, corrected


# ---------------------------------------------------------------------------
# Full anti-spurious report
# ---------------------------------------------------------------------------

def generate_anti_spurious_report(
    predictions_dict: dict[str, np.ndarray],
    labels: np.ndarray,
    features: np.ndarray,
    config: AntiSpuriousConfig,
    icm_fn: Callable[[dict[str, np.ndarray]], float] | None = None,
) -> AntiSpuriousReport:
    """Full anti-spurious pipeline.

    Steps:
    1. Compute observed pairwise distance.
    2. Generate negative controls, compute D_0.
    3. Normalize convergence.
    4. HSIC test on all pairs of model residuals.
    5. Ablation analysis (if ``icm_fn`` provided).
    6. FDR correction on collected p-values.

    Parameters
    ----------
    predictions_dict : dict mapping model name -> predictions (n_samples,)
    labels : array of shape (n_samples,)
    features : array of shape (n_samples, d)
    config : AntiSpuriousConfig
    icm_fn : callable, optional
        ICM scoring function for ablation analysis.

    Returns
    -------
    AntiSpuriousReport
    """
    labels = np.asarray(labels, dtype=float)
    features = np.asarray(features, dtype=float)
    model_names = list(predictions_dict.keys())
    pred_matrix = np.stack([predictions_dict[k] for k in model_names])  # (M, N)

    # 1. Observed distance
    D_observed = _mean_pairwise_distance(pred_matrix)

    # 2. Negative controls
    controls = generate_negative_controls(
        pred_matrix, labels, features,
        method="label_shuffle",
        n_controls=config.n_negative_controls,
    )
    control_distances = np.array([_mean_pairwise_distance(c) for c in controls])
    d0 = estimate_d0(control_distances)

    # 3. Normalize
    c_norm = normalize_convergence(D_observed, d0)

    # 4. HSIC on residual pairs
    residuals = {k: predictions_dict[k] - labels for k in model_names}
    p_values_list: list[float] = []
    for i in range(len(model_names)):
        for j in range(i + 1, len(model_names)):
            _, pv = hsic_test(
                residuals[model_names[i]],
                residuals[model_names[j]],
                n_permutations=config.n_permutations,
            )
            p_values_list.append(pv)

    if p_values_list:
        rejected, corrected = fdr_correction(
            np.array(p_values_list), alpha=config.fdr_level,
        )
        # All pairs must be non-significant (independent residuals) for genuine
        all_independent = not np.any(rejected)
        hsic_pval = float(np.min(corrected))
    else:
        all_independent = True
        hsic_pval = 1.0
        corrected = np.array([])

    # 5. Ablation
    ablation = {}
    if icm_fn is not None:
        ablation = ablation_analysis(predictions_dict, icm_fn, config)

    # 6. Determine genuineness
    is_genuine = (c_norm > 0.5) and all_independent

    return AntiSpuriousReport(
        d0_baseline=d0,
        c_normalized=c_norm,
        hsic_pvalue=hsic_pval,
        ablation_results=ablation,
        is_genuine=is_genuine,
        fdr_corrected=len(corrected) > 0,
    )
