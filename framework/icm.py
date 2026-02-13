"""ICM v1.1 -- Index of Convergence Multi-epistemic.

Core engine for measuring how strongly independent scientific methods
converge on compatible conclusions.  All public functions return values
bounded in [0, 1] unless documented otherwise.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from numpy.typing import NDArray
from scipy import linalg as la
from scipy.special import expit
from scipy.stats import beta as _scipy_beta
from scipy.stats import entropy as _scipy_entropy

from framework.config import ICMConfig
from framework.types import ICMComponents, ICMResult


# ============================================================
# 1. Distance functions
# ============================================================

def hellinger_distance(p: NDArray, q: NDArray) -> float:
    """Hellinger distance between two discrete distributions.

    H^2(P, Q) = (1/2) * sum( (sqrt(p_i) - sqrt(q_i))^2 )
    H   in [0, 1].

    Parameters
    ----------
    p, q : 1-D arrays of non-negative reals that sum to 1.

    Returns
    -------
    float  Hellinger distance (not squared).
    """
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    if np.any(p < 0) or np.any(q < 0):
        raise ValueError("Distribution entries must be non-negative")
    if p.shape != q.shape:
        raise ValueError(f"Shape mismatch: p {p.shape} vs q {q.shape}")
    sq_diff = (np.sqrt(p) - np.sqrt(q)) ** 2
    return float(np.sqrt(0.5 * sq_diff.sum()))


def wasserstein2_distance(
    mu1: NDArray,
    sigma1: NDArray,
    mu2: NDArray,
    sigma2: NDArray,
) -> float:
    """Closed-form 2-Wasserstein distance between two Gaussians.

    W_2^2 = ||mu1 - mu2||^2
            + Tr(Sigma1 + Sigma2
                 - 2 (Sigma2^{1/2} Sigma1 Sigma2^{1/2})^{1/2})

    Parameters
    ----------
    mu1, mu2   : 1-D mean vectors (d,).
    sigma1, sigma2 : 2-D covariance matrices (d, d).

    Returns
    -------
    float  W_2 distance (the square root of W_2^2).
    """
    mu1 = np.asarray(mu1, dtype=np.float64)
    mu2 = np.asarray(mu2, dtype=np.float64)
    sigma1 = np.atleast_2d(np.asarray(sigma1, dtype=np.float64))
    sigma2 = np.atleast_2d(np.asarray(sigma2, dtype=np.float64))

    mean_sq = float(np.sum((mu1 - mu2) ** 2))

    # Sigma2^{1/2}
    sqrt_sigma2 = _matrix_sqrt_psd(sigma2)

    # M = Sigma2^{1/2} @ Sigma1 @ Sigma2^{1/2}
    M = sqrt_sigma2 @ sigma1 @ sqrt_sigma2
    sqrt_M = _matrix_sqrt_psd(M)

    trace_term = float(
        np.trace(sigma1) + np.trace(sigma2) - 2.0 * np.trace(sqrt_M)
    )
    # Clamp numerical noise
    w2_sq = max(mean_sq + trace_term, 0.0)
    return float(np.sqrt(w2_sq))


def wasserstein2_empirical(X: NDArray, Y: NDArray) -> float:
    """2-Wasserstein distance between two empirical distributions via POT.

    Falls back to a simple sorted-quantile approximation when both inputs
    are 1-D, which avoids the POT dependency for the common univariate case.

    Parameters
    ----------
    X, Y : 2-D arrays (n, d) of samples.

    Returns
    -------
    float  Approximate W_2 distance.
    """
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)

    # 1-D fast path: sorted-quantile coupling
    if X.ndim == 1 or (X.ndim == 2 and X.shape[1] == 1):
        x = X.ravel()
        y = Y.ravel()
        xs = np.sort(x)
        ys = np.sort(y)
        # Interpolate to equal length
        n = max(len(xs), len(ys))
        xs_interp = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(xs)), xs)
        ys_interp = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(ys)), ys)
        return float(np.sqrt(np.mean((xs_interp - ys_interp) ** 2)))

    # Multi-dimensional: use POT
    try:
        import ot  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ImportError(
            "POT (Python Optimal Transport) is required for multi-dimensional "
            "empirical Wasserstein distance.  Install with: pip install POT"
        ) from exc

    n, m = len(X), len(Y)
    a = np.ones(n) / n
    b = np.ones(m) / m
    M = ot.dist(X, Y, metric="sqeuclidean")
    w2_sq = float(ot.emd2(a, b, M))
    return float(np.sqrt(max(w2_sq, 0.0)))


def mmd_distance(
    X: NDArray,
    Y: NDArray,
    bandwidth: float = 1.0,
) -> float:
    """Maximum Mean Discrepancy with RBF (Gaussian) kernel.

    MMD^2 = E[k(X,X')] + E[k(Y,Y')] - 2 E[k(X,Y)]

    Parameters
    ----------
    X, Y      : 2-D arrays (n, d) and (m, d).
    bandwidth : RBF kernel bandwidth (sigma).

    Returns
    -------
    float  MMD (square root of MMD^2, clamped >= 0).
    """
    X = np.atleast_2d(np.asarray(X, dtype=np.float64))
    Y = np.atleast_2d(np.asarray(Y, dtype=np.float64))
    gamma = 1.0 / (2.0 * bandwidth ** 2)

    Kxx = _rbf_kernel(X, X, gamma)
    Kyy = _rbf_kernel(Y, Y, gamma)
    Kxy = _rbf_kernel(X, Y, gamma)

    mmd_sq = float(Kxx.mean() + Kyy.mean() - 2.0 * Kxy.mean())
    return float(np.sqrt(max(mmd_sq, 0.0)))


def frechet_variance(embeddings: NDArray) -> float:
    """Frechet variance in an embedding space.

    V_F = (1/m) * sum_i ||phi_i - phi_bar||^2

    Parameters
    ----------
    embeddings : 2-D array (m, d) of embedding vectors.

    Returns
    -------
    float  Frechet variance (>= 0).
    """
    embeddings = np.atleast_2d(np.asarray(embeddings, dtype=np.float64))
    centroid = embeddings.mean(axis=0)
    diffs = embeddings - centroid
    return float(np.mean(np.sum(diffs ** 2, axis=1)))


# ============================================================
# 2. ICM components
# ============================================================

def compute_agreement(
    predictions: Sequence[NDArray],
    distance_fn: str = "hellinger",
    C_A: float | None = None,
    config: ICMConfig | None = None,
) -> float:
    """Distributional agreement A_i.

    Compute pairwise distances among K model predictions, take the mean,
    normalize by C_A, and map to [0, 1] via  A = 1 - mean_d / C_A.

    Parameters
    ----------
    predictions : list of K arrays, each representing one model's output
        distribution or prediction vector.
    distance_fn : one of {"hellinger", "wasserstein", "mmd"}.
    C_A : normalization constant.  If None, read from *config* or default.
    config : ICMConfig (used for default C_A and bandwidth).

    Returns
    -------
    float  A in [0, 1].  Higher means stronger agreement.
    """
    if config is None:
        config = ICMConfig()

    K = len(predictions)
    if K < 2:
        return 1.0  # Trivial agreement with a single model

    if C_A is None:
        C_A = {
            "hellinger": config.C_A_hellinger,
            "wasserstein": config.C_A_wasserstein,
            "mmd": config.C_A_mmd,
        }.get(distance_fn, 1.0)

    dist_fn = _resolve_distance_fn(distance_fn, config)

    # Pairwise distances
    dists: list[float] = []
    for i in range(K):
        for j in range(i + 1, K):
            dists.append(dist_fn(predictions[i], predictions[j]))

    mean_d = float(np.mean(dists))
    A = 1.0 - min(mean_d / C_A, 1.0)
    return float(np.clip(A, 0.0, 1.0))


def compute_direction(signs_or_gradients: NDArray) -> float:
    """Direction/sign agreement D_i.

    D = 1 - H(sign_distribution) / log(K)

    where H is Shannon entropy (base e) of the empirical sign distribution
    and K is the number of distinct sign categories.

    Parameters
    ----------
    signs_or_gradients : 1-D array of signs (+1, -1, or 0) or gradient
        values from which signs are extracted.

    Returns
    -------
    float  D in [0, 1].  1 = perfect directional agreement.
    """
    arr = np.asarray(signs_or_gradients, dtype=np.float64)
    signs = np.sign(arr).astype(int)

    K = len(signs)
    if K <= 1:
        return 1.0

    # Empirical distribution over unique sign values
    unique, counts = np.unique(signs, return_counts=True)
    if len(unique) <= 1:
        return 1.0  # All agree

    probs = counts / counts.sum()
    H = float(_scipy_entropy(probs))  # natural log
    H_max = float(np.log(K))

    if H_max == 0.0:
        return 1.0

    D = 1.0 - H / H_max
    return float(np.clip(D, 0.0, 1.0))


def compute_uncertainty_overlap(intervals: Sequence[tuple[float, float]]) -> float:
    """Uncertainty overlap U_i via pairwise interval overlap.

    For each pair (a, b) of intervals:
        IO(a,b) = len(intersection) / len(union)

    U = mean of pairwise IO values.

    Parameters
    ----------
    intervals : list of (lo, hi) tuples representing credible/confidence
        intervals from each model.

    Returns
    -------
    float  U in [0, 1].  1 = perfect overlap.
    """
    K = len(intervals)
    if K < 2:
        return 1.0

    overlaps: list[float] = []
    for i in range(K):
        for j in range(i + 1, K):
            lo_i, hi_i = intervals[i]
            lo_j, hi_j = intervals[j]
            inter_lo = max(lo_i, lo_j)
            inter_hi = min(hi_i, hi_j)
            intersection = max(inter_hi - inter_lo, 0.0)
            union_lo = min(lo_i, lo_j)
            union_hi = max(hi_i, hi_j)
            union = max(union_hi - union_lo, 1e-12)  # avoid div-by-zero
            overlaps.append(intersection / union)

    return float(np.clip(np.mean(overlaps), 0.0, 1.0))


def compute_invariance(
    pre_scores: NDArray,
    post_scores: NDArray,
) -> float:
    """Invariance/stability score C_i.

    C = 1 - ||pre - post|| / (||pre|| + eps)

    Measures how stable ICM-relevant predictions remain after an
    intervention or perturbation.

    Parameters
    ----------
    pre_scores, post_scores : 1-D arrays of scores (e.g. per-model
        predictions) before and after intervention.

    Returns
    -------
    float  C in [0, 1].  1 = perfectly stable.
    """
    pre = np.asarray(pre_scores, dtype=np.float64)
    post = np.asarray(post_scores, dtype=np.float64)
    eps = 1e-12
    diff_norm = float(la.norm(pre - post))
    pre_norm = float(la.norm(pre)) + eps
    C = 1.0 - min(diff_norm / pre_norm, 1.0)
    return float(np.clip(C, 0.0, 1.0))


def compute_dependency_penalty(
    residuals: NDArray | None = None,
    features: Sequence[set[str]] | None = None,
    gradients: NDArray | None = None,
    config: ICMConfig | None = None,
) -> float:
    """Dependency penalty Pi_i.

    Pi = gamma_rho * rho_corr + gamma_J * J_overlap + gamma_grad * g_sim

    Sub-components:
    - rho_corr: mean |off-diagonal| of Ledoit-Wolf shrunk residual
      correlation matrix.
    - J_overlap: mean pairwise Jaccard similarity of feature/provenance sets.
    - g_sim: mean pairwise cosine similarity of gradient vectors.

    Parameters
    ----------
    residuals : (K, n) array of model residuals.
    features  : list of K sets of feature names.
    gradients : (K, d) array of gradient vectors.
    config    : ICMConfig for sub-weights.

    Returns
    -------
    float  Pi in [0, 1].
    """
    if config is None:
        config = ICMConfig()

    terms: list[float] = []
    weights: list[float] = []

    # --- Residual correlation (Ledoit-Wolf) ---
    if residuals is not None:
        residuals = np.asarray(residuals, dtype=np.float64)
        if residuals.ndim == 2 and residuals.shape[0] >= 2:
            rho = _ledoit_wolf_corr(residuals)
            K = rho.shape[0]
            mask = ~np.eye(K, dtype=bool)
            rho_corr = float(np.mean(np.abs(rho[mask])))
        else:
            rho_corr = 0.0
        terms.append(rho_corr)
        weights.append(config.gamma_rho)

    # --- Feature / provenance overlap (Jaccard) ---
    if features is not None and len(features) >= 2:
        jaccards: list[float] = []
        feat_list = list(features)
        for i in range(len(feat_list)):
            for j in range(i + 1, len(feat_list)):
                inter = len(feat_list[i] & feat_list[j])
                union = len(feat_list[i] | feat_list[j])
                jaccards.append(inter / max(union, 1))
        J_overlap = float(np.mean(jaccards))
        terms.append(J_overlap)
        weights.append(config.gamma_J)

    # --- Gradient similarity (cosine) ---
    if gradients is not None:
        gradients = np.atleast_2d(np.asarray(gradients, dtype=np.float64))
        if gradients.shape[0] >= 2:
            cosines: list[float] = []
            for i in range(gradients.shape[0]):
                for j in range(i + 1, gradients.shape[0]):
                    g_i = gradients[i]
                    g_j = gradients[j]
                    denom = la.norm(g_i) * la.norm(g_j)
                    if denom < 1e-12:
                        cosines.append(0.0)
                    else:
                        cosines.append(float(np.dot(g_i, g_j) / denom))
            g_sim = float(np.mean(cosines))
            # Cosine sim is in [-1, 1]; map to [0, 1]
            g_sim = (g_sim + 1.0) / 2.0
        else:
            g_sim = 0.0
        terms.append(g_sim)
        weights.append(config.gamma_grad)

    if not terms:
        return 0.0

    weights_arr = np.array(weights)
    weights_arr = weights_arr / weights_arr.sum()  # renormalize
    Pi = float(np.dot(weights_arr, terms))
    return float(np.clip(Pi, 0.0, 1.0))


# ============================================================
# 3. ICM aggregation
# ============================================================

def compute_icm(
    components: ICMComponents,
    config: ICMConfig | None = None,
) -> ICMResult:
    """Logistic ICM aggregation.

    ICM = sigma(scale * (z_raw - shift))

    where z_raw = w_A*A + w_D*D + w_U*U + w_C*C - lambda*Pi.

    When scale=1 and shift=0 (the defaults), this reduces to the legacy
    formula ``sigma(z_raw)``.  Use ``ICMConfig.wide_range_preset()`` for
    scale=10, shift=0.5 which spreads scores across the full [0, 1]
    range instead of the narrow ~[0.46, 0.70] band.

    Parameters
    ----------
    components : ICMComponents with fields A, D, U, C, Pi.
    config     : ICMConfig with weights, lambda, scale, and shift.

    Returns
    -------
    ICMResult
    """
    if config is None:
        config = ICMConfig()

    z_raw = (
        config.w_A * components.A
        + config.w_D * components.D
        + config.w_U * components.U
        + config.w_C * components.C
        - config.lam * components.Pi
    )
    # Apply scale and shift for better sigmoid utilization.
    # Legacy defaults (scale=1.0, shift=0.0) give z == z_raw.
    z = config.logistic_scale * (z_raw - config.logistic_shift)
    icm_score = float(expit(z))

    weights = {
        "w_A": config.w_A,
        "w_D": config.w_D,
        "w_U": config.w_U,
        "w_C": config.w_C,
        "lam": config.lam,
    }

    return ICMResult(
        icm_score=icm_score,
        components=components,
        aggregation_method="logistic",
        weights=weights,
    )


def compute_icm_geometric(
    components: ICMComponents,
    config: ICMConfig | None = None,
) -> ICMResult:
    """Weighted geometric-mean ICM aggregation.

    ICM_geo = A^w_A * D^w_D * U^w_U * C^w_C * (1 - Pi)^lam

    Each factor is clamped to [eps, 1] to avoid log(0).

    Parameters
    ----------
    components : ICMComponents with fields A, D, U, C, Pi.
    config     : ICMConfig with weights and lambda.

    Returns
    -------
    ICMResult
    """
    if config is None:
        config = ICMConfig()

    eps = 1e-12
    factors = [
        (max(components.A, eps), config.w_A),
        (max(components.D, eps), config.w_D),
        (max(components.U, eps), config.w_U),
        (max(components.C, eps), config.w_C),
        (max(1.0 - components.Pi, eps), config.lam),
    ]

    # Normalize exponents to sum to 1
    total_w = sum(w for _, w in factors)
    log_icm = sum(w / total_w * np.log(v) for v, w in factors)
    icm_score = float(np.exp(log_icm))

    weights = {
        "w_A": config.w_A,
        "w_D": config.w_D,
        "w_U": config.w_U,
        "w_C": config.w_C,
        "lam": config.lam,
    }

    return ICMResult(
        icm_score=float(np.clip(icm_score, 0.0, 1.0)),
        components=components,
        aggregation_method="geometric",
        weights=weights,
    )


def compute_icm_calibrated(
    components: ICMComponents,
    config: ICMConfig | None = None,
) -> ICMResult:
    """Beta-calibrated ICM aggregation.

    Maps the weighted linear combination through a Beta CDF instead of
    a logistic sigmoid.  The Beta CDF is monotonic, bounded in [0, 1],
    smooth (and therefore Lipschitz-continuous on [0, 1]), and symmetric
    when shape_a == shape_b.

    Compared to the logistic sigmoid, which compresses typical scores
    into a narrow band around 0.5, the Beta CDF (with appropriate shape
    parameters) spreads the output across the full [0, 1] range, giving
    the downstream ACT/DEFER/AUDIT decision gate much better
    discrimination power.

    Formula
    -------
    z = w_A*A + w_D*D + w_U*U + w_C*C - lambda*Pi
    z_norm = clip(z / z_max, 0, 1)          # normalize to [0, 1]
    ICM = BetaCDF(z_norm; shape_a, shape_b)

    where z_max = w_A + w_D + w_U + w_C  (maximum possible z when all
    components are 1 and Pi is 0).

    Parameters
    ----------
    components : ICMComponents with fields A, D, U, C, Pi.
    config     : ICMConfig with weights, lambda, beta_shape_a, beta_shape_b.

    Returns
    -------
    ICMResult
    """
    if config is None:
        config = ICMConfig()

    z = (
        config.w_A * components.A
        + config.w_D * components.D
        + config.w_U * components.U
        + config.w_C * components.C
        - config.lam * components.Pi
    )

    # Normalize z to [0, 1].  The theoretical range of z is
    # [-lam, w_A + w_D + w_U + w_C] (all components in [0,1], Pi in [0,1]).
    z_min = -config.lam
    z_max = config.w_A + config.w_D + config.w_U + config.w_C
    denom = z_max - z_min
    if denom < 1e-12:
        z_norm = 0.5
    else:
        z_norm = (z - z_min) / denom
    z_norm = float(np.clip(z_norm, 0.0, 1.0))

    # Apply Beta CDF
    icm_score = float(_scipy_beta.cdf(z_norm, config.beta_shape_a, config.beta_shape_b))

    weights = {
        "w_A": config.w_A,
        "w_D": config.w_D,
        "w_U": config.w_U,
        "w_C": config.w_C,
        "lam": config.lam,
        "beta_shape_a": config.beta_shape_a,
        "beta_shape_b": config.beta_shape_b,
    }

    return ICMResult(
        icm_score=float(np.clip(icm_score, 0.0, 1.0)),
        components=components,
        aggregation_method="calibrated",
        weights=weights,
    )


def compute_icm_adaptive(
    components: ICMComponents,
    config: ICMConfig | None = None,
) -> ICMResult:
    """Adaptive / percentile-based ICM aggregation.

    Uses a calibration set of historical raw linear scores to determine
    the empirical CDF, then maps the current score to its percentile
    rank within that distribution.  This guarantees output uniformity
    over the calibration distribution, maximizing discrimination.

    When no calibration set is provided in
    ``config.adaptive_calibration_scores``, falls back to a default
    reference distribution generated from a uniform grid over the
    component space.

    Formal properties
    -----------------
    - **Monotonic**: the percentile rank is a monotone function of the
      raw score.
    - **Bounded in [0, 1]**: by definition.
    - **Symmetric**: the mapping is symmetric if the calibration
      distribution is symmetric.
    - **Lipschitz**: the empirical CDF is 1/N-Lipschitz (piecewise
      constant), and interpolation yields a Lipschitz-continuous
      approximation.

    Parameters
    ----------
    components : ICMComponents with fields A, D, U, C, Pi.
    config     : ICMConfig with weights, lambda, and optionally
                 adaptive_calibration_scores.

    Returns
    -------
    ICMResult
    """
    if config is None:
        config = ICMConfig()

    z = (
        config.w_A * components.A
        + config.w_D * components.D
        + config.w_U * components.U
        + config.w_C * components.C
        - config.lam * components.Pi
    )

    cal_scores = config.adaptive_calibration_scores

    if cal_scores is None or len(cal_scores) < 2:
        # Generate a default calibration set by sampling the component
        # space on a regular grid.  This gives a stable reference
        # distribution that covers the full range.
        rng = np.random.default_rng(0)
        n_samples = 500
        default_scores: list[float] = []
        for _ in range(n_samples):
            a = rng.uniform(0, 1)
            d = rng.uniform(0, 1)
            u = rng.uniform(0, 1)
            c = rng.uniform(0, 1)
            pi = rng.uniform(0, 1)
            s = (
                config.w_A * a
                + config.w_D * d
                + config.w_U * u
                + config.w_C * c
                - config.lam * pi
            )
            default_scores.append(s)
        cal_scores = default_scores

    sorted_cal = np.sort(cal_scores)
    n = len(sorted_cal)

    # Compute percentile rank via linear interpolation for smoothness.
    # searchsorted gives the insertion index; dividing by n gives the
    # empirical CDF value.
    idx = np.searchsorted(sorted_cal, z, side="right")
    # Linear interpolation between neighboring calibration points
    if idx <= 0:
        icm_score = 0.0
    elif idx >= n:
        icm_score = 1.0
    else:
        # Interpolate between sorted_cal[idx-1] and sorted_cal[idx]
        lo = sorted_cal[idx - 1]
        hi = sorted_cal[idx]
        if hi - lo < 1e-12:
            icm_score = idx / n
        else:
            frac = (z - lo) / (hi - lo)
            icm_score = (idx - 1 + frac) / n
    icm_score = float(np.clip(icm_score, 0.0, 1.0))

    weights = {
        "w_A": config.w_A,
        "w_D": config.w_D,
        "w_U": config.w_U,
        "w_C": config.w_C,
        "lam": config.lam,
    }

    return ICMResult(
        icm_score=icm_score,
        components=components,
        aggregation_method="adaptive",
        weights=weights,
    )


def _dispatch_aggregation(
    components: ICMComponents,
    config: ICMConfig,
) -> ICMResult:
    """Dispatch to the appropriate aggregation function based on config.

    Parameters
    ----------
    components : ICMComponents
    config     : ICMConfig (uses config.aggregation to select method)

    Returns
    -------
    ICMResult
    """
    method = config.aggregation
    if method == "logistic":
        return compute_icm(components, config)
    elif method == "geometric":
        return compute_icm_geometric(components, config)
    elif method == "calibrated":
        return compute_icm_calibrated(components, config)
    elif method == "adaptive":
        return compute_icm_adaptive(components, config)
    else:
        raise ValueError(
            f"Unknown aggregation method: {method!r}.  "
            f"Choose from 'logistic', 'geometric', 'calibrated', 'adaptive'."
        )


# ============================================================
# 4. Convenience / high-level functions
# ============================================================

def compute_pi_from_predictions(
    predictions: dict[str, NDArray],
    y_true: NDArray,
    config: ICMConfig | None = None,
) -> float:
    """Compute Pi (dependency penalty) from model predictions and ground truth.

    Extracts residuals from each model's predictions by subtracting the
    ground truth, then computes the dependency penalty using Ledoit-Wolf
    shrinkage estimation of inter-model residual correlation.

    For classification (probability vectors), residuals are computed as
    ``predicted_proba - one_hot(y_true)`` when predictions are 2-D, or
    ``predicted - y_true`` for 1-D predictions (regression or scalar
    classification outputs).

    Parameters
    ----------
    predictions : dict mapping model names to prediction arrays.
        Each value can be:
        - 1-D array of length n (regression or scalar predictions)
        - 2-D array of shape (n, C) (classification probabilities)
    y_true : 1-D array of ground truth values.
        For regression: continuous targets.
        For classification: integer class labels (used to build one-hot).
    config : ICMConfig for dependency penalty sub-weights.

    Returns
    -------
    float  Pi in [0, 1].
    """
    if config is None:
        config = ICMConfig()

    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    n = len(y_true)

    model_names = list(predictions.keys())
    if len(model_names) < 2:
        return 0.0

    residual_rows: list[NDArray] = []
    for name in model_names:
        pred = np.asarray(predictions[name], dtype=np.float64)

        if pred.ndim == 2 and pred.shape[1] > 1:
            # Classification probabilities: (n, C) -> flatten residuals
            n_classes = pred.shape[1]
            # Build one-hot from y_true
            y_int = y_true.astype(int)
            y_int = np.clip(y_int, 0, n_classes - 1)
            one_hot = np.zeros((n, n_classes), dtype=np.float64)
            one_hot[np.arange(n), y_int] = 1.0
            resid = (pred[:n] - one_hot).ravel()
        else:
            # Regression or 1-D predictions
            pred_flat = pred.ravel()[:n]
            resid = pred_flat - y_true[:len(pred_flat)]

        residual_rows.append(resid)

    # Truncate to common length
    min_len = min(len(r) for r in residual_rows)
    residuals = np.stack([r[:min_len] for r in residual_rows])

    return compute_dependency_penalty(residuals=residuals, config=config)


def compute_icm_from_predictions(
    predictions_dict: dict[str, NDArray],
    config: ICMConfig | None = None,
    distance_fn: str = "hellinger",
    intervals: Sequence[tuple[float, float]] | None = None,
    signs: NDArray | None = None,
    pre_scores: NDArray | None = None,
    post_scores: NDArray | None = None,
    residuals: NDArray | None = None,
    features: Sequence[set[str]] | None = None,
    gradients: NDArray | None = None,
    y_true: NDArray | None = None,
) -> ICMResult:
    """Compute full ICM from a dict of {model_name: predictions_array}.

    This is the main entry point for users who just want an ICM score
    from their multi-model outputs.

    Parameters
    ----------
    predictions_dict : {model_name: 1-D prediction array}.
    config : ICMConfig.
    distance_fn : distance metric for agreement.
    intervals : optional per-model (lo, hi) intervals for U.
    signs : optional 1-D array of gradient signs for D.
    pre_scores, post_scores : optional for invariance C.
    residuals, features, gradients : optional for dependency penalty Pi.
    y_true : optional ground truth array.  When provided *and* no
        explicit residuals are given, Pi is automatically computed from
        model predictions vs. ground truth via
        ``compute_pi_from_predictions``.

    Returns
    -------
    ICMResult
    """
    if config is None:
        config = ICMConfig()

    model_names = list(predictions_dict.keys())
    preds_list = [predictions_dict[m] for m in model_names]
    K = len(preds_list)

    # --- A: agreement ---
    A = compute_agreement(preds_list, distance_fn=distance_fn, config=config)

    # --- D: direction ---
    if signs is not None:
        D = compute_direction(signs)
    else:
        # Derive signs from mean predictions
        means = np.array([np.mean(p) for p in preds_list])
        D = compute_direction(means)

    # --- U: uncertainty overlap ---
    if intervals is not None:
        U = compute_uncertainty_overlap(intervals)
    else:
        # Derive rough intervals from prediction quantiles
        derived_intervals: list[tuple[float, float]] = []
        for p in preds_list:
            p_arr = np.asarray(p).ravel()
            if len(p_arr) >= 4:
                lo = float(np.percentile(p_arr, 10))
                hi = float(np.percentile(p_arr, 90))
            else:
                lo = float(p_arr.min())
                hi = float(p_arr.max())
            derived_intervals.append((lo, hi))
        U = compute_uncertainty_overlap(derived_intervals)

    # --- C: invariance ---
    if pre_scores is not None and post_scores is not None:
        C_val = compute_invariance(pre_scores, post_scores)
    else:
        C_val = 1.0  # Assume stable if no perturbation provided

    # --- Pi: dependency penalty ---
    # Auto-compute Pi from y_true when no explicit residuals are provided.
    if residuals is None and y_true is not None and K >= 2:
        Pi = compute_pi_from_predictions(
            predictions_dict, y_true, config=config,
        )
        # Still incorporate features/gradients if provided
        if features is not None or gradients is not None:
            Pi_extra = compute_dependency_penalty(
                features=features, gradients=gradients, config=config,
            )
            # Blend: take the max to be conservative (higher penalty)
            Pi = max(Pi, Pi_extra)
    else:
        Pi = compute_dependency_penalty(
            residuals=residuals,
            features=features,
            gradients=gradients,
            config=config,
        )

    components = ICMComponents(A=A, D=D, U=U, C=C_val, Pi=Pi)
    result = _dispatch_aggregation(components, config)
    result.n_models = K
    return result


def compute_icm_timeseries(
    predictions_dict_over_time: Sequence[dict[str, NDArray]],
    config: ICMConfig | None = None,
    window_size: int = 1,
    distance_fn: str = "hellinger",
) -> list[ICMResult]:
    """Rolling-window ICM over a time series of multi-model predictions.

    Parameters
    ----------
    predictions_dict_over_time : list of T dicts, each
        {model_name: predictions_array} for one time step.
    config : ICMConfig.
    window_size : number of time steps per window.  If 1, compute ICM
        at each step independently.
    distance_fn : distance metric for agreement.

    Returns
    -------
    list[ICMResult]  One result per window position.
    """
    if config is None:
        config = ICMConfig()

    T = len(predictions_dict_over_time)
    if window_size < 1:
        window_size = 1

    results: list[ICMResult] = []
    for start in range(T - window_size + 1):
        window = predictions_dict_over_time[start : start + window_size]

        # Merge predictions across the window
        merged: dict[str, list[NDArray]] = {}
        for step in window:
            for name, pred in step.items():
                merged.setdefault(name, []).append(np.asarray(pred))

        merged_dict: dict[str, NDArray] = {
            name: np.concatenate(arrs) for name, arrs in merged.items()
        }
        # compute_icm_from_predictions uses _dispatch_aggregation
        # internally, so the aggregation mode from config is respected.
        result = compute_icm_from_predictions(
            merged_dict, config=config, distance_fn=distance_fn
        )
        results.append(result)

    return results


# ============================================================
# Private helpers
# ============================================================

def _matrix_sqrt_psd(A: NDArray) -> NDArray:
    """Compute matrix square root of a positive semi-definite matrix."""
    eigvals, eigvecs = la.eigh(A)
    eigvals = np.maximum(eigvals, 0.0)
    return (eigvecs * np.sqrt(eigvals)) @ eigvecs.T


def _rbf_kernel(X: NDArray, Y: NDArray, gamma: float) -> NDArray:
    """RBF (Gaussian) kernel matrix K(X, Y)."""
    # ||x - y||^2 = ||x||^2 + ||y||^2 - 2 x.y
    X_sq = np.sum(X ** 2, axis=1, keepdims=True)
    Y_sq = np.sum(Y ** 2, axis=1, keepdims=True)
    sq_dists = X_sq + Y_sq.T - 2.0 * X @ Y.T
    return np.exp(-gamma * sq_dists)


def _ledoit_wolf_corr(residuals: NDArray) -> NDArray:
    """Ledoit-Wolf shrunk correlation matrix from (K, n) residuals."""
    # Standardize each row
    K, n = residuals.shape
    mu = residuals.mean(axis=1, keepdims=True)
    std = residuals.std(axis=1, keepdims=True)
    std = np.where(std < 1e-12, 1.0, std)
    Z = (residuals - mu) / std

    # Sample correlation
    S = (Z @ Z.T) / n

    # Ledoit-Wolf shrinkage toward identity
    trace_S2 = float(np.trace(S @ S))
    trace_S_sq = float(np.trace(S) ** 2)

    # Optimal shrinkage intensity (Oracle Approximation Shrinkage)
    numerator = (1.0 - 2.0 / K) * trace_S2 + trace_S_sq
    denominator = (n + 1.0 - 2.0 / K) * (trace_S2 - trace_S_sq / K)

    if abs(denominator) < 1e-12:
        alpha = 1.0
    else:
        alpha = float(np.clip(numerator / denominator, 0.0, 1.0))

    shrunk = (1.0 - alpha) * S + alpha * np.eye(K)
    return shrunk


def _resolve_distance_fn(name: str, config: ICMConfig):
    """Return a callable (p, q) -> float for the named distance."""
    if name == "hellinger":
        return hellinger_distance
    elif name == "wasserstein":
        def _w2(p, q):
            return wasserstein2_empirical(p, q)
        return _w2
    elif name == "mmd":
        def _mmd(p, q):
            return mmd_distance(p, q, bandwidth=config.mmd_bandwidth)
        return _mmd
    else:
        raise ValueError(f"Unknown distance function: {name!r}")
