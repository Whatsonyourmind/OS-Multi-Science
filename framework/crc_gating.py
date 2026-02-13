"""Conformal Risk Control (CRC) gating for OS Multi-Science.

Provides calibrated epistemic-risk bounds via isotonic regression
and split-conformal calibration, plus decision gating logic.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from sklearn.isotonic import IsotonicRegression

from framework.config import CRCConfig
from framework.types import DecisionAction


# ---------------------------------------------------------------------------
# Isotonic fitting
# ---------------------------------------------------------------------------

def fit_isotonic(
    C_values: np.ndarray,
    L_values: np.ndarray,
) -> IsotonicRegression:
    """Fit monotone-decreasing mapping g: ICM score -> expected loss.

    Parameters
    ----------
    C_values : array of shape (n,)
        ICM convergence scores in [0, 1].
    L_values : array of shape (n,)
        Observed losses (lower is better).

    Returns
    -------
    IsotonicRegression
        Fitted model with ``increasing=False``.
    """
    C_values = np.asarray(C_values, dtype=float)
    L_values = np.asarray(L_values, dtype=float)
    model = IsotonicRegression(increasing=False, out_of_bounds="clip")
    model.fit(C_values, L_values)
    return model


# ---------------------------------------------------------------------------
# Conformal calibration
# ---------------------------------------------------------------------------

def conformalize(
    g_fitted: IsotonicRegression,
    C_cal: np.ndarray,
    L_cal: np.ndarray,
    alpha: float = 0.10,
) -> Callable[[float | np.ndarray], float | np.ndarray]:
    """Split-conformal calibration of the isotonic risk curve.

    Computes residuals r_i = L_i - g(C_i) on the calibration set and
    returns g_alpha(c) = g(c) + q, where q is the ceil((1-alpha)(n+1)/n)-th
    empirical quantile of the residuals.

    Parameters
    ----------
    g_fitted : IsotonicRegression
        Fitted isotonic regression model.
    C_cal : array of shape (n_cal,)
        Calibration ICM scores.
    L_cal : array of shape (n_cal,)
        Calibration losses.
    alpha : float
        Miscoverage level; the bound holds with probability >= 1 - alpha.

    Returns
    -------
    g_alpha : callable
        Conformalized risk function mapping ICM score(s) to upper risk bound(s).
    """
    C_cal = np.asarray(C_cal, dtype=float)
    L_cal = np.asarray(L_cal, dtype=float)
    residuals = L_cal - g_fitted.predict(C_cal)
    n = len(residuals)
    quantile_level = np.ceil((1 - alpha) * (n + 1)) / n
    quantile_level = min(quantile_level, 1.0)
    q = float(np.quantile(residuals, quantile_level))

    def g_alpha(c: float | np.ndarray) -> float | np.ndarray:
        c_arr = np.asarray(c, dtype=float)
        result = g_fitted.predict(c_arr.ravel()) + q
        if c_arr.ndim == 0:
            return float(result[0])
        return result

    return g_alpha


# ---------------------------------------------------------------------------
# Epistemic risk
# ---------------------------------------------------------------------------

def compute_re(
    icm_score: float,
    g_alpha: Callable[[float], float],
) -> float:
    """Compute epistemic risk Re = g_alpha(ICM).

    Parameters
    ----------
    icm_score : float
        ICM convergence score.
    g_alpha : callable
        Conformalized risk function.

    Returns
    -------
    float
        Upper-bounded epistemic risk.
    """
    return float(g_alpha(icm_score))


# ---------------------------------------------------------------------------
# Decision gate
# ---------------------------------------------------------------------------

def decision_gate(
    icm_score: float,
    re_score: float,
    config: CRCConfig,
) -> DecisionAction:
    """Three-way decision gate based on ICM and epistemic risk.

    - ACT   if ICM >= tau_hi
    - DEFER if tau_lo <= ICM < tau_hi
    - AUDIT if ICM < tau_lo

    Parameters
    ----------
    icm_score : float
        ICM convergence score in [0, 1].
    re_score : float
        Epistemic risk score (unused in threshold logic but available
        for downstream logging / override rules).
    config : CRCConfig
        Contains ``tau_hi`` and ``tau_lo``.

    Returns
    -------
    DecisionAction
    """
    if icm_score >= config.tau_hi:
        return DecisionAction.ACT
    elif icm_score >= config.tau_lo:
        return DecisionAction.DEFER
    else:
        return DecisionAction.AUDIT


# ---------------------------------------------------------------------------
# Risk-coverage curve
# ---------------------------------------------------------------------------

def risk_coverage_curve(
    C_values: np.ndarray,
    L_values: np.ndarray,
    thresholds: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Sweep tau and compute coverage and average risk.

    Parameters
    ----------
    C_values : array of shape (n,)
        ICM scores.
    L_values : array of shape (n,)
        Losses.
    thresholds : array, optional
        Threshold values to sweep.  Defaults to 50 evenly spaced values
        in [0, 1].

    Returns
    -------
    dict with keys 'thresholds', 'coverage', 'avg_risk'
        Each value is an array of the same length as *thresholds*.
    """
    C_values = np.asarray(C_values, dtype=float)
    L_values = np.asarray(L_values, dtype=float)
    if thresholds is None:
        thresholds = np.linspace(0.0, 1.0, 50)
    thresholds = np.asarray(thresholds, dtype=float)

    n = len(C_values)
    coverage = np.empty(len(thresholds))
    avg_risk = np.empty(len(thresholds))

    for i, tau in enumerate(thresholds):
        mask = C_values >= tau
        coverage[i] = mask.sum() / n if n > 0 else 0.0
        avg_risk[i] = L_values[mask].mean() if mask.any() else np.nan

    return {
        "thresholds": thresholds,
        "coverage": coverage,
        "avg_risk": avg_risk,
    }


# ---------------------------------------------------------------------------
# Threshold calibration
# ---------------------------------------------------------------------------

def calibrate_thresholds(
    C_cal: np.ndarray,
    L_cal: np.ndarray,
    target_coverage: float = 0.80,
    alpha: float = 0.10,
    seed: int = 42,
) -> tuple[float, float]:
    """Find tau_hi and tau_lo that achieve *target_coverage* at level alpha.

    Strategy:
    - Fit isotonic + conformalize on the calibration data.
    - tau_hi is the smallest ICM value whose conformalized risk <= alpha.
    - tau_lo is set to achieve target_coverage: fraction of data with
      ICM >= tau_lo equals target_coverage.

    Parameters
    ----------
    C_cal : array of shape (n,)
        Calibration ICM scores.
    L_cal : array of shape (n,)
        Calibration losses.
    target_coverage : float
        Desired coverage (fraction of instances with ICM >= tau_lo).
    alpha : float
        Risk level.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    (tau_hi, tau_lo)
    """
    C_cal = np.asarray(C_cal, dtype=float)
    L_cal = np.asarray(L_cal, dtype=float)

    # Split calibration data in half for fit vs. conformal
    n = len(C_cal)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    mid = n // 2
    fit_idx, conf_idx = idx[:mid], idx[mid:]

    g = fit_isotonic(C_cal[fit_idx], L_cal[fit_idx])
    g_alpha = conformalize(g, C_cal[conf_idx], L_cal[conf_idx], alpha)

    # Sweep to find tau_hi
    thresholds = np.linspace(0.0, 1.0, 200)
    risk_at_tau = np.array([g_alpha(t) for t in thresholds])
    valid = risk_at_tau <= alpha
    if valid.any():
        tau_hi = float(thresholds[valid][0])
    else:
        tau_hi = 1.0  # No threshold satisfies the risk bound

    # tau_lo from target coverage quantile
    tau_lo = float(np.quantile(C_cal, 1.0 - target_coverage))
    tau_lo = min(tau_lo, tau_hi)

    return tau_hi, tau_lo
