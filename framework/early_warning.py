"""Early warning system for OS Multi-Science.

Detects degradation in inter-model convergence before it manifests
as decision failure, using rolling statistics and change-point detection.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from framework.config import EarlyWarningConfig


# ---------------------------------------------------------------------------
# Rolling statistics
# ---------------------------------------------------------------------------

def compute_rolling_icm(
    icm_series: np.ndarray,
    window_size: int,
) -> np.ndarray:
    """Rolling mean of ICM scores.

    Parameters
    ----------
    icm_series : array of shape (T,)
        Time series of ICM scores.
    window_size : int
        Window length for the rolling mean.

    Returns
    -------
    array of shape (T,)
        Rolling mean; the first ``window_size - 1`` entries use a growing
        window (same as ``min_periods=1``).
    """
    icm_series = np.asarray(icm_series, dtype=float)
    T = len(icm_series)
    result = np.empty(T)
    cumsum = np.cumsum(icm_series)
    for t in range(T):
        start = max(0, t - window_size + 1)
        if start == 0:
            result[t] = cumsum[t] / (t + 1)
        else:
            result[t] = (cumsum[t] - cumsum[start - 1]) / window_size
    return result


def compute_delta_icm(
    icm_series: np.ndarray,
    window_size: int,
) -> np.ndarray:
    """Approximate dC/dt via finite differences on the rolling ICM.

    Parameters
    ----------
    icm_series : array of shape (T,)
    window_size : int

    Returns
    -------
    array of shape (T,)
        Forward differences of the rolling mean; first element is 0.
    """
    rolling = compute_rolling_icm(icm_series, window_size)
    delta = np.zeros_like(rolling)
    delta[1:] = np.diff(rolling)
    return delta


# ---------------------------------------------------------------------------
# Prediction variance
# ---------------------------------------------------------------------------

def compute_prediction_variance(
    predictions_dict_t: dict[str, np.ndarray],
) -> np.ndarray:
    """Variance across models at each time step.

    Parameters
    ----------
    predictions_dict_t : dict mapping model name -> array of shape (T,)
        Time-indexed predictions from each model.

    Returns
    -------
    array of shape (T,)
        Point-wise variance across models.
    """
    preds = np.stack(list(predictions_dict_t.values()))  # (M, T)
    return np.var(preds, axis=0, ddof=0)


# ---------------------------------------------------------------------------
# Composite Z signal
# ---------------------------------------------------------------------------

def compute_z_signal(
    delta_icm: np.ndarray,
    var_predictions: np.ndarray,
    pi_trend: np.ndarray,
    config: EarlyWarningConfig,
) -> np.ndarray:
    """Composite early-warning signal.

    Z_t = a1 * (-delta_ICM) + a2 * Var + a3 * Pi

    Parameters
    ----------
    delta_icm : array of shape (T,)
    var_predictions : array of shape (T,)
    pi_trend : array of shape (T,)
        Dependency penalty trend.
    config : EarlyWarningConfig

    Returns
    -------
    array of shape (T,)
    """
    delta_icm = np.asarray(delta_icm, dtype=float)
    var_predictions = np.asarray(var_predictions, dtype=float)
    pi_trend = np.asarray(pi_trend, dtype=float)
    return (
        config.a1 * (-delta_icm)
        + config.a2 * var_predictions
        + config.a3 * pi_trend
    )


# ---------------------------------------------------------------------------
# CUSUM change-point detector
# ---------------------------------------------------------------------------

def cusum_detector(
    signal: np.ndarray,
    threshold: float,
    drift: float,
) -> tuple[list[int], np.ndarray]:
    """Tabular CUSUM change-point detection (upper side).

    Parameters
    ----------
    signal : array of shape (T,)
        The monitored signal (e.g., Z_t).
    threshold : float
        Decision boundary h; a change is flagged when CUSUM > h.
    drift : float
        Allowance parameter k (half the expected shift).

    Returns
    -------
    (change_points, cusum_values)
        ``change_points`` is a list of time indices where the CUSUM
        exceeds the threshold; ``cusum_values`` is the full CUSUM trace.
    """
    signal = np.asarray(signal, dtype=float)
    T = len(signal)
    cusum = np.zeros(T)
    change_points: list[int] = []

    for t in range(1, T):
        cusum[t] = max(0.0, cusum[t - 1] + signal[t] - drift)
        if cusum[t] > threshold:
            change_points.append(t)
            cusum[t] = 0.0  # reset after detection

    return change_points, cusum


# ---------------------------------------------------------------------------
# Page-Hinkley change-point detector
# ---------------------------------------------------------------------------

def page_hinkley_detector(
    signal: np.ndarray,
    threshold: float,
    alpha: float = 0.005,
) -> tuple[list[int], np.ndarray]:
    """Page-Hinkley test for change-point detection.

    Parameters
    ----------
    signal : array of shape (T,)
    threshold : float
        Detection threshold lambda.
    alpha : float
        Tolerance parameter (minimum magnitude of change to detect).

    Returns
    -------
    (change_points, statistics)
    """
    signal = np.asarray(signal, dtype=float)
    T = len(signal)
    m_t = 0.0  # cumulative sum
    M_t = 0.0  # running minimum
    stats = np.zeros(T)
    change_points: list[int] = []

    for t in range(T):
        m_t += signal[t] - alpha
        M_t = min(M_t, m_t)
        ph = m_t - M_t
        stats[t] = ph
        if ph > threshold:
            change_points.append(t)
            # Reset
            m_t = 0.0
            M_t = 0.0

    return change_points, stats


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_early_warning(
    detected_changes: list[int],
    true_changes: list[int],
    max_lead_time: int,
    total_timesteps: int | None = None,
) -> dict[str, float | list[int]]:
    """Evaluate early-warning performance against known change-points.

    A detection is a true positive if it falls within ``max_lead_time``
    steps *before* a true change-point.

    Parameters
    ----------
    detected_changes : list of int
        Time indices of detected changes.
    true_changes : list of int
        Time indices of actual changes.
    max_lead_time : int
        Maximum look-ahead window.
    total_timesteps : int, optional
        Total number of timesteps in the signal.  Used for computing
        true negatives and proper FPR.  If None, estimated from detected
        and true change indices.

    Returns
    -------
    dict with 'informedness', 'lead_times', 'false_positive_rate',
    'true_positive_rate'.  Also includes legacy key 'auroc' (alias
    for 'informedness') for backward compatibility.
    """
    detected = set(detected_changes)
    true_set = set(true_changes)

    tp_lead_times: list[int] = []
    matched_true: set[int] = set()

    for d in sorted(detected):
        for tc in true_changes:
            if 0 <= tc - d <= max_lead_time and tc not in matched_true:
                tp_lead_times.append(tc - d)
                matched_true.add(tc)
                break

    n_tp = len(matched_true)
    n_fn = len(true_set) - n_tp
    n_fp = len(detected) - n_tp

    tpr = n_tp / max(len(true_set), 1)

    # FPR = FP / (FP + TN) where TN is non-change timesteps without detection
    if total_timesteps is not None:
        n_non_change = total_timesteps - len(true_set)
    else:
        # Estimate total timesteps from the data
        all_indices = list(detected) + list(true_set)
        n_total = max(all_indices) + 1 if all_indices else 1
        n_non_change = n_total - len(true_set)
    n_tn = max(n_non_change - n_fp, 0)
    fpr = n_fp / max(n_fp + n_tn, 1)

    # Informedness (Youden's J statistic) = TPR - FPR
    informedness = tpr - fpr if (tpr + fpr) > 0 else 0.0

    return {
        "informedness": informedness,
        "auroc": informedness,  # backward-compat alias
        "lead_times": tp_lead_times,
        "false_positive_rate": fpr,
        "true_positive_rate": tpr,
    }


# ---------------------------------------------------------------------------
# Placebo test
# ---------------------------------------------------------------------------

def placebo_test(
    signal: np.ndarray,
    stable_periods: list[tuple[int, int]],
    detector_fn: Callable[[np.ndarray], list[int]],
) -> dict[str, float]:
    """Run detector on known-stable periods to estimate false alarm rate.

    Parameters
    ----------
    signal : array of shape (T,)
    stable_periods : list of (start, end) index tuples
        Periods known to be stable (no real change).
    detector_fn : callable
        Takes a signal array, returns list of change-point indices.

    Returns
    -------
    dict with 'false_alarm_rate'.
    """
    total_length = 0
    total_alarms = 0

    for start, end in stable_periods:
        segment = signal[start:end]
        alarms = detector_fn(segment)
        total_alarms += len(alarms)
        total_length += len(segment)

    far = total_alarms / max(total_length, 1)
    return {"false_alarm_rate": far}
