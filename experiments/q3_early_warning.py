"""Experiment Q3 -- Validate dC/dt as an early warning signal.

Uses CUSUM and Page-Hinkley detectors on the composite Z signal derived
from ICM dynamics (dC/dt), prediction variance, and dependency-penalty
trend.  Compares against naive and variance-only baselines across
multiple change magnitudes, window sizes, and repeated trials.

Key design decisions:
  - Shared probability grid so Hellinger distance captures real divergence.
  - Adaptive threshold calibration from the stable pre-change window.
  - Debounced detections (cooldown after each alarm) to avoid repeated
    alerts on the same sustained shift -- standard practice for online
    change-point detectors.
  - Grace-period-aware evaluation that does not penalize post-change
    sustained alarms.

Run:
    python experiments/q3_early_warning.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# Ensure project root is on the path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from numpy.typing import NDArray

from framework.config import ICMConfig, EarlyWarningConfig
from framework.icm import compute_icm_from_predictions
from framework.early_warning import (
    compute_rolling_icm,
    compute_delta_icm,
    compute_prediction_variance,
    compute_z_signal,
    cusum_detector,
    page_hinkley_detector,
    evaluate_early_warning,
    placebo_test,
)
from benchmarks.synthetic.generators import (
    generate_change_point_series,
    generate_multi_model_predictions,
)


# =====================================================================
# Helper: build time-indexed multi-model predictions with a change-point
# =====================================================================

def build_timeseries_predictions(
    n_timesteps: int,
    change_at: int,
    shift_magnitude: float,
    n_models: int = 5,
    n_bins: int = 20,
    seed: int = 42,
) -> tuple[dict[str, NDArray], list[float], list[float]]:
    """Create per-model prediction time series that agree pre-change
    and diverge post-change.

    All models produce probability distributions over a SHARED grid so
    that Hellinger distance properly captures inter-model divergence.

    Returns
    -------
    predictions_dict_t : {model_name: array(T,)} point predictions
    icm_scores         : list of ICM scores at each time step
    pi_scores          : list of dependency-penalty scores at each step
    """
    rng = np.random.default_rng(seed)
    config = ICMConfig()

    t_axis = np.arange(n_timesteps, dtype=float)
    base_signal = np.sin(2 * np.pi * t_axis / 100) * 0.3

    model_names = [f"model_{i}" for i in range(n_models)]
    predictions_dict_t: dict[str, NDArray] = {}

    for i, name in enumerate(model_names):
        noise = rng.normal(0, 0.05, n_timesteps)
        pred = base_signal.copy() + noise

        # Post-change divergence
        model_bias = shift_magnitude * (i - (n_models - 1) / 2.0) / n_models
        post_noise = rng.normal(0, 0.1 * shift_magnitude, n_timesteps)
        pred[change_at:] += model_bias + post_noise[change_at:]

        predictions_dict_t[name] = pred

    # Shared grid for probability distributions
    all_vals = np.concatenate([predictions_dict_t[n] for n in model_names])
    grid_lo = float(np.min(all_vals)) - 2.0
    grid_hi = float(np.max(all_vals)) + 2.0
    shared_grid = np.linspace(grid_lo, grid_hi, n_bins)

    icm_scores: list[float] = []
    pi_scores: list[float] = []

    for t_idx in range(n_timesteps):
        step_preds: dict[str, NDArray] = {}
        for name in model_names:
            val = predictions_dict_t[name][t_idx]
            probs = np.exp(-0.5 * (shared_grid - val) ** 2)
            probs /= probs.sum()
            step_preds[name] = probs

        result = compute_icm_from_predictions(
            step_preds, config=config, distance_fn="hellinger"
        )
        icm_scores.append(result.icm_score)
        pi_scores.append(result.components.Pi)

    return predictions_dict_t, icm_scores, pi_scores


# =====================================================================
# Debounced detector wrappers
# =====================================================================

def debounce_detections(detections: list[int], cooldown: int) -> list[int]:
    """Keep only the first detection in each cooldown window."""
    if not detections:
        return []
    result = [detections[0]]
    for d in detections[1:]:
        if d - result[-1] >= cooldown:
            result.append(d)
    return result


def cusum_debounced(
    signal: NDArray,
    threshold: float,
    drift: float,
    cooldown: int = 50,
) -> list[int]:
    """CUSUM with post-detection cooldown."""
    raw, _ = cusum_detector(signal, threshold, drift)
    return debounce_detections(raw, cooldown)


def page_hinkley_debounced(
    signal: NDArray,
    threshold: float,
    alpha: float,
    cooldown: int = 50,
) -> list[int]:
    """Page-Hinkley with post-detection cooldown."""
    raw, _ = page_hinkley_detector(signal, threshold, alpha)
    return debounce_detections(raw, cooldown)


# =====================================================================
# Baseline detectors (already edge-triggered)
# =====================================================================

def naive_icm_threshold_detector(
    icm_series: NDArray,
    threshold: float,
) -> list[int]:
    """Flag the first time step in each consecutive run below threshold."""
    icm = np.asarray(icm_series, dtype=float)
    detections = []
    below = False
    for t in range(len(icm)):
        if icm[t] < threshold and not below:
            detections.append(t)
            below = True
        elif icm[t] >= threshold:
            below = False
    return detections


def variance_only_detector(
    var_series: NDArray,
    threshold: float,
) -> list[int]:
    """Flag first crossing per consecutive run above threshold."""
    var = np.asarray(var_series, dtype=float)
    detections = []
    above = False
    for t in range(len(var)):
        if var[t] > threshold and not above:
            detections.append(t)
            above = True
        elif var[t] <= threshold:
            above = False
    return detections


# =====================================================================
# Evaluation with grace period
# =====================================================================

def evaluate_detections(
    detected_changes: list[int],
    true_changes: list[int],
    total_length: int,
    max_lead_time: int,
    grace_period: int = 30,
) -> dict[str, float]:
    """Evaluate detection quality.

    A detection within [tc - max_lead_time, tc + grace_period] is
    associated with true change tc.  The first such detection is TP;
    later ones in the window are ignored (neither TP nor FP).
    Detections outside all windows are false positives.
    """
    detected = sorted(detected_changes)
    true_cps = sorted(true_changes)

    # Exclusion windows
    windows = []
    for tc in true_cps:
        w_start = max(0, tc - max_lead_time)
        w_end = min(total_length, tc + grace_period)
        windows.append((w_start, w_end, tc))

    matched_true: set[int] = set()
    lead_times: list[int] = []
    associated: set[int] = set()

    for d in detected:
        for w_start, w_end, tc in windows:
            if w_start <= d <= w_end:
                associated.add(d)
                if tc not in matched_true and d <= tc:
                    matched_true.add(tc)
                    lead_times.append(tc - d)
                break

    n_tp = len(matched_true)
    n_fp = len(detected) - len(associated)

    tpr = n_tp / max(len(true_cps), 1)
    # FPR relative to non-change timesteps
    window_length = sum(
        min(w_end, total_length) - w_start for w_start, w_end, _ in windows
    )
    n_non_change = max(total_length - window_length, 1)
    fpr = n_fp / n_non_change

    precision = n_tp / max(n_tp + n_fp, 1)
    recall = tpr
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    mean_lead = float(np.mean(lead_times)) if lead_times else 0.0

    return {
        "tpr": tpr,
        "fpr": fpr,
        "precision": precision,
        "f1": f1,
        "mean_lead_time": mean_lead,
        "lead_times": lead_times,
        "n_tp": n_tp,
        "n_fp": n_fp,
        "n_detections": len(detected),
    }


# =====================================================================
# Adaptive threshold calibration
# =====================================================================

def calibrate_thresholds(
    icm_series: NDArray,
    var_series: NDArray,
    z_signal: NDArray,
    calibration_end: int,
) -> dict[str, float]:
    """Calibrate thresholds from the known-stable calibration window."""
    icm_cal = icm_series[:calibration_end]
    var_cal = var_series[:calibration_end]
    z_cal = z_signal[:calibration_end]

    icm_mean, icm_std = float(np.mean(icm_cal)), float(np.std(icm_cal))
    var_mean, var_std = float(np.mean(var_cal)), float(np.std(var_cal))
    z_mean, z_std = float(np.mean(z_cal)), max(float(np.std(z_cal)), 1e-6)

    return {
        "naive_icm": icm_mean - 3.0 * max(icm_std, 1e-4),
        "variance": var_mean + 5.0 * max(var_std, 1e-6),
        "cusum_threshold": 4.0 * z_std,
        "cusum_drift": z_mean + 0.5 * z_std,
        "ph_threshold": 8.0 * z_std,
        "ph_alpha": z_mean + 0.25 * z_std,
        "z_cal_mean": z_mean,
        "z_cal_std": z_std,
    }


# =====================================================================
# Single-trial runner
# =====================================================================

def run_single_trial(
    n_timesteps: int,
    change_at: int,
    shift_magnitude: float,
    window_size: int,
    ew_config: EarlyWarningConfig,
    seed: int,
    max_lead_time: int = 50,
    cooldown: int = 50,
) -> dict:
    """Run one trial and return metrics for all detectors."""

    preds_t, icm_scores, pi_scores = build_timeseries_predictions(
        n_timesteps=n_timesteps,
        change_at=change_at,
        shift_magnitude=shift_magnitude,
        seed=seed,
    )

    icm_arr = np.array(icm_scores)
    pi_arr = np.array(pi_scores)

    delta_icm = compute_delta_icm(icm_arr, window_size)
    var_preds = compute_prediction_variance(preds_t)
    pi_trend = compute_rolling_icm(pi_arr, window_size)
    z_signal = compute_z_signal(delta_icm, var_preds, pi_trend, ew_config)

    calibration_end = max(change_at // 2, window_size + 10)
    thr = calibrate_thresholds(icm_arr, var_preds, z_signal, calibration_end)

    # Detectors
    cusum_det = cusum_debounced(
        z_signal, thr["cusum_threshold"], thr["cusum_drift"], cooldown
    )
    ph_det = page_hinkley_debounced(
        z_signal, thr["ph_threshold"], thr["ph_alpha"], cooldown
    )
    naive_det = naive_icm_threshold_detector(icm_arr, thr["naive_icm"])
    var_det = variance_only_detector(var_preds, thr["variance"])

    true_changes = [change_at]

    results: dict[str, dict] = {}
    for name, detected in [
        ("cusum", cusum_det),
        ("page_hinkley", ph_det),
        ("naive_icm", naive_det),
        ("variance_only", var_det),
    ]:
        ev = evaluate_detections(
            detected, true_changes, n_timesteps, max_lead_time,
            grace_period=cooldown,
        )
        results[name] = ev

    # Also record framework's built-in eval for CUSUM/PH
    for name, detected in [("cusum", cusum_det), ("page_hinkley", ph_det)]:
        ev_fw = evaluate_early_warning(detected, true_changes, max_lead_time)
        results[name]["fw_tpr"] = ev_fw["true_positive_rate"]
        results[name]["fw_fpr"] = ev_fw["false_positive_rate"]

    # Placebo test on stable segment
    stable_start = window_size + 5
    stable_end = change_at - 20
    if stable_end > stable_start + 10:
        stable_periods = [(stable_start, stable_end)]
        for det_name, det_fn in [
            (
                "cusum",
                lambda sig: cusum_debounced(
                    sig, thr["cusum_threshold"], thr["cusum_drift"], cooldown
                ),
            ),
            (
                "page_hinkley",
                lambda sig: page_hinkley_debounced(
                    sig, thr["ph_threshold"], thr["ph_alpha"], cooldown
                ),
            ),
        ]:
            placebo = placebo_test(z_signal, stable_periods, det_fn)
            results[det_name]["placebo_far"] = placebo["false_alarm_rate"]
    else:
        for det_name in ["cusum", "page_hinkley"]:
            results[det_name]["placebo_far"] = float("nan")

    results["_diagnostics"] = {
        "icm_pre_mean": float(np.mean(icm_arr[:change_at])),
        "icm_post_mean": float(np.mean(icm_arr[change_at:])),
        "icm_drop": float(np.mean(icm_arr[:change_at]) - np.mean(icm_arr[change_at:])),
        "var_pre_mean": float(np.mean(var_preds[:change_at])),
        "var_post_mean": float(np.mean(var_preds[change_at:])),
        "z_pre_mean": float(np.mean(z_signal[:change_at])),
        "z_post_mean": float(np.mean(z_signal[change_at:])),
        "z_pre_std": float(np.std(z_signal[:change_at])),
        "z_post_std": float(np.std(z_signal[change_at:])),
        "thresholds": thr,
    }

    return results


# =====================================================================
# Aggregation
# =====================================================================

def aggregate_results(trial_results: list[dict]) -> dict:
    detectors = [k for k in trial_results[0].keys() if not k.startswith("_")]
    agg: dict[str, dict] = {}

    for det in detectors:
        metrics = {
            "tpr": [], "fpr": [], "precision": [], "f1": [],
            "mean_lead_time": [], "n_detections": [],
        }
        for tr in trial_results:
            for m in metrics:
                metrics[m].append(float(tr[det][m]))

        agg[det] = {}
        for m, vals in metrics.items():
            agg[det][f"{m}_mean"] = float(np.mean(vals))
            agg[det][f"{m}_std"] = float(np.std(vals))

        if "placebo_far" in trial_results[0].get(det, {}):
            fars = [
                tr[det]["placebo_far"]
                for tr in trial_results
                if not np.isnan(tr[det].get("placebo_far", float("nan")))
            ]
            if fars:
                agg[det]["placebo_far_mean"] = float(np.mean(fars))
                agg[det]["placebo_far_std"] = float(np.std(fars))
            else:
                agg[det]["placebo_far_mean"] = float("nan")
                agg[det]["placebo_far_std"] = float("nan")

    # Diagnostics
    diag_keys = [
        "icm_pre_mean", "icm_post_mean", "icm_drop",
        "var_pre_mean", "var_post_mean",
        "z_pre_mean", "z_post_mean", "z_pre_std", "z_post_std",
    ]
    diag_agg: dict[str, float] = {}
    for k in diag_keys:
        vals = [tr["_diagnostics"][k] for tr in trial_results]
        diag_agg[k] = float(np.mean(vals))
    agg["_diagnostics"] = diag_agg

    return agg


# =====================================================================
# Pretty printing
# =====================================================================

def fmt(val: float, std: float, dec: int = 3) -> str:
    """Format value +/- std."""
    return f"{val:.{dec}f}+/-{std:.{dec}f}"


def print_table_header(title: str, col_widths: list[int], headers: list[str]):
    total = sum(col_widths)
    print()
    print("=" * total)
    print(title)
    print("=" * total)
    row = ""
    for h, w in zip(headers, col_widths):
        row += h.ljust(w)
    print(row)
    print("-" * total)


def print_table_row(values: list[str], col_widths: list[int]):
    row = ""
    for v, w in zip(values, col_widths):
        row += v.ljust(w)
    print(row)


def print_main_table(
    all_results: dict[str, dict],
    title: str,
    param_name: str,
):
    cw = [16, 16, 14, 14, 12, 12, 12, 8]
    hdr = [param_name, "Detector", "TPR", "FPR", "Prec", "F1", "Lead", "N"]

    print_table_header(title, cw, hdr)

    for pv, agg in sorted(all_results.items()):
        first = True
        for det in ["cusum", "page_hinkley", "naive_icm", "variance_only"]:
            if det not in agg:
                continue
            d = agg[det]
            label = pv if first else ""
            print_table_row(
                [
                    str(label),
                    det,
                    fmt(d["tpr_mean"], d["tpr_std"]),
                    fmt(d["fpr_mean"], d["fpr_std"], 4),
                    fmt(d["precision_mean"], d["precision_std"]),
                    fmt(d["f1_mean"], d["f1_std"]),
                    fmt(d["mean_lead_time_mean"], d["mean_lead_time_std"], 1),
                    f"{d['n_detections_mean']:.1f}",
                ],
                cw,
            )
            first = False
        print("-" * sum(cw))


def print_diagnostics(all_results: dict[str, dict]):
    cw = [18, 12, 12, 12, 14, 14]
    hdr = ["Config", "ICM_pre", "ICM_post", "ICM_drop", "Var_pre", "Var_post"]
    print_table_header("SIGNAL DIAGNOSTICS (averaged over trials)", cw, hdr)
    for pv, agg in sorted(all_results.items()):
        d = agg.get("_diagnostics", {})
        if not d:
            continue
        print_table_row(
            [
                str(pv),
                f"{d.get('icm_pre_mean',0):.4f}",
                f"{d.get('icm_post_mean',0):.4f}",
                f"{d.get('icm_drop',0):.4f}",
                f"{d.get('var_pre_mean',0):.6f}",
                f"{d.get('var_post_mean',0):.6f}",
            ],
            cw,
        )
    print("-" * sum(cw))


def print_placebo(all_results: dict[str, dict]):
    cw = [20, 18, 25]
    hdr = ["Configuration", "Detector", "Placebo FAR"]
    print_table_header("PLACEBO FALSE ALARM RATE (on stable segments)", cw, hdr)
    for pv, agg in sorted(all_results.items()):
        first = True
        for det in ["cusum", "page_hinkley"]:
            if det not in agg:
                continue
            d = agg[det]
            far_m = d.get("placebo_far_mean", float("nan"))
            far_s = d.get("placebo_far_std", float("nan"))
            label = str(pv) if first else ""
            if np.isnan(far_m):
                s = "N/A"
            else:
                s = f"{far_m:.5f}+/-{far_s:.5f}"
            print_table_row([label, det, s], cw)
            first = False
    print("-" * sum(cw))


def print_cross_product(cross_results: dict[str, dict], detector: str, title: str):
    cw = [18, 14, 14, 12, 12, 12]
    hdr = ["Config", "TPR", "FPR", "Prec", "F1", "Lead"]
    print_table_header(title, cw, hdr)
    for key in sorted(cross_results.keys()):
        d = cross_results[key][detector]
        print_table_row(
            [
                key,
                fmt(d["tpr_mean"], d["tpr_std"]),
                fmt(d["fpr_mean"], d["fpr_std"], 4),
                fmt(d["precision_mean"], d["precision_std"]),
                fmt(d["f1_mean"], d["f1_std"]),
                fmt(d["mean_lead_time_mean"], d["mean_lead_time_std"], 1),
            ],
            cw,
        )
    print("-" * sum(cw))


# =====================================================================
# Main experiment
# =====================================================================

def main():
    print("=" * 80)
    print("EXPERIMENT Q3: Validate dC/dt as Early Warning Signal")
    print("  Detectors : CUSUM, Page-Hinkley (with cooldown debounce)")
    print("  Baselines : Naive ICM threshold, Variance-only threshold")
    print("  Framework : OS Multi-Science ICM v1.1")
    print("  Signal    : Z_t = a1*(-dC/dt) + a2*Var(y_hat) + a3*Pi_trend")
    print("=" * 80)

    start_time = time.time()

    n_timesteps = 500
    change_at = 250
    n_reps = 10
    max_lead_time = 50
    cooldown = 50
    base_seed = 42

    ew_config_default = EarlyWarningConfig()

    # ================================================================
    # Experiment 1: Vary change magnitude
    # ================================================================
    print("\n[1/3] Running magnitude sweep (shift = 0.5, 1.5, 3.0) ...")
    shifts = [0.5, 1.5, 3.0]
    shift_labels = {0.5: "small(0.5)", 1.5: "medium(1.5)", 3.0: "large(3.0)"}
    mag_results: dict[str, dict] = {}

    for s in shifts:
        trials = []
        for rep in range(n_reps):
            seed = base_seed + rep * 137
            res = run_single_trial(
                n_timesteps, change_at, s,
                ew_config_default.window_size, ew_config_default, seed,
                max_lead_time, cooldown,
            )
            trials.append(res)
        mag_results[shift_labels[s]] = aggregate_results(trials)
        print(f"    shift={s} done.")

    print_main_table(
        mag_results,
        "EXPERIMENT 1: Effect of Change Magnitude",
        "Magnitude",
    )
    print_diagnostics(mag_results)
    print_placebo(mag_results)

    # ================================================================
    # Experiment 2: Vary window size
    # ================================================================
    print("\n[2/3] Running window-size sweep (w = 20, 50, 100, 200) ...")
    window_sizes = [20, 50, 100, 200]
    win_results: dict[str, dict] = {}

    for ws in window_sizes:
        cfg = EarlyWarningConfig(
            window_size=ws,
            a1=ew_config_default.a1,
            a2=ew_config_default.a2,
            a3=ew_config_default.a3,
            cusum_threshold=ew_config_default.cusum_threshold,
            cusum_drift=ew_config_default.cusum_drift,
        )
        trials = []
        for rep in range(n_reps):
            seed = base_seed + rep * 137
            res = run_single_trial(
                n_timesteps, change_at, 1.5,
                ws, cfg, seed, max_lead_time, cooldown,
            )
            trials.append(res)
        win_results[f"w={ws:03d}"] = aggregate_results(trials)
        print(f"    window_size={ws} done.")

    print_main_table(
        win_results,
        "EXPERIMENT 2: Effect of Window Size (shift=1.5)",
        "Window",
    )
    print_diagnostics(win_results)
    print_placebo(win_results)

    # ================================================================
    # Experiment 3: Full cross-product
    # ================================================================
    print("\n[3/3] Running cross-product sweep ...")
    cross: dict[str, dict] = {}

    for s in shifts:
        for ws in window_sizes:
            cfg = EarlyWarningConfig(
                window_size=ws,
                a1=ew_config_default.a1,
                a2=ew_config_default.a2,
                a3=ew_config_default.a3,
                cusum_threshold=ew_config_default.cusum_threshold,
                cusum_drift=ew_config_default.cusum_drift,
            )
            trials = []
            for rep in range(n_reps):
                seed = base_seed + rep * 137
                res = run_single_trial(
                    n_timesteps, change_at, s,
                    ws, cfg, seed, max_lead_time, cooldown,
                )
                trials.append(res)
            cross[f"s={s},w={ws:03d}"] = aggregate_results(trials)

    print_cross_product(cross, "cusum",
        "CROSS-PRODUCT: CUSUM (magnitude x window)")
    print_cross_product(cross, "page_hinkley",
        "CROSS-PRODUCT: Page-Hinkley (magnitude x window)")

    # ================================================================
    # Summary
    # ================================================================
    elapsed = time.time() - start_time
    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)

    for det_label, det_key in [("CUSUM", "cusum"), ("Page-Hinkley", "page_hinkley")]:
        best_key = max(
            cross.keys(),
            key=lambda k: cross[k][det_key]["f1_mean"],
        )
        b = cross[best_key][det_key]
        print(
            f"Best {det_label} (by F1): {best_key}  "
            f"TPR={b['tpr_mean']:.3f}  FPR={b['fpr_mean']:.5f}  "
            f"Prec={b['precision_mean']:.3f}  F1={b['f1_mean']:.3f}  "
            f"Lead={b['mean_lead_time_mean']:.1f}"
        )

    ref_key = "medium(1.5)"
    if ref_key in mag_results:
        ref = mag_results[ref_key]
        print()
        print("Detector comparison at medium shift (1.5), window=100:")
        print(f"  {'Detector':<18s} {'TPR':>6s} {'FPR':>8s} {'Prec':>6s} {'F1':>6s} {'Lead':>6s} {'N':>4s}")
        print(f"  {'-'*56}")
        for det in ["cusum", "page_hinkley", "naive_icm", "variance_only"]:
            d = ref[det]
            print(
                f"  {det:<18s} {d['tpr_mean']:6.3f} {d['fpr_mean']:8.5f} "
                f"{d['precision_mean']:6.3f} {d['f1_mean']:6.3f} "
                f"{d['mean_lead_time_mean']:6.1f} {d['n_detections_mean']:4.1f}"
            )

    print()
    n_total = (len(shifts) + len(window_sizes) + len(shifts)*len(window_sizes)) * n_reps
    print(f"Total experiment time: {elapsed:.1f}s")
    print(f"Configurations tested: {len(cross)}")
    print(f"Total trials: {n_total}")
    print()

    return {
        "magnitude_results": mag_results,
        "window_results": win_results,
        "cross_results": cross,
        "n_reps": n_reps,
        "elapsed": elapsed,
    }


if __name__ == "__main__":
    main()
