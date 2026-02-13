"""Timing benchmark for OS Multi-Science computational complexity analysis.

Measures wall-clock time for core operations across varying problem sizes:
  - ICM computation for varying K (models) and N (samples)
  - HSIC permutation test for varying N and n_permutations
  - Full anti-spurious pipeline
  - CRC gating (isotonic + conformal calibration)
  - Early warning detectors (CUSUM, Page-Hinkley)
  - Distance functions (Hellinger, Wasserstein, MMD)
  - Meta-learner grid search

Usage:
    python -m benchmarks.timing_benchmark
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Ensure project root is on the path
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from framework.config import (
    AntiSpuriousConfig,
    CRCConfig,
    EarlyWarningConfig,
    ICMConfig,
)
from framework.icm import (
    compute_agreement,
    compute_icm_from_predictions,
    hellinger_distance,
    mmd_distance,
    wasserstein2_distance,
    wasserstein2_empirical,
)
from framework.crc_gating import (
    calibrate_thresholds,
    conformalize,
    fit_isotonic,
)
from framework.early_warning import (
    compute_rolling_icm,
    cusum_detector,
    page_hinkley_detector,
)
from framework.anti_spurious import (
    generate_negative_controls,
    hsic_test,
    generate_anti_spurious_report,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _time_fn(fn, *args, n_repeats: int = 3, **kwargs) -> float:
    """Return the median wall-clock time (seconds) over n_repeats calls."""
    times = []
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        fn(*args, **kwargs)
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return float(np.median(times))


def _make_predictions(K: int, N: int, C: int, rng: np.random.Generator):
    """Generate K softmax prediction vectors of dimension C for N samples."""
    preds = {}
    for k in range(K):
        raw = rng.dirichlet(np.ones(C), size=N)  # (N, C)
        preds[f"model_{k}"] = raw
    return preds


def _make_scalar_predictions(K: int, N: int, rng: np.random.Generator):
    """Generate K scalar prediction arrays of length N."""
    return {f"model_{k}": rng.normal(0, 1, size=N) for k in range(K)}


def _format_table(headers: list[str], rows: list[list[str]], col_widths: list[int] | None = None) -> str:
    """Format a simple ASCII table."""
    if col_widths is None:
        col_widths = [max(len(h), max((len(r[i]) for r in rows), default=0)) + 2 for i, h in enumerate(headers)]
    lines = []
    header_line = "| " + " | ".join(h.ljust(w) for h, w in zip(headers, col_widths)) + " |"
    sep_line = "|-" + "-|-".join("-" * w for w in col_widths) + "-|"
    lines.append(header_line)
    lines.append(sep_line)
    for row in rows:
        lines.append("| " + " | ".join(r.ljust(w) for r, w in zip(row, col_widths)) + " |")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Benchmark 1: ICM computation -- vary K
# ---------------------------------------------------------------------------

def bench_icm_vary_k():
    """Measure ICM computation time as K (number of models) varies."""
    print("\n=== Benchmark 1: ICM computation -- vary K ===")
    rng = np.random.default_rng(42)
    N = 500
    C = 5
    K_values = [2, 3, 5, 7, 10, 15, 20]
    config = ICMConfig()

    headers = ["K", "Time (ms)", "Time per pair (ms)", "Pairs"]
    rows = []

    for K in K_values:
        preds = _make_predictions(K, N, C, rng)
        # Average each model's predictions to get a single distribution per model
        preds_avg = {k: v.mean(axis=0) for k, v in preds.items()}
        t = _time_fn(compute_icm_from_predictions, preds_avg, config=config, n_repeats=5)
        n_pairs = K * (K - 1) // 2
        t_per_pair = (t / n_pairs * 1000) if n_pairs > 0 else 0
        rows.append([str(K), f"{t*1000:.2f}", f"{t_per_pair:.3f}", str(n_pairs)])

    print(_format_table(headers, rows))
    return headers, rows


# ---------------------------------------------------------------------------
# Benchmark 2: ICM computation -- vary N
# ---------------------------------------------------------------------------

def bench_icm_vary_n():
    """Measure ICM computation time as N (number of samples) varies."""
    print("\n=== Benchmark 2: ICM computation -- vary N ===")
    rng = np.random.default_rng(42)
    K = 5
    C = 5
    N_values = [100, 500, 1000, 2000, 5000, 10000]
    config = ICMConfig()

    headers = ["N", "Time (ms)", "Notes"]
    rows = []

    for N in N_values:
        preds = _make_predictions(K, N, C, rng)
        # Use per-sample average distributions for agreement
        preds_avg = {k: v.mean(axis=0) for k, v in preds.items()}
        # Build intervals from quantiles
        intervals = []
        for k, v in preds.items():
            mean_pred = v.mean(axis=1)
            intervals.append((float(np.percentile(mean_pred, 10)),
                              float(np.percentile(mean_pred, 90))))

        t = _time_fn(
            compute_icm_from_predictions,
            preds_avg,
            config=config,
            intervals=intervals,
            n_repeats=5,
        )
        rows.append([str(N), f"{t*1000:.2f}", "5 models, C=5"])

    print(_format_table(headers, rows))
    return headers, rows


# ---------------------------------------------------------------------------
# Benchmark 3: Distance functions
# ---------------------------------------------------------------------------

def bench_distances():
    """Measure distance function computation time."""
    print("\n=== Benchmark 3: Distance functions ===")
    rng = np.random.default_rng(42)

    headers = ["Distance", "Dim/N", "Time (ms)"]
    rows = []

    # Hellinger on discrete distributions
    for C in [5, 20, 100, 500]:
        p = rng.dirichlet(np.ones(C))
        q = rng.dirichlet(np.ones(C))
        t = _time_fn(hellinger_distance, p, q, n_repeats=20)
        rows.append(["Hellinger", f"C={C}", f"{t*1000:.4f}"])

    # Wasserstein-2 (Gaussian, closed-form)
    for D in [2, 10, 50, 100]:
        mu1, mu2 = rng.normal(size=D), rng.normal(size=D)
        A1, A2 = rng.normal(size=(D, D)), rng.normal(size=(D, D))
        s1, s2 = A1 @ A1.T + np.eye(D), A2 @ A2.T + np.eye(D)
        t = _time_fn(wasserstein2_distance, mu1, s1, mu2, s2, n_repeats=5)
        rows.append(["W2-Gaussian", f"D={D}", f"{t*1000:.3f}"])

    # Wasserstein-2 (Empirical, 1-D)
    for N in [100, 500, 1000, 5000]:
        X = rng.normal(size=N)
        Y = rng.normal(0.5, 1.2, size=N)
        t = _time_fn(wasserstein2_empirical, X, Y, n_repeats=5)
        rows.append(["W2-Emp-1D", f"N={N}", f"{t*1000:.3f}"])

    # MMD with RBF kernel
    for N in [50, 100, 500, 1000]:
        D_feat = 5
        X = rng.normal(size=(N, D_feat))
        Y = rng.normal(0.5, 1.0, size=(N, D_feat))
        t = _time_fn(mmd_distance, X, Y, n_repeats=3)
        rows.append(["MMD-RBF", f"N={N},D={D_feat}", f"{t*1000:.3f}"])

    print(_format_table(headers, rows))
    return headers, rows


# ---------------------------------------------------------------------------
# Benchmark 4: HSIC permutation test
# ---------------------------------------------------------------------------

def bench_hsic():
    """Measure HSIC test time for varying N and n_permutations."""
    print("\n=== Benchmark 4: HSIC permutation test ===")
    rng = np.random.default_rng(42)

    headers = ["N", "n_perms", "Time (ms)", "Time per perm (ms)"]
    rows = []

    for N in [50, 100, 200, 500]:
        for n_perms in [100, 500, 1000]:
            r1 = rng.normal(size=N)
            r2 = rng.normal(size=N) + 0.3 * r1
            t = _time_fn(hsic_test, r1, r2, n_perms, n_repeats=3)
            t_per_perm = t / n_perms * 1000
            rows.append([str(N), str(n_perms), f"{t*1000:.1f}", f"{t_per_perm:.3f}"])

    print(_format_table(headers, rows))
    return headers, rows


# ---------------------------------------------------------------------------
# Benchmark 5: CRC gating pipeline
# ---------------------------------------------------------------------------

def bench_crc():
    """Measure CRC gating (isotonic fit + conformal calibration) time."""
    print("\n=== Benchmark 5: CRC gating pipeline ===")
    rng = np.random.default_rng(42)

    headers = ["N_cal", "Isotonic fit (ms)", "Conformalize (ms)", "Calibrate thresholds (ms)"]
    rows = []

    for N in [100, 500, 1000, 5000]:
        C_vals = rng.uniform(0, 1, size=N)
        L_vals = 1.0 - C_vals + rng.normal(0, 0.1, size=N)
        L_vals = np.clip(L_vals, 0, 2)

        t_iso = _time_fn(fit_isotonic, C_vals, L_vals, n_repeats=5)

        g = fit_isotonic(C_vals[:N//2], L_vals[:N//2])
        t_conf = _time_fn(conformalize, g, C_vals[N//2:], L_vals[N//2:], n_repeats=5)

        t_cal = _time_fn(calibrate_thresholds, C_vals, L_vals, n_repeats=3)

        rows.append([
            str(N),
            f"{t_iso*1000:.2f}",
            f"{t_conf*1000:.2f}",
            f"{t_cal*1000:.2f}",
        ])

    print(_format_table(headers, rows))
    return headers, rows


# ---------------------------------------------------------------------------
# Benchmark 6: Early warning detectors
# ---------------------------------------------------------------------------

def bench_early_warning():
    """Measure early warning system computation time."""
    print("\n=== Benchmark 6: Early warning detectors ===")
    rng = np.random.default_rng(42)

    headers = ["T", "Window", "Rolling ICM (ms)", "CUSUM (ms)", "Page-Hinkley (ms)"]
    rows = []

    for T in [100, 500, 1000, 5000, 10000]:
        signal = rng.normal(0, 1, size=T)
        icm_series = np.clip(0.5 + 0.1 * np.cumsum(rng.normal(0, 0.01, T)), 0, 1)
        window = min(50, T // 5)

        t_roll = _time_fn(compute_rolling_icm, icm_series, window, n_repeats=5)
        t_cusum = _time_fn(cusum_detector, signal, 5.0, 0.5, n_repeats=5)
        t_ph = _time_fn(page_hinkley_detector, signal, 5.0, n_repeats=5)

        rows.append([
            str(T),
            str(window),
            f"{t_roll*1000:.2f}",
            f"{t_cusum*1000:.2f}",
            f"{t_ph*1000:.2f}",
        ])

    print(_format_table(headers, rows))
    return headers, rows


# ---------------------------------------------------------------------------
# Benchmark 7: Negative controls + full anti-spurious
# ---------------------------------------------------------------------------

def bench_anti_spurious():
    """Measure anti-spurious pipeline time."""
    print("\n=== Benchmark 7: Anti-spurious pipeline ===")
    rng = np.random.default_rng(42)

    headers = ["K", "N", "n_ctrl", "n_perms", "Neg ctrl (ms)", "Full report (ms)"]
    rows = []

    for K, N, n_ctrl, n_perms in [
        (3, 100, 50, 100),
        (5, 100, 100, 500),
        (5, 500, 100, 500),
        (5, 500, 100, 1000),
    ]:
        preds = {f"m{i}": rng.normal(0, 1, size=N) for i in range(K)}
        labels = rng.normal(0, 1, size=N)
        features = rng.normal(0, 1, size=(N, 5))
        pred_matrix = np.stack(list(preds.values()))

        config = AntiSpuriousConfig(
            n_negative_controls=n_ctrl,
            n_permutations=n_perms,
        )

        t_ctrl = _time_fn(
            generate_negative_controls,
            pred_matrix, labels, features,
            method="label_shuffle",
            n_controls=n_ctrl,
            n_repeats=3,
        )

        t_full = _time_fn(
            generate_anti_spurious_report,
            preds, labels, features, config,
            n_repeats=2,
        )

        rows.append([
            str(K), str(N), str(n_ctrl), str(n_perms),
            f"{t_ctrl*1000:.1f}", f"{t_full*1000:.1f}",
        ])

    print(_format_table(headers, rows))
    return headers, rows


# ---------------------------------------------------------------------------
# Benchmark 8: Full end-to-end pipeline
# ---------------------------------------------------------------------------

def bench_full_pipeline():
    """Measure complete end-to-end pipeline time."""
    print("\n=== Benchmark 8: Full end-to-end pipeline ===")
    rng = np.random.default_rng(42)

    headers = ["K", "N", "Total (ms)", "ICM (ms)", "CRC (ms)", "Notes"]
    rows = []

    for K, N in [(3, 100), (5, 500), (10, 1000), (5, 5000)]:
        C = 5
        preds = _make_predictions(K, N, C, rng)
        preds_avg = {k: v.mean(axis=0) for k, v in preds.items()}
        y_true = rng.integers(0, C, size=N)
        config = ICMConfig()

        # ICM time
        t_icm = _time_fn(
            compute_icm_from_predictions,
            preds_avg, config=config,
            n_repeats=5,
        )

        # CRC time (isotonic + conformalize + decision gate)
        icm_scores = rng.uniform(0, 1, size=N)
        losses = 1.0 - icm_scores + rng.normal(0, 0.1, size=N)
        losses = np.clip(losses, 0, 2)
        t_crc = _time_fn(calibrate_thresholds, icm_scores, losses, n_repeats=3)

        t_total = t_icm + t_crc

        rows.append([
            str(K), str(N),
            f"{t_total*1000:.1f}",
            f"{t_icm*1000:.2f}",
            f"{t_crc*1000:.1f}",
            f"C={C}",
        ])

    print(_format_table(headers, rows))
    return headers, rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("OS Multi-Science -- Computational Timing Benchmark")
    print("=" * 70)
    print(f"NumPy version: {np.__version__}")
    print(f"Platform: {sys.platform}")
    print()

    results = {}
    results["icm_k"] = bench_icm_vary_k()
    results["icm_n"] = bench_icm_vary_n()
    results["distances"] = bench_distances()
    results["hsic"] = bench_hsic()
    results["crc"] = bench_crc()
    results["early_warning"] = bench_early_warning()
    results["anti_spurious"] = bench_anti_spurious()
    results["full_pipeline"] = bench_full_pipeline()

    print("\n" + "=" * 70)
    print("All benchmarks complete.")
    print("=" * 70)

    return results


if __name__ == "__main__":
    main()
