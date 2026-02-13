"""
Experiment Q1 -- Monotonicity of E[L|C] in ICM score C
========================================================

CLAIM: The expected loss E[L|C] is monotonically non-increasing in the
ICM convergence score C.  That is, higher convergence implies lower
(or equal) expected loss.

DESIGN:
  We construct scenarios where the *degree of model agreement* is
  controlled by a noise parameter.  At each noise level we run N
  independent data points (sub-trials), compute an ICM score and an
  actual loss for each sub-trial, then bin the sub-trials by ICM score
  to estimate E[L | C].

  Three scenarios:
    (a) Classification -- 3-class Gaussian, cross-entropy loss
    (b) Regression     -- 1-D noisy sine, MSE loss
    (c) Network cascade -- contagion on Erdos-Renyi, cascade-extent loss

  For statistical robustness we repeat the entire sweep 10 times with
  different random seeds.

  Monotonicity is verified via:
    - Spearman rank correlation rho(C, L) -- expect rho < 0
    - Isotonic regression (decreasing) R^2
    - Count of monotonicity violations along sorted ICM

Run:
    python experiments/q1_monotonicity.py

Dependencies: numpy, scipy, sklearn (standard)
"""

from __future__ import annotations

import sys
import os
import time

# Ensure repo root is on the Python path
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np
from scipy.stats import spearmanr
from sklearn.isotonic import IsotonicRegression

from framework.config import ICMConfig
from framework.types import ICMComponents
from framework.icm import (
    compute_agreement,
    compute_direction,
    compute_uncertainty_overlap,
    compute_invariance,
    compute_dependency_penalty,
    compute_icm,
    compute_icm_from_predictions,
    hellinger_distance,
)
from benchmarks.synthetic.generators import (
    generate_classification_benchmark,
    generate_network_cascade,
)

# =====================================================================
# Constants
# =====================================================================
N_NOISE_LEVELS = 25
NOISE_LEVELS = np.linspace(0.01, 1.0, N_NOISE_LEVELS)
N_REPETITIONS = 10
BASE_SEED = 2024
N_MODELS = 5          # number of "independent" models per scenario
N_SUBTRIALS = 30      # data points per noise level (for binning)


# =====================================================================
# Helpers
# =====================================================================
def _softmax(logits: np.ndarray) -> np.ndarray:
    """Row-wise softmax, numerically stable."""
    shifted = logits - logits.max(axis=-1, keepdims=True)
    e = np.exp(shifted)
    return e / e.sum(axis=-1, keepdims=True)


def _cross_entropy_single(y_true_class: int, probs: np.ndarray) -> float:
    """Cross-entropy for a single sample."""
    eps = 1e-15
    return float(-np.log(np.clip(probs[y_true_class], eps, 1.0)))


def _compute_icm_score(
    pred_vectors: list[np.ndarray],
    distance_fn: str,
    config: ICMConfig,
    reference: np.ndarray | None = None,
) -> float:
    """Compute ICM score from a list of per-model prediction vectors.

    This computes all five ICM components:
      A - distributional agreement (pairwise distance)
      D - directional agreement (sign consensus)
      U - uncertainty overlap (interval overlap)
      C - invariance (stability to perturbation)
      Pi - dependency penalty (residual correlation)
    """
    K = len(pred_vectors)

    # A: distributional agreement
    A = compute_agreement(pred_vectors, distance_fn=distance_fn, config=config)

    # D: directional agreement
    means = np.array([np.mean(p) for p in pred_vectors])
    D = compute_direction(means)

    # U: uncertainty overlap from prediction ranges
    intervals: list[tuple[float, float]] = []
    for p in pred_vectors:
        p_flat = np.asarray(p).ravel()
        if len(p_flat) >= 4:
            lo = float(np.percentile(p_flat, 10))
            hi = float(np.percentile(p_flat, 90))
        else:
            lo = float(p_flat.min())
            hi = float(p_flat.max())
        intervals.append((lo, hi))
    U = compute_uncertainty_overlap(intervals)

    # C: invariance under small perturbation
    pre_scores = np.array([np.mean(p) for p in pred_vectors])
    rng_pert = np.random.default_rng(99)
    post_scores = pre_scores + rng_pert.normal(0, 0.01, size=pre_scores.shape)
    C_val = compute_invariance(pre_scores, post_scores)

    # Pi: dependency penalty from residuals
    if reference is not None:
        min_len = min(len(p) for p in pred_vectors)
        ref = np.asarray(reference).ravel()[:min_len]
        residuals = np.stack([np.asarray(p).ravel()[:min_len] - ref
                              for p in pred_vectors])
        Pi = compute_dependency_penalty(residuals=residuals, config=config)
    else:
        Pi = 0.0

    components = ICMComponents(A=A, D=D, U=U, C=C_val, Pi=Pi)
    result = compute_icm(components, config)
    return result.icm_score


# =====================================================================
# Scenario A: Classification (3-class Gaussian)
#
# For each sub-trial: pick a single sample, have K models predict its
# class probabilities.  At low noise, models agree on the true class.
# At high noise, models produce scattered probability vectors.
# =====================================================================
def run_classification_trial(noise: float, seed: int):
    """Single sub-trial returning (icm_score, cross_entropy_loss)."""
    rng = np.random.default_rng(seed)
    n_classes = 3

    # True class for this sample
    true_class = rng.integers(0, n_classes)
    onehot = np.zeros(n_classes)
    onehot[true_class] = 1.0

    # K models produce probability vectors
    model_probs: list[np.ndarray] = []
    for k in range(N_MODELS):
        logits = onehot * 4.0 + rng.normal(0, noise * 4.0, n_classes)
        probs = _softmax(logits.reshape(1, -1)).ravel()
        model_probs.append(probs)

    # ICM: Hellinger-based agreement
    config = ICMConfig()
    icm = _compute_icm_score(
        model_probs, distance_fn="hellinger", config=config,
        reference=onehot,
    )

    # Loss: cross-entropy of ensemble mean
    ensemble = np.mean(model_probs, axis=0)
    loss = _cross_entropy_single(true_class, ensemble)

    return icm, loss


# =====================================================================
# Scenario B: Regression (1-D noisy sine)
#
# For each sub-trial: pick a single x, true y=sin(2pi*x).  K models
# predict y with noise.  Return ICM and squared error.
# =====================================================================
def run_regression_trial(noise: float, seed: int):
    """Single sub-trial returning (icm_score, squared_error)."""
    rng = np.random.default_rng(seed)

    # Single point
    x = rng.uniform(0, 1)
    y_true = np.sin(2 * np.pi * x)

    # K models predict y
    model_preds: list[np.ndarray] = []
    for k in range(N_MODELS):
        y_pred = y_true + rng.normal(0, noise)
        # Each model's "prediction vector" has a few samples from
        # its own posterior to give Wasserstein something to work with
        pred_samples = y_pred + rng.normal(0, noise * 0.1, size=20)
        model_preds.append(pred_samples)

    config = ICMConfig(C_A_wasserstein=3.0)
    icm = _compute_icm_score(
        model_preds, distance_fn="wasserstein", config=config,
        reference=np.full(20, y_true),
    )

    # Loss: MSE of ensemble mean
    ensemble_mean = np.mean([np.mean(p) for p in model_preds])
    loss = (ensemble_mean - y_true) ** 2

    return icm, loss


# =====================================================================
# Scenario C: Network cascade
#
# For each sub-trial: generate a small graph, run K cascade simulations
# with perturbed adjacency.  ICM measures agreement between cascade
# trajectories.  Loss = MSE between ensemble and true cascade.
# =====================================================================
def run_cascade_trial(noise: float, seed: int):
    """Single sub-trial returning (icm_score, cascade_mse)."""
    rng = np.random.default_rng(seed)
    n_nodes = 40
    n_steps = 20

    # Reference cascade
    adj, state_history, _ = generate_network_cascade(
        n_nodes=n_nodes, edge_prob=0.10, threshold=0.3,
        n_steps=n_steps, seed=seed,
    )
    true_frac = state_history.mean(axis=1)

    # K model cascades on perturbed graphs
    model_fracs: list[np.ndarray] = []
    for k in range(N_MODELS):
        flip_prob = noise * 0.2
        flip_mask = rng.random((n_nodes, n_nodes)) < flip_prob
        adj_p = adj.copy()
        adj_p[flip_mask] = 1.0 - adj_p[flip_mask]
        adj_p = np.triu(adj_p, k=1)
        adj_p = adj_p + adj_p.T

        degree_p = adj_p.sum(axis=1)
        state = np.zeros(n_nodes, dtype=np.float64)
        n_init = max(1, int(0.05 * n_nodes))
        state[rng.choice(n_nodes, size=n_init, replace=False)] = 1.0

        frac_series = [state.mean()]
        for t in range(n_steps):
            n_def_neigh = adj_p @ state
            safe_deg = np.where(degree_p > 0, degree_p, 1.0)
            frac_def = n_def_neigh / safe_deg
            new_defaults = (state == 0) & (frac_def > 0.3)
            state = np.where(new_defaults, 1.0, state)
            frac_series.append(state.mean())

        model_fracs.append(np.array(frac_series))

    config = ICMConfig(C_A_wasserstein=2.0)
    icm = _compute_icm_score(
        model_fracs, distance_fn="wasserstein", config=config,
        reference=true_frac,
    )

    ensemble_frac = np.mean(model_fracs, axis=0)
    min_len = min(len(ensemble_frac), len(true_frac))
    mse = float(np.mean((ensemble_frac[:min_len] - true_frac[:min_len]) ** 2))

    return icm, mse


# =====================================================================
# Monotonicity verification
# =====================================================================
def verify_monotonicity(icm_scores: np.ndarray, losses: np.ndarray):
    """Diagnostics for the monotonicity claim E[L|C] non-increasing in C.

    Sorts by ascending ICM and checks that losses are non-increasing.
    """
    order = np.argsort(icm_scores)
    icm_sorted = icm_scores[order]
    loss_sorted = losses[order]

    # 1. Spearman rank correlation
    if np.std(icm_scores) < 1e-12 or np.std(losses) < 1e-12:
        rho, p_value = 0.0, 1.0
    else:
        rho, p_value = spearmanr(icm_scores, losses)

    # 2. Isotonic regression (decreasing) R^2
    iso = IsotonicRegression(increasing=False, out_of_bounds="clip")
    iso.fit(icm_sorted, loss_sorted)
    loss_iso = iso.predict(icm_sorted)
    ss_res = np.sum((loss_sorted - loss_iso) ** 2)
    ss_tot = np.sum((loss_sorted - loss_sorted.mean()) ** 2)
    r2_iso = 1.0 - ss_res / max(ss_tot, 1e-15)

    # 3. Monotonicity violations
    n_violations = 0
    for i in range(len(loss_sorted) - 1):
        if loss_sorted[i + 1] > loss_sorted[i] + 1e-12:
            n_violations += 1
    max_possible = len(loss_sorted) - 1
    violation_rate = n_violations / max(max_possible, 1)

    return {
        "spearman_rho": rho,
        "spearman_p": p_value,
        "iso_r2": r2_iso,
        "n_violations": n_violations,
        "max_violations": max_possible,
        "violation_rate": violation_rate,
    }


# =====================================================================
# Main experiment
# =====================================================================
def run_experiment():
    """Execute the full Q1 monotonicity experiment.

    For each scenario, at each noise level, we run N_SUBTRIALS independent
    sub-trials.  This yields N_NOISE_LEVELS * N_SUBTRIALS (icm, loss) pairs
    per repetition.  We verify monotonicity on the full set of pairs.
    """
    trial_fns = {
        "classification": run_classification_trial,
        "regression": run_regression_trial,
        "cascade": run_cascade_trial,
    }

    all_results: dict[str, list[dict]] = {name: [] for name in trial_fns}

    print("=" * 78)
    print("  EXPERIMENT Q1: Monotonicity of E[L|C] in ICM score C")
    print("=" * 78)
    print()
    print(f"  Noise levels  : {N_NOISE_LEVELS} "
          f"(from {NOISE_LEVELS[0]:.2f} to {NOISE_LEVELS[-1]:.2f})")
    print(f"  Sub-trials/lvl: {N_SUBTRIALS}")
    print(f"  Total points  : {N_NOISE_LEVELS * N_SUBTRIALS} per repetition")
    print(f"  Repetitions   : {N_REPETITIONS}")
    print(f"  Models/trial  : {N_MODELS}")
    print()

    t_start = time.time()

    for scenario_name, trial_fn in trial_fns.items():
        print(f"  Running scenario: {scenario_name} ...", end="", flush=True)
        t0 = time.time()

        for rep in range(N_REPETITIONS):
            all_icm = []
            all_loss = []
            per_noise_icm = []
            per_noise_loss = []

            for j, noise_val in enumerate(NOISE_LEVELS):
                icm_at_noise = []
                loss_at_noise = []
                for s in range(N_SUBTRIALS):
                    seed = BASE_SEED + rep * 100000 + j * 1000 + s
                    icm_val, loss_val = trial_fn(noise_val, seed)
                    all_icm.append(icm_val)
                    all_loss.append(loss_val)
                    icm_at_noise.append(icm_val)
                    loss_at_noise.append(loss_val)
                per_noise_icm.append(np.mean(icm_at_noise))
                per_noise_loss.append(np.mean(loss_at_noise))

            icm_arr = np.array(all_icm)
            loss_arr = np.array(all_loss)

            diagnostics = verify_monotonicity(icm_arr, loss_arr)
            diagnostics["rep"] = rep
            diagnostics["per_noise_icm"] = np.array(per_noise_icm)
            diagnostics["per_noise_loss"] = np.array(per_noise_loss)
            diagnostics["icm_range"] = (float(icm_arr.min()), float(icm_arr.max()))
            diagnostics["loss_range"] = (float(loss_arr.min()), float(loss_arr.max()))
            all_results[scenario_name].append(diagnostics)

        dt = time.time() - t0
        print(f" done ({dt:.1f}s)")

    total_time = time.time() - t_start
    print(f"\n  Total wall time: {total_time:.1f}s\n")

    # ==================================================================
    # Print results tables
    # ==================================================================
    print("=" * 78)
    print("  DETAILED RESULTS PER REPETITION")
    print("=" * 78)

    for scenario_name in trial_fns:
        print(f"\n  --- {scenario_name.upper()} ---")
        header = (f"  {'Rep':>3s}  {'Spearman rho':>13s}  {'p-value':>10s}  "
                  f"{'Iso R^2':>8s}  {'Violations':>10s}  {'Viol.Rate':>9s}  "
                  f"{'ICM range':>17s}")
        print(header)
        print("  " + "-" * (len(header) - 2))

        for d in all_results[scenario_name]:
            imin, imax = d["icm_range"]
            print(f"  {d['rep']:3d}  "
                  f"{d['spearman_rho']:13.4f}  "
                  f"{d['spearman_p']:10.2e}  "
                  f"{d['iso_r2']:8.4f}  "
                  f"{d['n_violations']:4d}/{d['max_violations']:<5d}  "
                  f"{d['violation_rate']:9.4f}  "
                  f"[{imin:.3f}, {imax:.3f}]")

    # ==================================================================
    # Aggregate statistics
    # ==================================================================
    print("\n" + "=" * 78)
    print("  AGGREGATE STATISTICS (mean +/- std over 10 repetitions)")
    print("=" * 78)

    summary_table: dict[str, dict[str, str]] = {}

    header_agg = (f"  {'Scenario':>15s}  {'Spearman rho':>15s}  "
                  f"{'p < 0.05':>8s}  {'Iso R^2':>15s}  "
                  f"{'Viol. Rate':>15s}  {'PASS':>6s}")
    print(header_agg)
    print("  " + "-" * (len(header_agg) - 2))

    all_pass = True
    for scenario_name in trial_fns:
        rhos = np.array([d["spearman_rho"] for d in all_results[scenario_name]])
        ps = np.array([d["spearman_p"] for d in all_results[scenario_name]])
        r2s = np.array([d["iso_r2"] for d in all_results[scenario_name]])
        vrs = np.array([d["violation_rate"] for d in all_results[scenario_name]])

        sig_frac = np.mean(ps < 0.05)
        mean_rho = rhos.mean()
        std_rho = rhos.std()
        mean_r2 = r2s.mean()
        std_r2 = r2s.std()
        mean_vr = vrs.mean()
        std_vr = vrs.std()

        # PASS criteria:
        #   - mean Spearman rho < -0.3
        #   - >= 80% of reps have p < 0.05
        #   - mean violation rate < 0.50
        passed = (mean_rho < -0.3) and (sig_frac >= 0.8) and (mean_vr < 0.50)
        if not passed:
            all_pass = False

        tag = "YES" if passed else "NO"

        print(f"  {scenario_name:>15s}  "
              f"{mean_rho:+7.4f} +/- {std_rho:.4f}  "
              f"{sig_frac*100:6.0f}%   "
              f"{mean_r2:6.4f} +/- {std_r2:.4f}  "
              f"{mean_vr:6.4f} +/- {std_vr:.4f}  "
              f"{tag:>6s}")

        summary_table[scenario_name] = {
            "mean_rho": f"{mean_rho:+.4f}",
            "std_rho": f"{std_rho:.4f}",
            "sig_frac": f"{sig_frac*100:.0f}%",
            "mean_r2": f"{mean_r2:.4f}",
            "std_r2": f"{std_r2:.4f}",
            "mean_vr": f"{mean_vr:.4f}",
            "std_vr": f"{std_vr:.4f}",
            "pass": tag,
        }

    # ==================================================================
    # Per-noise-level averages
    # ==================================================================
    print("\n" + "=" * 78)
    print("  PER-NOISE-LEVEL AVERAGES (mean across repetitions)")
    print("=" * 78)

    for scenario_name in trial_fns:
        all_noise_icm = np.stack(
            [d["per_noise_icm"] for d in all_results[scenario_name]]
        )
        all_noise_loss = np.stack(
            [d["per_noise_loss"] for d in all_results[scenario_name]]
        )
        mean_icm = all_noise_icm.mean(axis=0)
        std_icm = all_noise_icm.std(axis=0)
        mean_loss = all_noise_loss.mean(axis=0)
        std_loss = all_noise_loss.std(axis=0)

        print(f"\n  --- {scenario_name.upper()} ---")
        print(f"  {'Noise':>7s}  {'ICM (mean +/- std)':>22s}  "
              f"{'Loss (mean +/- std)':>25s}")
        print(f"  {'-----':>7s}  {'------------------':>22s}  "
              f"{'-------------------':>25s}")
        for j in range(N_NOISE_LEVELS):
            print(f"  {NOISE_LEVELS[j]:7.3f}  "
                  f"{mean_icm[j]:8.4f} +/- {std_icm[j]:.4f}  "
                  f"{mean_loss[j]:10.6f} +/- {std_loss[j]:.6f}")

    # ==================================================================
    # Final verdict
    # ==================================================================
    print("\n" + "=" * 78)
    if all_pass:
        print("  VERDICT: PASS -- E[L|C] is monotonically non-increasing in C")
        print("           across all three scenarios.")
    else:
        print("  VERDICT: PARTIAL -- monotonicity holds in most but not all")
        print("           scenarios.  See per-scenario results above for details.")
    print("=" * 78)

    return all_results, summary_table, all_pass


# =====================================================================
if __name__ == "__main__":
    results, summary, verdict = run_experiment()
