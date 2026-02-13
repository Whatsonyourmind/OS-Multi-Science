"""
Compute-ICM Demo  --  OS Multi-Science
=======================================
A self-contained walkthrough that generates synthetic data, computes
every ICM component step by step, tracks convergence over time, and
prints a summary DecisionCard.

Run with:
    python -m notebooks.compute_icm_demo
    (from the repository root)

Dependencies: numpy, scipy, sklearn  (all standard)
"""

import sys
import numpy as np

# Ensure the repo root is on the path
sys.path.insert(0, ".")

# ---- framework imports -------------------------------------------------
from framework.config import ICMConfig, CRCConfig, EarlyWarningConfig
from framework.types import (
    ICMComponents,
    ICMResult,
    DecisionCard,
    DecisionAction,
)
from framework.icm import (
    hellinger_distance,
    compute_agreement,
    compute_direction,
    compute_uncertainty_overlap,
    compute_invariance,
    compute_dependency_penalty,
    compute_icm,
    compute_icm_from_predictions,
    compute_icm_timeseries,
)
from framework.early_warning import (
    compute_delta_icm,
    compute_prediction_variance,
    compute_z_signal,
    cusum_detector,
)
from framework.crc_gating import decision_gate

# ---- benchmark generators ----------------------------------------------
from benchmarks.synthetic.generators import (
    generate_classification_benchmark,
    generate_multi_model_predictions,
    generate_change_point_series,
)


def section(title: str) -> None:
    """Print a visible section header."""
    width = 64
    print()
    print("=" * width)
    print(f"  {title}")
    print("=" * width)


# =====================================================================
# PART 1 -- Synthetic classification data
# =====================================================================
section("1. Generate synthetic classification data")

X, y_true, centres = generate_classification_benchmark(
    n_samples=600, n_classes=3, noise=0.15, seed=42,
)
print(f"  Samples : {X.shape[0]}")
print(f"  Features: {X.shape[1]}")
print(f"  Classes : {len(centres)}")
print(f"  Class centres:\n{centres}")


# =====================================================================
# PART 2 -- Create model predictions (3 agreeing, 1 disagreeing)
# =====================================================================
section("2. Create multi-model predictions")

predictions = generate_multi_model_predictions(
    X, y_true,
    n_agreeing=3, n_disagreeing=1,
    noise_agree=0.05, noise_disagree=0.5,
    seed=42,
)

for name, preds in predictions.items():
    acc = (preds.argmax(axis=1) == y_true).mean()
    print(f"  {name:15s}  accuracy = {acc:.3f}   shape = {preds.shape}")


# =====================================================================
# PART 3 -- Compute ICM components step by step
# =====================================================================
section("3. ICM components (step by step)")

config = ICMConfig()

# For Hellinger we need per-sample class distributions, so we take the
# average distribution from each model as a summary.
mean_dists = [preds.mean(axis=0) for preds in predictions.values()]

# A: Distributional agreement
A = compute_agreement(mean_dists, distance_fn="hellinger", config=config)
print(f"  A (agreement)        = {A:.4f}")

# D: Direction agreement (sign of mean prediction - 1/K baseline)
overall_mean = np.mean([d.mean() for d in mean_dists])
signs = np.array([d.mean() - overall_mean for d in mean_dists])
D = compute_direction(signs)
print(f"  D (direction)        = {D:.4f}")

# U: Uncertainty overlap (use 10-90 percentile of max-prob per model)
intervals = []
for preds in predictions.values():
    max_probs = preds.max(axis=1)
    lo = float(np.percentile(max_probs, 10))
    hi = float(np.percentile(max_probs, 90))
    intervals.append((lo, hi))
    print(f"    interval: [{lo:.3f}, {hi:.3f}]")

U = compute_uncertainty_overlap(intervals)
print(f"  U (uncertainty)      = {U:.4f}")

# C: Invariance (simulate a small perturbation)
pre = np.array([preds.mean() for preds in predictions.values()])
# Perturb each model's mean slightly
rng = np.random.default_rng(99)
post = pre + rng.normal(0, 0.01, size=pre.shape)
C_val = compute_invariance(pre, post)
print(f"  C (invariance)       = {C_val:.4f}")

# Pi: Dependency penalty (residuals + feature overlap)
residuals = np.stack([
    (preds.argmax(axis=1) - y_true).astype(float)
    for preds in predictions.values()
])
features = [
    {"x0", "x1"},              # agree_0
    {"x0", "x1"},              # agree_1
    {"x0", "x1"},              # agree_2
    {"x0", "x1", "noise_col"}, # disagree_0 uses an extra junk feature
]
Pi = compute_dependency_penalty(
    residuals=residuals, features=features, config=config,
)
print(f"  Pi (dependency)      = {Pi:.4f}")


# =====================================================================
# PART 4 -- Aggregate into ICM score
# =====================================================================
section("4. Full ICM score")

components = ICMComponents(A=A, D=D, U=U, C=C_val, Pi=Pi)
result = compute_icm(components, config)
print(f"  ICM (logistic)       = {result.icm_score:.4f}")
print(f"  High convergence?    = {result.is_high_convergence}")
print(f"  Weights used         = {result.weights}")

# Also try the one-call convenience function
result_auto = compute_icm_from_predictions(
    predictions,
    config=config,
    distance_fn="hellinger",
    intervals=intervals,
)
print(f"  ICM (auto pipeline)  = {result_auto.icm_score:.4f}")
print(f"  Models               = {result_auto.n_models}")


# =====================================================================
# PART 5 -- Effect of removing the disagreeing model
# =====================================================================
section("5. ICM with / without disagreeing model")

preds_agree_only = {k: v for k, v in predictions.items() if k.startswith("agree")}
preds_all = predictions

result_all = compute_icm_from_predictions(preds_all, config=config)
result_agree = compute_icm_from_predictions(preds_agree_only, config=config)

print(f"  4 models (3 agree + 1 disagree): ICM = {result_all.icm_score:.4f}")
print(f"  3 models (agree only):           ICM = {result_agree.icm_score:.4f}")
print(f"  Removing the dissenter raised ICM by {result_agree.icm_score - result_all.icm_score:+.4f}")


# =====================================================================
# PART 6 -- Change-point data + ICM timeseries
# =====================================================================
section("6. Change-point ICM timeseries")

# Simulate 4 "models" tracking a signal.  Before the change-point they
# agree; afterwards, one model lags and another drifts, creating
# divergence that ICM can detect.
T_total = 200
cp_idx = 100
rng_ts = np.random.default_rng(7)

base_signal = np.zeros(T_total)
base_signal[cp_idx:] = np.linspace(0, 3.0, T_total - cp_idx)

ts_preds: list[dict[str, np.ndarray]] = []
for t in range(T_total):
    noise = 0.02
    # Model A: tracks the signal well
    m_a = np.array([base_signal[t] + rng_ts.normal(0, noise)])
    # Model B: tracks with small delay
    lagged_t = max(t - 5, 0)
    m_b = np.array([base_signal[lagged_t] + rng_ts.normal(0, noise)])
    # Model C: tracks but with growing bias after change
    bias = 0.0 if t < cp_idx else 0.5 * (t - cp_idx) / (T_total - cp_idx)
    m_c = np.array([base_signal[t] + bias + rng_ts.normal(0, noise)])
    # Model D: stops tracking after change (stays flat)
    if t < cp_idx:
        m_d = np.array([base_signal[t] + rng_ts.normal(0, noise)])
    else:
        m_d = np.array([base_signal[cp_idx] + rng_ts.normal(0, noise)])
    ts_preds.append({"tracker": m_a, "lagged": m_b, "biased": m_c, "frozen": m_d})

print(f"  Time steps           = {T_total}")
print(f"  True change point    = {cp_idx}")
print(f"  Models: tracker, lagged, biased, frozen")

icm_ts_config = ICMConfig(C_A_wasserstein=3.0)
icm_ts = compute_icm_timeseries(ts_preds, config=icm_ts_config, distance_fn="wasserstein")
icm_scores = np.array([r.icm_score for r in icm_ts])

print(f"  ICM timeseries length = {len(icm_scores)}")
pre_mean = icm_scores[:cp_idx].mean()
post_mean = icm_scores[cp_idx:].mean()
print(f"  ICM before change     = {pre_mean:.4f} (mean of first {cp_idx})")
print(f"  ICM after  change     = {post_mean:.4f} (mean of last {len(icm_scores)-cp_idx})")
print(f"  Delta                 = {post_mean - pre_mean:+.4f}")


# =====================================================================
# PART 7 -- Early-warning detection on the ICM timeseries
# =====================================================================
section("7. Early-warning detection")

ew_config = EarlyWarningConfig(window_size=10, cusum_threshold=1.5, cusum_drift=0.1)

delta = compute_delta_icm(icm_scores, window_size=ew_config.window_size)

# Prediction-variance proxy: variance across model outputs at each step
model_ts_dict: dict[str, list[float]] = {
    "tracker": [], "lagged": [], "biased": [], "frozen": [],
}
for step in ts_preds:
    for k in model_ts_dict:
        model_ts_dict[k].append(float(step[k].mean()))
model_ts_arr = {k: np.array(v) for k, v in model_ts_dict.items()}
var_pred = compute_prediction_variance(model_ts_arr)

# Dependency-penalty trend: flat placeholder (no dependency data per step)
pi_trend = np.zeros_like(icm_scores)

z_signal = compute_z_signal(delta, var_pred, pi_trend, ew_config)
change_pts, cusum_vals = cusum_detector(
    z_signal, ew_config.cusum_threshold, ew_config.cusum_drift,
)

print(f"  CUSUM change points detected at steps: {change_pts[:10]}")
if change_pts:
    first_det = change_pts[0]
    print(f"  Earliest detection   = step {first_det}")
    print(f"  True change point    = step {cp_idx}")
    if first_det <= cp_idx:
        print(f"  Lead time            = {cp_idx - first_det} steps before change")
    else:
        print(f"  Detection delay      = {first_det - cp_idx} steps after change")
else:
    print("  (no change points detected with current thresholds)")


# =====================================================================
# PART 8 -- Decision gating + summary DecisionCard
# =====================================================================
section("8. Decision Card")

crc_config = CRCConfig(tau_hi=0.60, tau_lo=0.40)
decision = decision_gate(result_auto.icm_score, re_score=0.0, config=crc_config)

card = DecisionCard(
    problem_snapshot=(
        "Classify 600 samples into 3 Gaussian clusters using 4 independent "
        "models (3 agreeing, 1 dissenting)."
    ),
    recommended_kit=[
        {"method": "Logistic Regression", "role": "baseline", "why": "linear boundary"},
        {"method": "Random Forest", "role": "structure", "why": "non-linear splits"},
        {"method": "KNN", "role": "structure", "why": "local decision surface"},
        {"method": "Neural Net", "role": "forecast", "why": "flexible capacity"},
    ],
    main_findings=[
        f"ICM score = {result_auto.icm_score:.3f}  ({result_auto.aggregation_method} aggregation)",
        f"A={result_auto.components.A:.3f}  D={result_auto.components.D:.3f}  "
        f"U={result_auto.components.U:.3f}  C={result_auto.components.C:.3f}  "
        f"Pi={result_auto.components.Pi:.3f}",
        f"Removing the dissenting model raises ICM by {result_agree.icm_score - result_all.icm_score:+.3f}",
        f"Decision gate: {decision.value.upper()}",
    ],
    risks_and_limitations=[
        "Synthetic data -- real-world noise patterns may differ.",
        "Disagreeing model may capture structure that agreeing models miss.",
        "Dependency penalty is incomplete (no gradient information supplied).",
    ],
    next_actions=[
        "Investigate disagreeing model's predictions for minority-class samples.",
        "Run anti-spurious tests to confirm convergence is genuine.",
        "Repeat with real data and domain-specific distance metrics.",
    ],
    icm_summary=result_auto,
)

print(f"  Problem   : {card.problem_snapshot}")
print()
print("  Recommended kit:")
for m in card.recommended_kit:
    print(f"    - {m['method']:25s}  role={m['role']:12s}  {m['why']}")
print()
print("  Main findings:")
for f in card.main_findings:
    print(f"    * {f}")
print()
print("  Risks:")
for r in card.risks_and_limitations:
    print(f"    ! {r}")
print()
print("  Next actions:")
for a in card.next_actions:
    print(f"    > {a}")

section("Demo complete")
print("  All outputs above were computed from scratch with no external data.")
print("  Modify this script or import framework.icm in your own code.")
