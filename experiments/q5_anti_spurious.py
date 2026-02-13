"""Experiment Q5: Anti-Spurious Convergence Detection Validation.

Validates that the anti-spurious convergence detection system can
distinguish genuine convergence from spurious agreement arising from
shared biases or overfitting.

Three scenarios are tested:
  1. GENUINE        - Independent models converging on a real signal.
  2. SPURIOUS_SHARED_BIAS - Models agree due to a shared hidden factor.
  3. SPURIOUS_OVERFIT     - Models memorize training data; fail on held-out.

Sensitivity analysis varies n_permutations and sample sizes.
Robustness is assessed over 10 independent repetitions.

Usage:
    python experiments/q5_anti_spurious.py
"""

from __future__ import annotations

import sys
import os
import time

# ---------------------------------------------------------------------------
# Ensure the project root is on sys.path so that `framework` / `benchmarks`
# can be imported regardless of the working directory.
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
from sklearn.model_selection import train_test_split

from framework.config import AntiSpuriousConfig, ICMConfig
from framework.types import AntiSpuriousReport
from framework.anti_spurious import (
    generate_anti_spurious_report,
    ablation_analysis,
    hsic_test,
)
from framework.icm import compute_icm_from_predictions
from benchmarks.synthetic.generators import (
    generate_classification_benchmark,
    generate_multi_model_predictions,
)


# ===================================================================
# Helper utilities
# ===================================================================

def print_header(title: str, char: str = "=") -> None:
    """Print a section header."""
    width = 78
    print()
    print(char * width)
    print(f"  {title}")
    print(char * width)


def print_subheader(title: str) -> None:
    """Print a subsection header."""
    print(f"\n--- {title} ---")


def print_report(name: str, report: AntiSpuriousReport) -> None:
    """Pretty-print an AntiSpuriousReport."""
    print(f"  Scenario          : {name}")
    print(f"  D0 baseline       : {report.d0_baseline:.4f}")
    print(f"  C_normalized      : {report.c_normalized:.4f}")
    print(f"  HSIC p-value      : {report.hsic_pvalue:.4f}")
    print(f"  is_genuine        : {report.is_genuine}")
    print(f"  FDR corrected     : {report.fdr_corrected}")
    if report.ablation_results:
        print(f"  Ablation results  :")
        for model, delta in report.ablation_results.items():
            print(f"    {model:20s} -> delta_ICM = {delta:+.4f}")


def print_table(headers: list[str], rows: list[list[str]],
                col_widths: list[int] | None = None) -> None:
    """Print a simple text table."""
    if col_widths is None:
        col_widths = []
        for i, h in enumerate(headers):
            max_w = len(h)
            for row in rows:
                if i < len(row):
                    max_w = max(max_w, len(row[i]))
            col_widths.append(max_w + 2)

    # Header
    header_line = "".join(h.ljust(w) for h, w in zip(headers, col_widths))
    print(header_line)
    print("-" * len(header_line))

    # Rows
    for row in rows:
        line = "".join(
            (row[i] if i < len(row) else "").ljust(col_widths[i])
            for i in range(len(headers))
        )
        print(line)


# ===================================================================
# Scenario generators
# ===================================================================

def make_genuine_scenario(
    n_samples: int = 500,
    seed: int = 42,
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """GENUINE: Independent models converging on the same real signal.

    Simulates 4 models that each recover the true labels well but
    whose errors are completely independent.

    Construction:
      - labels = sigmoid(X @ true_coef)
      - Each model: pred_i = labels + eps_i, where eps_i ~ N(0, sigma)
        independently for each model.

    This guarantees:
      - High agreement (all predictions cluster near the true labels).
      - High normalized convergence (D_observed << D_0).
      - Independent residuals: HSIC should NOT reject the null.

    Returns (predictions_dict, labels, features).
    """
    rng = np.random.default_rng(seed)

    # Generate a continuous regression signal in [0, 1]
    d = 10
    X = rng.standard_normal((n_samples, d))
    true_coef = rng.standard_normal(d) * 0.5
    from scipy.special import expit
    labels = expit(X @ true_coef)

    predictions_dict: dict[str, np.ndarray] = {}
    n_models = 4
    noise_scale = 0.08  # moderate noise relative to signal range [0, 1]

    for i in range(n_models):
        # Each model = true signal + INDEPENDENT Gaussian noise
        eps = rng.normal(0, noise_scale, n_samples)
        pred = np.clip(labels + eps, 0.0, 1.0)
        predictions_dict[f"indep_model_{i}"] = pred

    return predictions_dict, labels, X


def make_spurious_shared_bias_scenario(
    n_samples: int = 500,
    seed: int = 42,
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """SPURIOUS_SHARED_BIAS: Models agree because of a shared hidden factor.

    All models share a latent confounder Z that is NOT part of the
    true data-generating process.  Each model's prediction becomes:
        pred_i = label + beta_i * Z + tiny_independent_noise

    Because every model's residual contains a component proportional
    to Z, the pairwise residual correlations are strong, and the HSIC
    test should reject independence (p < 0.05).

    The independent noise is kept very small relative to the shared
    Z component so the dependence signal dominates.

    Returns (predictions_dict, labels, features).
    """
    rng = np.random.default_rng(seed)

    d = 10
    X = rng.standard_normal((n_samples, d))
    true_coef = rng.standard_normal(d) * 0.5
    from scipy.special import expit
    labels = expit(X @ true_coef)

    # Hidden confounding variable -- shared by ALL models, NOT in X or labels
    Z = rng.standard_normal(n_samples)

    predictions_dict: dict[str, np.ndarray] = {}
    n_models = 4
    for i in range(n_models):
        # Each model approximates the true labels well ...
        base = labels.copy()
        # ... but ALL models are biased by the SAME hidden confounder Z
        # Use a strong, consistent shared bias
        beta = 0.3 + rng.uniform(0.0, 0.1)  # similar betas across models
        shared_component = beta * Z
        # Tiny independent noise (much smaller than shared component)
        indep_noise = rng.normal(0, 0.02, n_samples)
        pred = base + shared_component + indep_noise
        pred = np.clip(pred, 0.0, 1.0)
        predictions_dict[f"biased_model_{i}"] = pred

    return predictions_dict, labels, X


def make_spurious_overfit_scenario(
    n_samples: int = 500,
    seed: int = 42,
) -> tuple[
    dict[str, np.ndarray], np.ndarray, np.ndarray,
    dict[str, np.ndarray], np.ndarray, np.ndarray,
]:
    """SPURIOUS_OVERFIT: Models memorize training data.

    On the TRAINING set: all models output near-perfect predictions
    (simulating memorisation) with independent noise.
    On a HELD-OUT test set: the same models produce diverse random
    noise (their memorised patterns do not transfer), resulting in
    low normalized convergence C_norm.

    Returns (train_preds, train_labels, train_X,
             test_preds, test_labels, test_X).
    """
    rng = np.random.default_rng(seed)

    from scipy.special import expit

    d = 10
    X = rng.standard_normal((n_samples, d))
    true_coef = rng.standard_normal(d) * 0.5
    labels_all = expit(X @ true_coef)

    # Split into train / test
    n_train = int(n_samples * 0.6)
    idx = rng.permutation(n_samples)
    train_idx, test_idx = idx[:n_train], idx[n_train:]

    X_train, X_test = X[train_idx], X[test_idx]
    labels_train, labels_test = labels_all[train_idx], labels_all[test_idx]

    n_models = 4

    # --- TRAINING predictions: near-perfect (memorised) with independent noise ---
    train_preds: dict[str, np.ndarray] = {}
    for i in range(n_models):
        noise = rng.normal(0, 0.03, len(labels_train))
        pred = np.clip(labels_train + noise, 0.0, 1.0)
        train_preds[f"overfit_model_{i}"] = pred

    # --- TEST predictions: each model outputs random garbage ---
    # Models disagree with each other AND with truth on held-out data.
    test_preds: dict[str, np.ndarray] = {}
    for i in range(n_models):
        # Random predictions spread across [0, 1], different per model
        pred = rng.uniform(0.1, 0.9, len(labels_test))
        test_preds[f"overfit_model_{i}"] = pred

    return train_preds, labels_train, X_train, test_preds, labels_test, X_test


# ===================================================================
# ICM helper for ablation_analysis
# ===================================================================

def make_icm_fn(labels: np.ndarray) -> callable:
    """Return a callable that computes ICM from a predictions dict.

    The callable signature matches what ablation_analysis expects:
        icm_fn(predictions_dict) -> float
    """
    config = ICMConfig()

    def _icm_fn(predictions_dict: dict[str, np.ndarray]) -> float:
        if len(predictions_dict) < 2:
            return 0.5  # degenerate
        result = compute_icm_from_predictions(
            predictions_dict, config=config, distance_fn="wasserstein",
        )
        return result.icm_score

    return _icm_fn


# ===================================================================
# Core experiment runners
# ===================================================================

def run_single_scenario(
    name: str,
    predictions_dict: dict[str, np.ndarray],
    labels: np.ndarray,
    features: np.ndarray,
    config: AntiSpuriousConfig,
) -> AntiSpuriousReport:
    """Run generate_anti_spurious_report on one scenario."""
    icm_fn = make_icm_fn(labels)
    report = generate_anti_spurious_report(
        predictions_dict=predictions_dict,
        labels=labels,
        features=features,
        config=config,
        icm_fn=icm_fn,
    )
    return report


def run_three_scenarios(
    n_samples: int = 500,
    n_permutations: int = 200,
    seed: int = 42,
    verbose: bool = True,
) -> dict[str, dict]:
    """Run all three scenarios and return structured results."""
    config = AntiSpuriousConfig(
        n_permutations=n_permutations,
        fdr_level=0.05,
        n_negative_controls=30,  # keep reasonable for speed
    )

    results: dict[str, dict] = {}

    # ---- 1. GENUINE ----
    preds_g, labels_g, X_g = make_genuine_scenario(n_samples=n_samples, seed=seed)
    report_g = run_single_scenario("GENUINE", preds_g, labels_g, X_g, config)
    ablation_g = ablation_analysis(preds_g, make_icm_fn(labels_g), config)

    results["GENUINE"] = {
        "report": report_g,
        "ablation": ablation_g,
        "is_genuine": report_g.is_genuine,
        "c_normalized": report_g.c_normalized,
        "hsic_pvalue": report_g.hsic_pvalue,
    }

    if verbose:
        print_subheader("GENUINE scenario")
        print_report("GENUINE", report_g)

    # ---- 2. SPURIOUS_SHARED_BIAS ----
    preds_sb, labels_sb, X_sb = make_spurious_shared_bias_scenario(
        n_samples=n_samples, seed=seed,
    )
    report_sb = run_single_scenario(
        "SPURIOUS_SHARED_BIAS", preds_sb, labels_sb, X_sb, config,
    )
    ablation_sb = ablation_analysis(preds_sb, make_icm_fn(labels_sb), config)

    results["SPURIOUS_SHARED_BIAS"] = {
        "report": report_sb,
        "ablation": ablation_sb,
        "is_genuine": report_sb.is_genuine,
        "c_normalized": report_sb.c_normalized,
        "hsic_pvalue": report_sb.hsic_pvalue,
    }

    if verbose:
        print_subheader("SPURIOUS_SHARED_BIAS scenario")
        print_report("SPURIOUS_SHARED_BIAS", report_sb)

    # ---- 3. SPURIOUS_OVERFIT ----
    (preds_train, labels_train, X_train,
     preds_test, labels_test, X_test) = make_spurious_overfit_scenario(
        n_samples=n_samples, seed=seed,
    )

    # Run on TRAINING data (should look "genuine" at first glance)
    report_of_train = run_single_scenario(
        "SPURIOUS_OVERFIT (train)", preds_train, labels_train, X_train, config,
    )

    # Run on TEST data (should be flagged as NOT genuine)
    report_of_test = run_single_scenario(
        "SPURIOUS_OVERFIT (test)", preds_test, labels_test, X_test, config,
    )

    ablation_of = ablation_analysis(preds_test, make_icm_fn(labels_test), config)

    results["SPURIOUS_OVERFIT"] = {
        "report_train": report_of_train,
        "report_test": report_of_test,
        "ablation": ablation_of,
        "is_genuine_train": report_of_train.is_genuine,
        "is_genuine_test": report_of_test.is_genuine,
        "c_normalized_train": report_of_train.c_normalized,
        "c_normalized_test": report_of_test.c_normalized,
        "hsic_pvalue_train": report_of_train.hsic_pvalue,
        "hsic_pvalue_test": report_of_test.hsic_pvalue,
    }

    if verbose:
        print_subheader("SPURIOUS_OVERFIT scenario (training data)")
        print_report("OVERFIT-TRAIN", report_of_train)
        print_subheader("SPURIOUS_OVERFIT scenario (held-out test data)")
        print_report("OVERFIT-TEST", report_of_test)

    return results


# ===================================================================
# Sensitivity analysis
# ===================================================================

def run_sensitivity_analysis(
    n_permutations_list: list[int],
    sample_sizes: list[int],
    seed: int = 42,
) -> list[dict]:
    """Vary n_permutations and sample sizes, record detection outcomes.

    Note: HSIC computational cost scales as O(n^2 * n_permutations).
    We cap n_negative_controls at 30 during sensitivity sweeps for speed.
    """
    rows: list[dict] = []
    total = len(n_permutations_list) * len(sample_sizes)
    idx = 0

    for n_perm in n_permutations_list:
        for n_samp in sample_sizes:
            idx += 1
            print(f"  Sensitivity config {idx}/{total}: "
                  f"n_perm={n_perm}, n_samp={n_samp} ...", end=" ",
                  flush=True)
            t0 = time.time()
            res = run_three_scenarios(
                n_samples=n_samp,
                n_permutations=n_perm,
                seed=seed,
                verbose=False,
            )
            dt = time.time() - t0
            print(f"({dt:.1f}s)")
            row = {
                "n_permutations": n_perm,
                "n_samples": n_samp,
                "genuine_correct": res["GENUINE"]["is_genuine"],
                "genuine_cnorm": res["GENUINE"]["c_normalized"],
                "genuine_hsic_p": res["GENUINE"]["hsic_pvalue"],
                "bias_correct": not res["SPURIOUS_SHARED_BIAS"]["is_genuine"],
                "bias_cnorm": res["SPURIOUS_SHARED_BIAS"]["c_normalized"],
                "bias_hsic_p": res["SPURIOUS_SHARED_BIAS"]["hsic_pvalue"],
                "overfit_correct": not res["SPURIOUS_OVERFIT"]["is_genuine_test"],
                "overfit_cnorm_train": res["SPURIOUS_OVERFIT"]["c_normalized_train"],
                "overfit_cnorm_test": res["SPURIOUS_OVERFIT"]["c_normalized_test"],
                "overfit_hsic_p_test": res["SPURIOUS_OVERFIT"]["hsic_pvalue_test"],
            }
            rows.append(row)
    return rows


# ===================================================================
# Robustness analysis (10 repetitions)
# ===================================================================

def run_robustness_analysis(
    n_repetitions: int = 10,
    n_samples: int = 300,
    n_permutations: int = 200,
) -> dict[str, list[bool]]:
    """Run the full pipeline n_repetitions times with different seeds."""
    detection: dict[str, list[bool]] = {
        "genuine_correct": [],
        "bias_correct": [],
        "overfit_correct": [],
    }

    for rep in range(n_repetitions):
        seed = 1000 + rep * 7  # spread seeds
        print(f"  Repetition {rep+1}/{n_repetitions} (seed={seed}) ...",
              end=" ", flush=True)
        t0 = time.time()
        res = run_three_scenarios(
            n_samples=n_samples,
            n_permutations=n_permutations,
            seed=seed,
            verbose=False,
        )
        dt = time.time() - t0
        g = res["GENUINE"]["is_genuine"]
        b = not res["SPURIOUS_SHARED_BIAS"]["is_genuine"]
        o = not res["SPURIOUS_OVERFIT"]["is_genuine_test"]
        detection["genuine_correct"].append(g)
        detection["bias_correct"].append(b)
        detection["overfit_correct"].append(o)
        print(f"G={'P' if g else 'F'} B={'P' if b else 'F'} "
              f"O={'P' if o else 'F'} ({dt:.1f}s)")

    return detection


# ===================================================================
# Main experiment
# ===================================================================

def main() -> dict:
    """Entry point: run all parts of experiment Q5."""
    wall_start = time.time()

    print_header("EXPERIMENT Q5: Anti-Spurious Convergence Detection Validation")

    # Fix global numpy seed for top-level reproducibility
    np.random.seed(42)

    # ------------------------------------------------------------------
    # Part 1: Primary three-scenario test
    # ------------------------------------------------------------------
    print_header("Part 1: Three-Scenario Primary Test", char="-")
    results = run_three_scenarios(
        n_samples=500, n_permutations=200, seed=42, verbose=True,
    )

    # ------------------------------------------------------------------
    # Part 1b: Verification summary
    # ------------------------------------------------------------------
    print_header("Part 1b: Verification Summary", char="-")

    checks = []

    # GENUINE checks
    g = results["GENUINE"]
    chk_genuine_flag = g["is_genuine"] is True
    chk_genuine_hsic = g["hsic_pvalue"] > 0.05
    chk_genuine_cnorm = g["c_normalized"] > 0.5
    checks.append(("GENUINE is_genuine=True", chk_genuine_flag))
    checks.append(("GENUINE HSIC p > 0.05", chk_genuine_hsic))
    checks.append(("GENUINE C_norm > 0.5", chk_genuine_cnorm))

    # SPURIOUS_SHARED_BIAS checks
    sb = results["SPURIOUS_SHARED_BIAS"]
    chk_bias_flag = sb["is_genuine"] is False
    chk_bias_hsic = sb["hsic_pvalue"] < 0.05
    checks.append(("BIAS is_genuine=False", chk_bias_flag))
    checks.append(("BIAS HSIC p < 0.05", chk_bias_hsic))

    # SPURIOUS_OVERFIT checks
    of = results["SPURIOUS_OVERFIT"]
    chk_overfit_flag = of["is_genuine_test"] is False
    checks.append(("OVERFIT is_genuine=False (test)", chk_overfit_flag))

    print()
    print_table(
        ["Check", "Expected", "Actual", "PASS?"],
        [
            [desc, "True", str(val), "PASS" if val else "FAIL"]
            for desc, val in checks
        ],
        col_widths=[38, 12, 12, 8],
    )

    all_passed = all(v for _, v in checks)
    print(f"\nOverall: {'ALL CHECKS PASSED' if all_passed else 'SOME CHECKS FAILED'}")

    # ------------------------------------------------------------------
    # Part 2: Ablation analysis
    # ------------------------------------------------------------------
    print_header("Part 2: Ablation Analysis", char="-")

    for scenario_name in ["GENUINE", "SPURIOUS_SHARED_BIAS", "SPURIOUS_OVERFIT"]:
        print_subheader(f"Ablation: {scenario_name}")
        ablation = results[scenario_name]["ablation"]
        abl_rows = [
            [model, f"{delta:+.4f}"]
            for model, delta in ablation.items()
        ]
        print_table(["Model", "delta_ICM"], abl_rows, col_widths=[24, 14])

    # ------------------------------------------------------------------
    # Part 3: Sensitivity analysis
    # ------------------------------------------------------------------
    print_header("Part 3: Sensitivity Analysis", char="-")
    print("Varying n_permutations in {100, 500, 1000}")
    print("Varying n_samples      in {100, 500, 2000}")
    print("(using reduced negative controls for speed)")
    print()

    sens_rows = run_sensitivity_analysis(
        n_permutations_list=[100, 500, 1000],
        sample_sizes=[100, 500, 2000],
        seed=42,
    )

    sens_table_rows = []
    for r in sens_rows:
        sens_table_rows.append([
            str(r["n_permutations"]),
            str(r["n_samples"]),
            "PASS" if r["genuine_correct"] else "FAIL",
            f"{r['genuine_cnorm']:.3f}",
            f"{r['genuine_hsic_p']:.3f}",
            "PASS" if r["bias_correct"] else "FAIL",
            f"{r['bias_hsic_p']:.3f}",
            "PASS" if r["overfit_correct"] else "FAIL",
            f"{r['overfit_cnorm_test']:.3f}",
        ])

    print_table(
        ["nPerm", "nSamp", "Gen", "G_Cnorm", "G_HSIC_p",
         "Bias", "B_HSIC_p", "Ofit", "O_Cnorm"],
        sens_table_rows,
        col_widths=[8, 8, 6, 9, 10, 6, 10, 6, 9],
    )

    # ------------------------------------------------------------------
    # Part 4: Robustness (10 repetitions)
    # ------------------------------------------------------------------
    print_header("Part 4: Robustness Analysis (10 repetitions)", char="-")

    rob = run_robustness_analysis(
        n_repetitions=10, n_samples=300, n_permutations=200,
    )

    for scenario, detections in rob.items():
        rate = sum(detections) / len(detections)
        print(f"  {scenario:25s}: {sum(detections)}/{len(detections)} "
              f"= {rate:.0%} detection rate")

    # ------------------------------------------------------------------
    # Timing
    # ------------------------------------------------------------------
    wall_end = time.time()
    elapsed = wall_end - wall_start
    print(f"\nTotal wall-clock time: {elapsed:.1f} s")

    # ------------------------------------------------------------------
    # Collect everything for the report generator
    # ------------------------------------------------------------------
    return {
        "primary": results,
        "checks": checks,
        "all_passed": all_passed,
        "sensitivity": sens_rows,
        "robustness": rob,
        "elapsed": elapsed,
    }


if __name__ == "__main__":
    experiment_data = main()
