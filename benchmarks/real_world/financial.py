"""Financial Systemic Risk Benchmark for OS Multi-Science.

A complete, runnable benchmark that:
  1. Generates a synthetic but realistic interbank network (50-100 banks).
  2. Simulates credit spreads, market returns, and interbank exposures.
  3. Creates a stress scenario with a known crisis period (step 200/400).
  4. Runs 5 model simulators through the OS Multi-Science pipeline.
  5. Computes ICM over time, runs early warning, CRC gating, and decision.
  6. Records everything in a KnowledgeGraph.
  7. Prints comprehensive text-based results.

Usage
-----
    python benchmarks/real_world/financial.py
"""

from __future__ import annotations

import sys
import os
import time
import warnings
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure the project root is on sys.path so that framework/ imports work
# regardless of where the script is invoked from.
# ---------------------------------------------------------------------------
_PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
from numpy.typing import NDArray
from scipy import stats as sp_stats
from sklearn.ensemble import GradientBoostingRegressor

from framework.aesc_profiler import FINANCIAL_ENERGY_SYSTEM, profile_summary
from framework.catalog import get_catalog
from framework.config import (
    CRCConfig,
    EarlyWarningConfig,
    ICMConfig,
    OSMultiScienceConfig,
)
from framework.crc_gating import (
    compute_re,
    conformalize,
    decision_gate,
    fit_isotonic,
)
from framework.early_warning import (
    compute_delta_icm,
    compute_prediction_variance,
    compute_rolling_icm,
    compute_z_signal,
    cusum_detector,
    evaluate_early_warning,
    page_hinkley_detector,
)
from framework.icm import compute_icm_from_predictions
from framework.router import select_kit
from framework.types import (
    DecisionAction,
    ICMResult,
    SystemProfile,
)
from knowledge.graph import (
    EdgeType,
    KnowledgeEdge,
    KnowledgeGraph,
    KnowledgeNode,
    NodeType,
    generate_id,
)
from orchestrator.pipeline import Pipeline

warnings.filterwarnings("ignore")

# ============================================================
# Constants
# ============================================================

N_BANKS = 75                  # Number of banks in the interbank network
T_TOTAL = 400                 # Total number of time steps
T_CRISIS = 200                # Crisis onset step
SEED = 42                     # Global random seed
ICM_WINDOW = 10               # Window for rolling ICM computation
EW_WINDOW = 30                # Window for early-warning rolling statistics
CUSUM_H = 5.0                 # CUSUM detection threshold
CUSUM_K = 0.5                 # CUSUM drift parameter
PH_THRESHOLD = 10.0           # Page-Hinkley threshold
N_EVAL_WINDOWS = 60           # Number of evaluation windows for ICM time series


# ============================================================
# 1.  Synthetic Interbank Network Generator
# ============================================================

def generate_interbank_network(
    n_banks: int = N_BANKS,
    seed: int = SEED,
) -> dict[str, NDArray]:
    """Generate a synthetic but realistic interbank network.

    The network has core-periphery structure: a small set of 'core' banks
    are densely connected, while 'periphery' banks connect mainly to
    core banks.  Exposure sizes follow a log-normal distribution, and
    the matrix is normalised so that each bank's total lending does not
    exceed its capital.

    Returns
    -------
    dict with keys:
        adjacency : (n, n) binary adjacency matrix
        exposures : (n, n) weighted exposure matrix (row i lends to column j)
        capital   : (n,)   capital buffer per bank
        is_core   : (n,)   boolean flag for core banks
    """
    rng = np.random.default_rng(seed)

    n_core = max(5, n_banks // 10)
    is_core = np.zeros(n_banks, dtype=bool)
    core_idx = rng.choice(n_banks, size=n_core, replace=False)
    is_core[core_idx] = True

    # Build adjacency: core-core dense, core-periphery moderate, periphery sparse
    adj = np.zeros((n_banks, n_banks), dtype=float)
    for i in range(n_banks):
        for j in range(i + 1, n_banks):
            if is_core[i] and is_core[j]:
                p = 0.7
            elif is_core[i] or is_core[j]:
                p = 0.15
            else:
                p = 0.03
            if rng.random() < p:
                adj[i, j] = 1.0
                adj[j, i] = 1.0

    # Capital: core banks are larger
    capital = np.where(is_core, rng.lognormal(3.0, 0.5, n_banks),
                       rng.lognormal(1.5, 0.8, n_banks))

    # Exposures: log-normal weights, scaled by lender capital
    raw_weights = rng.lognormal(0.0, 1.0, (n_banks, n_banks))
    exposures = adj * raw_weights
    # Normalise so total lending per bank <= 80% of capital
    row_sums = exposures.sum(axis=1)
    scale = np.where(row_sums > 0, 0.8 * capital / row_sums, 0.0)
    exposures = exposures * scale[:, np.newaxis]

    return {
        "adjacency": adj,
        "exposures": exposures,
        "capital": capital,
        "is_core": is_core,
    }


# ============================================================
# 2.  Financial Time-Series Simulator
# ============================================================

def simulate_financial_data(
    n_banks: int = N_BANKS,
    T: int = T_TOTAL,
    crisis_at: int = T_CRISIS,
    seed: int = SEED,
) -> dict[str, NDArray]:
    """Simulate credit spreads, market returns, and a systemic stress indicator.

    Pre-crisis:   Low volatility, mean-reverting credit spreads, positive drift.
    Crisis onset: Volatility doubles, credit spreads widen with a jump,
                  correlations increase, and a cascade-like contagion signal
                  propagates through the network.

    Returns
    -------
    dict with keys:
        credit_spreads : (T, n_banks) basis-point credit spreads
        market_returns : (T,)         aggregate market return index
        stress_index   : (T,)         systemic stress indicator in [0, 1]
        true_labels    : (T,)         binary: 1 = crisis regime
    """
    rng = np.random.default_rng(seed)

    # --- Market returns (AR(1) with regime switch) ---
    market = np.zeros(T)
    market[0] = 0.0
    for t in range(1, T):
        if t < crisis_at:
            mu, sigma = 0.0005, 0.01
        else:
            # Crisis: negative drift and higher volatility
            decay = np.exp(-0.01 * (t - crisis_at))
            mu = -0.003 * decay
            sigma = 0.025 * (1.0 + 0.5 * decay)
        market[t] = 0.3 * market[t - 1] + mu + sigma * rng.standard_normal()

    # --- Credit spreads (mean-reverting with jumps at crisis) ---
    spreads = np.zeros((T, n_banks))
    base_spread = rng.uniform(50, 200, n_banks)  # bps
    spreads[0] = base_spread

    for t in range(1, T):
        noise = rng.standard_normal(n_banks)
        if t < crisis_at:
            # Mean-reversion to base
            reversion = 0.05 * (base_spread - spreads[t - 1])
            spreads[t] = spreads[t - 1] + reversion + 3.0 * noise
        else:
            # Crisis: spreads widen with a jump and slower reversion
            crisis_base = base_spread * (1.5 + 0.8 * np.exp(-0.005 * (t - crisis_at)))
            reversion = 0.02 * (crisis_base - spreads[t - 1])
            jump = rng.poisson(0.1, n_banks) * rng.exponential(20.0, n_banks)
            spreads[t] = spreads[t - 1] + reversion + 8.0 * noise + jump

    spreads = np.maximum(spreads, 5.0)  # floor at 5 bps

    # --- Systemic stress index (ground truth) ---
    stress = np.zeros(T)
    for t in range(T):
        if t < crisis_at:
            stress[t] = 0.05 + 0.03 * rng.standard_normal()
        else:
            phase = min(1.0, (t - crisis_at) / 50.0)
            stress[t] = 0.05 + 0.85 * phase * np.exp(-0.003 * max(0, t - crisis_at - 50))
            stress[t] += 0.05 * rng.standard_normal()
    stress = np.clip(stress, 0.0, 1.0)

    true_labels = (np.arange(T) >= crisis_at).astype(float)

    return {
        "credit_spreads": spreads,
        "market_returns": market,
        "stress_index": stress,
        "true_labels": true_labels,
    }


# ============================================================
# 3.  Model Simulators
# ============================================================

def var_model(
    spreads: NDArray,
    market: NDArray,
    n_lags: int = 5,
) -> NDArray:
    """Vector Autoregression proxy: predict aggregate stress from lagged features.

    Uses a simple least-squares lag regression on the mean credit spread
    and market returns.  Returns a (T,) prediction of systemic stress.
    """
    T = len(market)
    # Build feature matrix from lags
    mean_spread = spreads.mean(axis=1)
    # Standardise
    ms_std = (mean_spread - mean_spread.mean()) / (mean_spread.std() + 1e-12)
    mk_std = (market - market.mean()) / (market.std() + 1e-12)

    X_lags = []
    for lag in range(1, n_lags + 1):
        feat = np.zeros(T)
        feat[lag:] = ms_std[:-lag]
        X_lags.append(feat)
        feat2 = np.zeros(T)
        feat2[lag:] = mk_std[:-lag]
        X_lags.append(feat2)

    X = np.column_stack(X_lags)
    # Target: normalised mean spread (proxy for stress)
    y = ms_std.copy()

    # Fit on first 80%, predict on all
    split = int(0.8 * T)
    # Regularised least squares (ridge)
    lam = 1.0
    XtX = X[:split].T @ X[:split] + lam * np.eye(X.shape[1])
    Xty = X[:split].T @ y[:split]
    try:
        beta = np.linalg.solve(XtX, Xty)
    except np.linalg.LinAlgError:
        beta = np.zeros(X.shape[1])

    preds = X @ beta
    # Rescale to [0, 1]
    preds = (preds - preds.min()) / (preds.max() - preds.min() + 1e-12)
    return preds


def volatility_model(
    spreads: NDArray,
    market: NDArray,
    halflife: int = 20,
) -> NDArray:
    """EWMA volatility model as a GARCH proxy.

    Computes exponentially weighted moving average of squared returns
    and squared spread changes.  Higher realised vol => higher stress.
    """
    T = len(market)
    alpha = 1.0 - np.exp(-np.log(2.0) / halflife)

    # Market volatility (EWMA of squared returns)
    ret = np.diff(market, prepend=market[0])
    sq_ret = ret ** 2
    ewma_vol = np.zeros(T)
    ewma_vol[0] = sq_ret[0]
    for t in range(1, T):
        ewma_vol[t] = alpha * sq_ret[t] + (1 - alpha) * ewma_vol[t - 1]

    # Spread volatility (EWMA of squared spread changes, averaged across banks)
    d_spreads = np.diff(spreads, axis=0, prepend=spreads[:1])
    sq_d = d_spreads ** 2
    ewma_spread = np.zeros(T)
    ewma_spread[0] = sq_d[0].mean()
    for t in range(1, T):
        ewma_spread[t] = alpha * sq_d[t].mean() + (1 - alpha) * ewma_spread[t - 1]

    # Combine
    combined = 0.5 * ewma_vol + 0.5 * ewma_spread
    combined = (combined - combined.min()) / (combined.max() - combined.min() + 1e-12)
    return combined


def network_contagion_model(
    network: dict[str, NDArray],
    spreads: NDArray,
    threshold_pct: float = 75.0,
    n_cascade_rounds: int = 3,
) -> NDArray:
    """Threshold cascade on the interbank exposure network.

    At each time step, a bank is considered 'distressed' if its credit
    spread exceeds the threshold_pct percentile of all spreads at that
    time.  The model then runs n_cascade_rounds of contagion: a healthy
    bank becomes distressed if the exposure-weighted fraction of its
    distressed counterparties exceeds 30%.  The systemic stress
    prediction is the final fraction of distressed banks.
    """
    adj = network["adjacency"]
    exposures = network["exposures"]
    capital = network["capital"]
    T, n_banks = spreads.shape

    preds = np.zeros(T)
    for t in range(T):
        threshold = np.percentile(spreads[t], threshold_pct)
        distressed = (spreads[t] > threshold).astype(float)

        # Run cascade rounds
        for _ in range(n_cascade_rounds):
            loss_from_distressed = exposures @ distressed
            # A bank defaults if losses exceed 30% of capital
            new_distressed = (loss_from_distressed > 0.3 * capital).astype(float)
            distressed = np.maximum(distressed, new_distressed)

        # Weighted contagion score: combine fraction distressed with
        # exposure-weighted distress signal
        frac_distressed = distressed.mean()
        weighted_distress = exposures @ distressed
        total_exposure = exposures.sum(axis=1)
        safe_total = np.where(total_exposure > 0, total_exposure, 1.0)
        contagion = (weighted_distress / safe_total).mean()
        preds[t] = 0.6 * frac_distressed + 0.4 * contagion

    # Normalise to [0, 1]
    preds = (preds - preds.min()) / (preds.max() - preds.min() + 1e-12)
    return preds


def gradient_boosting_model(
    spreads: NDArray,
    market: NDArray,
    stress: NDArray,
    seed: int = SEED,
) -> NDArray:
    """Gradient boosting regressor (sklearn) as a nonlinear forecaster.

    Features: rolling statistics of spreads and market returns.
    Target: stress index (using first 60% for training).
    """
    T = len(market)
    rng = np.random.default_rng(seed)

    # Feature engineering
    mean_spread = spreads.mean(axis=1)
    std_spread = spreads.std(axis=1)
    max_spread = spreads.max(axis=1)

    features_list = [mean_spread, std_spread, max_spread, market]
    # Rolling means (window 10 and 30)
    for w in [10, 30]:
        rm = np.convolve(mean_spread, np.ones(w) / w, mode="same")
        features_list.append(rm)
        rm_mkt = np.convolve(market, np.ones(w) / w, mode="same")
        features_list.append(rm_mkt)
    # Rolling volatility
    for w in [10, 30]:
        rv = np.array([mean_spread[max(0, i - w):i + 1].std()
                       for i in range(T)])
        features_list.append(rv)

    X = np.column_stack(features_list)

    # Train/test split
    split = int(0.6 * T)
    X_train, y_train = X[:split], stress[:split]

    gbr = GradientBoostingRegressor(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        random_state=seed,
    )
    gbr.fit(X_train, y_train)

    preds = gbr.predict(X)
    preds = np.clip(preds, 0.0, 1.0)
    return preds


def naive_baseline(
    spreads: NDArray,
    window: int = 50,
) -> NDArray:
    """Naive baseline: rolling mean of normalised mean credit spread.

    A deliberately simple model to serve as a reference point.
    """
    mean_spread = spreads.mean(axis=1)
    normalised = (mean_spread - mean_spread.min()) / (mean_spread.max() - mean_spread.min() + 1e-12)
    T = len(normalised)
    preds = np.zeros(T)
    for t in range(T):
        start = max(0, t - window + 1)
        preds[t] = normalised[start:t + 1].mean()
    return preds


# ============================================================
# 4.  ICM Over Time Computation
# ============================================================

def compute_icm_over_windows(
    all_preds: dict[str, NDArray],
    config: ICMConfig,
    window_size: int = ICM_WINDOW,
) -> list[tuple[int, ICMResult]]:
    """Compute ICM at evenly spaced windows across the time series.

    Returns a list of (window_centre_time, ICMResult) pairs.
    """
    T = next(iter(all_preds.values())).shape[0]
    n_windows = min(N_EVAL_WINDOWS, T // window_size)
    step = max(1, (T - window_size) // n_windows)

    results: list[tuple[int, ICMResult]] = []
    for i in range(0, T - window_size + 1, step):
        window_preds: dict[str, NDArray] = {}
        for name, arr in all_preds.items():
            window_preds[name] = arr[i:i + window_size]

        icm = compute_icm_from_predictions(
            window_preds,
            config=config,
            distance_fn="wasserstein",
        )
        icm.n_models = len(all_preds)
        centre = i + window_size // 2
        results.append((centre, icm))

    return results


# ============================================================
# 5.  CRC Gating on ICM Series
# ============================================================

def run_crc_gating(
    icm_scores: NDArray,
    losses: NDArray,
    crc_config: CRCConfig,
) -> dict:
    """Run CRC gating using the ICM time series and corresponding losses.

    Returns
    -------
    dict with re_score, decision, coverage info
    """
    n = len(icm_scores)
    split = max(n // 2, 4)

    g = fit_isotonic(icm_scores[:split], losses[:split])
    g_alpha = conformalize(g, icm_scores[split:], losses[split:],
                           alpha=crc_config.alpha)

    # Use the median ICM of the crisis period for gating
    median_icm = float(np.median(icm_scores))
    re = compute_re(median_icm, g_alpha)
    decision = decision_gate(median_icm, re, crc_config)

    return {
        "re_score": re,
        "decision": decision,
        "median_icm": median_icm,
        "g_alpha": g_alpha,
    }


# ============================================================
# 6.  Knowledge Graph Recording
# ============================================================

def build_knowledge_graph(
    profile: SystemProfile,
    model_names: list[str],
    icm_windows: list[tuple[int, ICMResult]],
    ew_results: dict,
    crc_results: dict,
) -> KnowledgeGraph:
    """Record all benchmark results in a KnowledgeGraph."""
    kg = KnowledgeGraph()

    # System node
    sys_id = "system_financial_risk"
    kg.add_node(KnowledgeNode(
        id=sys_id,
        node_type=NodeType.SYSTEM,
        data={
            "name": profile.name,
            "n_banks": N_BANKS,
            "T_total": T_TOTAL,
            "T_crisis": T_CRISIS,
        },
    ))

    # Method nodes
    method_ids: dict[str, str] = {}
    for name in model_names:
        mid = f"method_{name.lower().replace(' ', '_')}"
        method_ids[name] = mid
        kg.add_node(KnowledgeNode(
            id=mid,
            node_type=NodeType.METHOD,
            data={"name": name},
        ))
        kg.add_edge(KnowledgeEdge(
            source_id=sys_id,
            target_id=mid,
            edge_type=EdgeType.ANALYZED_BY,
        ))

    # Result nodes for each model
    result_ids: dict[str, str] = {}
    for name in model_names:
        rid = f"result_{name.lower().replace(' ', '_')}"
        result_ids[name] = rid
        kg.add_node(KnowledgeNode(
            id=rid,
            node_type=NodeType.RESULT,
            data={"model": name, "type": "stress_prediction"},
        ))
        kg.add_edge(KnowledgeEdge(
            source_id=method_ids[name],
            target_id=rid,
            edge_type=EdgeType.PRODUCED,
        ))

    # ICM score nodes (one per window)
    for i, (t_centre, icm_res) in enumerate(icm_windows):
        icm_id = f"icm_t{t_centre}"
        kg.add_node(KnowledgeNode(
            id=icm_id,
            node_type=NodeType.ICM_SCORE,
            data={
                "icm_score": round(icm_res.icm_score, 4),
                "t_centre": t_centre,
                "A": round(icm_res.components.A, 4),
                "D": round(icm_res.components.D, 4),
                "U": round(icm_res.components.U, 4),
                "C": round(icm_res.components.C, 4),
                "Pi": round(icm_res.components.Pi, 4),
                "n_models": icm_res.n_models,
            },
        ))
        # Link ICM score nodes to all result nodes
        for name in model_names:
            kg.add_edge(KnowledgeEdge(
                source_id=result_ids[name],
                target_id=icm_id,
                edge_type=EdgeType.CONVERGES_WITH,
                weight=icm_res.icm_score,
            ))

    # Decision node
    decision_id = "decision_systemic_risk"
    kg.add_node(KnowledgeNode(
        id=decision_id,
        node_type=NodeType.DECISION,
        data={
            "action": crc_results["decision"].value,
            "re_score": round(crc_results["re_score"], 4),
            "median_icm": round(crc_results["median_icm"], 4),
            "cusum_detections": len(ew_results.get("cusum_changes", [])),
            "ph_detections": len(ew_results.get("ph_changes", [])),
        },
    ))

    # Link last ICM to decision
    if icm_windows:
        last_icm_id = f"icm_t{icm_windows[-1][0]}"
        kg.add_edge(KnowledgeEdge(
            source_id=last_icm_id,
            target_id=decision_id,
            edge_type=EdgeType.LED_TO,
        ))

    return kg


# ============================================================
# 7.  Reporting Utilities
# ============================================================

def print_header(title: str, width: int = 80) -> None:
    """Print a section header."""
    print()
    print("=" * width)
    print(f"  {title}")
    print("=" * width)


def print_table(headers: list[str], rows: list[list], col_widths: list[int] | None = None) -> None:
    """Print a formatted text table."""
    if col_widths is None:
        col_widths = [max(len(str(h)), max((len(str(r[i])) for r in rows), default=4)) + 2
                      for i, h in enumerate(headers)]

    # Header
    header_line = "".join(str(h).ljust(w) for h, w in zip(headers, col_widths))
    print(header_line)
    print("-" * sum(col_widths))

    # Rows
    for row in rows:
        line = "".join(str(c).ljust(w) for c, w in zip(row, col_widths))
        print(line)


def format_results_report(
    profile: SystemProfile,
    network_info: dict,
    model_preds: dict[str, NDArray],
    icm_windows: list[tuple[int, ICMResult]],
    ew_results: dict,
    crc_results: dict,
    kg: KnowledgeGraph,
    elapsed: float,
    data: dict,
) -> str:
    """Format and print the full benchmark results. Returns the report as a string."""
    lines: list[str] = []

    def p(text: str = "") -> None:
        lines.append(text)
        print(text)

    # ---- Title ----
    p()
    p("=" * 80)
    p("  OS MULTI-SCIENCE: FINANCIAL SYSTEMIC RISK BENCHMARK")
    p("=" * 80)
    p()

    # ---- Setup ----
    p("SECTION 1: BENCHMARK SETUP")
    p("-" * 40)
    p(f"  Network:       {N_BANKS} banks, core-periphery topology")
    n_edges = int(network_info["adjacency"].sum() / 2)
    p(f"  Edges:         {n_edges} bilateral interbank links")
    n_core = int(network_info["is_core"].sum())
    p(f"  Core banks:    {n_core}")
    p(f"  Time steps:    {T_TOTAL}")
    p(f"  Crisis onset:  t = {T_CRISIS}")
    p(f"  ICM window:    {ICM_WINDOW} steps")
    p(f"  Random seed:   {SEED}")
    p()
    p("  System Profile:")
    p(f"    {profile_summary(profile)}")
    p()

    # ---- Model Descriptions ----
    p("SECTION 2: MODEL DESCRIPTIONS")
    p("-" * 40)

    model_descriptions = {
        "VAR": (
            "Vector Autoregression (ridge-regularised). Uses lagged mean credit "
            "spreads and market returns (5 lags each). Linear model capturing "
            "temporal dependencies."
        ),
        "Volatility": (
            "EWMA Volatility (GARCH proxy). Exponentially weighted moving average "
            "of squared returns and spread changes (halflife=20). Captures "
            "time-varying volatility clustering."
        ),
        "Network Contagion": (
            "Threshold cascade on interbank exposure network. A bank is 'distressed' "
            "if its spread exceeds the 75th percentile. Stress = average weighted "
            "fraction of distressed counterparties."
        ),
        "Gradient Boosting": (
            "Sklearn GradientBoostingRegressor (100 trees, depth 4). Features: "
            "rolling means, rolling volatility, cross-sectional spread statistics. "
            "Trained on first 60% of data."
        ),
        "Naive Baseline": (
            "Rolling 50-step mean of normalised average credit spread. A deliberately "
            "simple reference model with no learning."
        ),
    }

    for name, desc in model_descriptions.items():
        p(f"  [{name}]")
        # Word-wrap description at 70 chars
        words = desc.split()
        line = "    "
        for word in words:
            if len(line) + len(word) + 1 > 74:
                p(line)
                line = "    " + word
            else:
                line += " " + word if len(line) > 4 else word
        if line.strip():
            p(line)
        p()

    # ---- Per-model accuracy ----
    p("SECTION 3: PER-MODEL ACCURACY")
    p("-" * 40)

    stress = data["stress_index"]
    headers = ["Model", "RMSE", "MAE", "Corr", "Pre-crisis RMSE", "Crisis RMSE"]
    rows = []
    for name, preds in model_preds.items():
        rmse = np.sqrt(np.mean((preds - stress) ** 2))
        mae = np.mean(np.abs(preds - stress))
        corr = float(np.corrcoef(preds, stress)[0, 1]) if np.std(preds) > 1e-12 else 0.0
        pre_rmse = np.sqrt(np.mean((preds[:T_CRISIS] - stress[:T_CRISIS]) ** 2))
        crisis_rmse = np.sqrt(np.mean((preds[T_CRISIS:] - stress[T_CRISIS:]) ** 2))
        rows.append([name, f"{rmse:.4f}", f"{mae:.4f}", f"{corr:.4f}",
                      f"{pre_rmse:.4f}", f"{crisis_rmse:.4f}"])

    col_widths = [22, 10, 10, 10, 18, 14]
    hdr_line = "".join(str(h).ljust(w) for h, w in zip(headers, col_widths))
    p(f"  {hdr_line}")
    p(f"  {'-' * sum(col_widths)}")
    for row in rows:
        line = "".join(str(c).ljust(w) for c, w in zip(row, col_widths))
        p(f"  {line}")
    p()

    # ---- ICM Over Time ----
    p("SECTION 4: ICM SCORES OVER TIME")
    p("-" * 40)

    icm_headers = ["Window", "t_centre", "ICM", "A", "D", "U", "C", "Pi", "Phase"]
    icm_rows = []
    for i, (t_c, icm) in enumerate(icm_windows):
        phase = "pre-crisis" if t_c < T_CRISIS else "crisis"
        icm_rows.append([
            f"W{i + 1:02d}",
            str(t_c),
            f"{icm.icm_score:.4f}",
            f"{icm.components.A:.4f}",
            f"{icm.components.D:.4f}",
            f"{icm.components.U:.4f}",
            f"{icm.components.C:.4f}",
            f"{icm.components.Pi:.4f}",
            phase,
        ])

    icm_col = [8, 10, 8, 8, 8, 8, 8, 8, 14]
    hdr = "".join(str(h).ljust(w) for h, w in zip(icm_headers, icm_col))
    p(f"  {hdr}")
    p(f"  {'-' * sum(icm_col)}")
    for row in icm_rows:
        line = "".join(str(c).ljust(w) for c, w in zip(row, icm_col))
        p(f"  {line}")

    # Summary statistics
    pre_icms = [icm.icm_score for t, icm in icm_windows if t < T_CRISIS]
    crisis_icms = [icm.icm_score for t, icm in icm_windows if t >= T_CRISIS]
    p()
    p(f"  Pre-crisis ICM:  mean={np.mean(pre_icms):.4f}, "
      f"std={np.std(pre_icms):.4f}" if pre_icms else "  Pre-crisis ICM:  N/A")
    p(f"  Crisis ICM:      mean={np.mean(crisis_icms):.4f}, "
      f"std={np.std(crisis_icms):.4f}" if crisis_icms else "  Crisis ICM:      N/A")
    if pre_icms and crisis_icms:
        icm_drop = np.mean(pre_icms) - np.mean(crisis_icms)
        p(f"  ICM drop at crisis: {icm_drop:+.4f}")
    p()

    # ---- Early Warning ----
    p("SECTION 5: EARLY WARNING DETECTION")
    p("-" * 40)
    p(f"  CUSUM threshold:       {CUSUM_H}")
    p(f"  CUSUM drift:           {CUSUM_K}")
    p(f"  Page-Hinkley threshold: {PH_THRESHOLD}")
    p()

    cusum_changes = ew_results.get("cusum_changes", [])
    ph_changes = ew_results.get("ph_changes", [])

    p(f"  CUSUM detections:       {len(cusum_changes)}")
    if cusum_changes:
        first_cusum = cusum_changes[0]
        lead_time = T_CRISIS - first_cusum
        p(f"    First detection:      t = {first_cusum}")
        p(f"    Lead time to crisis:  {lead_time} steps {'(early)' if lead_time > 0 else '(late)'}")

    p(f"  Page-Hinkley detections: {len(ph_changes)}")
    if ph_changes:
        first_ph = ph_changes[0]
        lead_time_ph = T_CRISIS - first_ph
        p(f"    First detection:       t = {first_ph}")
        p(f"    Lead time to crisis:   {lead_time_ph} steps {'(early)' if lead_time_ph > 0 else '(late)'}")

    p()
    cusum_eval = ew_results.get("cusum_eval", {})
    ph_eval = ew_results.get("ph_eval", {})
    if cusum_eval:
        p(f"  CUSUM evaluation:")
        p(f"    True positive rate:  {cusum_eval.get('true_positive_rate', 0):.4f}")
        p(f"    False positive rate: {cusum_eval.get('false_positive_rate', 0):.4f}")
        p(f"    AUROC proxy:         {cusum_eval.get('auroc', 0):.4f}")
        lead_times = cusum_eval.get("lead_times", [])
        if lead_times:
            p(f"    Lead times:          {lead_times}")

    if ph_eval:
        p(f"  Page-Hinkley evaluation:")
        p(f"    True positive rate:  {ph_eval.get('true_positive_rate', 0):.4f}")
        p(f"    False positive rate: {ph_eval.get('false_positive_rate', 0):.4f}")
        p(f"    AUROC proxy:         {ph_eval.get('auroc', 0):.4f}")
    p()

    # ---- CRC Gating ----
    p("SECTION 6: CRC GATING AND DECISION")
    p("-" * 40)
    p(f"  Median ICM:        {crc_results['median_icm']:.4f}")
    p(f"  Epistemic risk Re: {crc_results['re_score']:.4f}")
    p(f"  Decision:          {crc_results['decision'].value}")
    p()
    action = crc_results["decision"]
    if action == DecisionAction.ACT:
        p("  Interpretation: High model convergence -- the ensemble's systemic")
        p("  risk assessment can be used for decision-making with confidence.")
    elif action == DecisionAction.DEFER:
        p("  Interpretation: Moderate convergence -- the assessment should be")
        p("  reviewed by a human expert before acting.")
    else:
        p("  Interpretation: Low convergence -- models disagree significantly.")
        p("  A full audit of model assumptions and data is recommended.")
    p()

    # ---- Knowledge Graph ----
    p("SECTION 7: KNOWLEDGE GRAPH SUMMARY")
    p("-" * 40)
    summary = kg.summary()
    for k, v in summary.items():
        p(f"  {k:30s}: {v}")
    p()

    # ---- Timing ----
    p(f"Total benchmark runtime: {elapsed:.2f} seconds")
    p()

    # ---- Key Findings ----
    p("SECTION 8: KEY FINDINGS AND CONCLUSIONS")
    p("-" * 40)
    p()

    # Determine which model was best
    rmses = {name: np.sqrt(np.mean((preds - stress) ** 2))
             for name, preds in model_preds.items()}
    best_model = min(rmses, key=rmses.get)
    worst_model = max(rmses, key=rmses.get)

    p(f"  1. Best individual model by RMSE: {best_model} ({rmses[best_model]:.4f})")
    p(f"     Worst individual model by RMSE: {worst_model} ({rmses[worst_model]:.4f})")
    p()

    if pre_icms and crisis_icms:
        if np.mean(crisis_icms) < np.mean(pre_icms):
            p("  2. ICM DECREASES during the crisis period, indicating that models")
            p("     DIVERGE under stress. This is the expected behaviour for a")
            p("     genuine systemic risk event -- different epistemic lenses")
            p("     yield increasingly different assessments when the system")
            p("     enters an unprecedented regime.")
        else:
            p("  2. ICM INCREASES during the crisis period, indicating that models")
            p("     CONVERGE under stress. This suggests the crisis signal is")
            p("     strong enough that all methods agree on its presence.")
    p()

    if cusum_changes and cusum_changes[0] < T_CRISIS:
        p("  3. The CUSUM early-warning detector successfully fired BEFORE")
        p(f"     the crisis onset (lead time = {T_CRISIS - cusum_changes[0]} steps),")
        p("     demonstrating that ICM dynamics contain predictive information")
        p("     about impending regime changes.")
    elif cusum_changes:
        p("  3. The CUSUM detector fired AFTER the crisis onset, detecting the")
        p("     regime change reactively rather than predictively.")
    else:
        p("  3. The CUSUM detector did not fire, suggesting the ICM signal")
        p("     change was too gradual for the chosen threshold.")
    p()

    p(f"  4. CRC gating decision: {action.value.upper()}.")
    p("     The conformal risk control framework provides a calibrated")
    p("     epistemic risk bound that accounts for model disagreement.")
    p()

    p("  5. The knowledge graph captures the full provenance chain from")
    p(f"     the system profile through {len(model_preds)} models, {len(icm_windows)} ICM windows,")
    p(f"     and the final {action.value} decision, enabling post-hoc audit.")
    p()
    p("=" * 80)
    p("  BENCHMARK COMPLETE")
    p("=" * 80)
    p()

    return "\n".join(lines)


# ============================================================
# 8.  Main Benchmark Runner
# ============================================================

def run_benchmark() -> str:
    """Execute the full financial systemic risk benchmark."""
    t_start = time.monotonic()

    print("Initializing financial systemic risk benchmark...")

    # ----- Configuration -----
    config = OSMultiScienceConfig(
        icm=ICMConfig(
            C_A_wasserstein=2.0,  # scale for financial data
        ),
        crc=CRCConfig(
            alpha=0.10,
            tau_hi=0.65,
            tau_lo=0.35,
        ),
        early_warning=EarlyWarningConfig(
            window_size=EW_WINDOW,
            cusum_threshold=CUSUM_H,
            cusum_drift=CUSUM_K,
        ),
        random_seed=SEED,
        verbose=True,
    )

    # ----- System Profile -----
    profile = FINANCIAL_ENERGY_SYSTEM
    print(f"  Profile: {profile.name}")

    # ----- Router: Select Method Kit -----
    print("  Routing method selection...")
    catalog = get_catalog()
    selection = select_kit(profile, catalog=catalog, config=config.router)
    print(f"  Selected kit: {[m.name for m in selection.selected_methods]}")
    print(f"  Kit diversity: {selection.avg_diversity:.3f}")

    # ----- Generate Interbank Network -----
    print("  Generating interbank network...")
    network = generate_interbank_network(n_banks=N_BANKS, seed=SEED)
    n_edges = int(network["adjacency"].sum() / 2)
    print(f"  Network: {N_BANKS} banks, {n_edges} edges, "
          f"{int(network['is_core'].sum())} core")

    # ----- Simulate Financial Data -----
    print("  Simulating financial time series...")
    data = simulate_financial_data(
        n_banks=N_BANKS, T=T_TOTAL, crisis_at=T_CRISIS, seed=SEED
    )
    print(f"  Time steps: {T_TOTAL}, Crisis at: {T_CRISIS}")

    # ----- Run Models -----
    print("  Running 5 model simulators...")

    model_preds: dict[str, NDArray] = {}

    print("    [1/5] VAR model...")
    model_preds["VAR"] = var_model(data["credit_spreads"], data["market_returns"])

    print("    [2/5] Volatility model...")
    model_preds["Volatility"] = volatility_model(
        data["credit_spreads"], data["market_returns"]
    )

    print("    [3/5] Network contagion model...")
    model_preds["Network Contagion"] = network_contagion_model(
        network, data["credit_spreads"]
    )

    print("    [4/5] Gradient boosting model...")
    model_preds["Gradient Boosting"] = gradient_boosting_model(
        data["credit_spreads"], data["market_returns"],
        data["stress_index"], seed=SEED,
    )

    print("    [5/5] Naive baseline...")
    model_preds["Naive Baseline"] = naive_baseline(data["credit_spreads"])

    # ----- Pipeline Integration -----
    print("  Running Pipeline orchestration...")
    pipeline = Pipeline(config=config)
    for name, pred_fn in [
        ("VAR", lambda X, p: model_preds["VAR"]),
        ("Volatility", lambda X, p: model_preds["Volatility"]),
        ("Network Contagion", lambda X, p: model_preds["Network Contagion"]),
        ("Gradient Boosting", lambda X, p: model_preds["Gradient Boosting"]),
        ("Naive Baseline", lambda X, p: model_preds["Naive Baseline"]),
    ]:
        pipeline.register_model(name, pred_fn)

    pipeline_result = pipeline.run(
        system_profile=profile,
        X=data["credit_spreads"],
        y_true=data["stress_index"],
        features=data["credit_spreads"],
        skip_anti_spurious=True,  # skip for performance in benchmark
    )

    print(f"  Pipeline status: {'OK' if pipeline_result.is_success else 'FAILED'}")
    for step in pipeline_result.step_results:
        status_icon = "OK" if step.status == "success" else (
            "SKIP" if step.status == "skipped" else "FAIL")
        print(f"    [{status_icon}] {step.step_name:20s} ({step.duration_seconds:.3f}s)")

    # ----- ICM Over Time -----
    print("  Computing ICM over time windows...")
    icm_windows = compute_icm_over_windows(model_preds, config.icm)
    print(f"  Computed {len(icm_windows)} ICM windows")

    # ----- Early Warning -----
    print("  Running early warning detection...")
    icm_scores_arr = np.array([icm.icm_score for _, icm in icm_windows])
    icm_times = np.array([t for t, _ in icm_windows])

    # Build per-model predictions at ICM window centres for variance computation
    window_model_preds: dict[str, NDArray] = {}
    for name, arr in model_preds.items():
        window_model_preds[name] = np.array([arr[t] for t in icm_times])

    ew_config = config.early_warning
    ew_config.window_size = min(ew_config.window_size, len(icm_scores_arr) // 2)
    ew_config.window_size = max(ew_config.window_size, 3)

    delta_icm = compute_delta_icm(icm_scores_arr, ew_config.window_size)
    var_preds = compute_prediction_variance(window_model_preds)

    # Standardise each component using ONLY the pre-crisis (training)
    # period statistics.  This means normal fluctuations in the stable
    # period have z ~ 0, while crisis deviations produce large positive z.
    n_pre = sum(1 for t, _ in icm_windows if t < T_CRISIS)
    n_pre = max(n_pre, 3)

    def _standardise_on_training(x: NDArray, n_train: int) -> NDArray:
        mu = x[:n_train].mean()
        s = x[:n_train].std()
        if s < 1e-12:
            return x - mu
        return (x - mu) / s

    delta_icm_std = _standardise_on_training(delta_icm, n_pre)
    var_preds_std = _standardise_on_training(var_preds, n_pre)

    # Pi trend: extract from the ICM windows
    pi_trend = np.array([icm.components.Pi for _, icm in icm_windows])
    pi_trend_std = _standardise_on_training(pi_trend, n_pre)

    z_signal = compute_z_signal(delta_icm_std, var_preds_std, pi_trend_std, ew_config)

    cusum_changes, cusum_vals = cusum_detector(z_signal, CUSUM_H, CUSUM_K)
    ph_changes, ph_vals = page_hinkley_detector(z_signal, PH_THRESHOLD)

    # Map detection times back to original timeline
    cusum_real = [int(icm_times[c]) for c in cusum_changes if c < len(icm_times)]
    ph_real = [int(icm_times[c]) for c in ph_changes if c < len(icm_times)]

    # Evaluate against known crisis point
    cusum_eval = evaluate_early_warning(
        cusum_real, [T_CRISIS], max_lead_time=50
    )
    ph_eval = evaluate_early_warning(
        ph_real, [T_CRISIS], max_lead_time=50
    )

    ew_results = {
        "cusum_changes": cusum_real,
        "ph_changes": ph_real,
        "cusum_eval": cusum_eval,
        "ph_eval": ph_eval,
        "z_signal": z_signal,
    }

    # ----- CRC Gating -----
    print("  Running CRC gating...")
    # Build losses at ICM window centres
    losses_at_windows = np.array([
        (model_preds["Gradient Boosting"][t] - data["stress_index"][t]) ** 2
        for t in icm_times
    ])
    # Average across models for a more robust loss
    ensemble_mean = np.mean(
        np.column_stack([model_preds[n][icm_times] for n in model_preds]),
        axis=1,
    )
    losses_at_windows = (ensemble_mean - data["stress_index"][icm_times]) ** 2

    crc_results = run_crc_gating(icm_scores_arr, losses_at_windows, config.crc)

    # ----- Knowledge Graph -----
    print("  Building knowledge graph...")
    kg = build_knowledge_graph(
        profile, list(model_preds.keys()), icm_windows,
        ew_results, crc_results,
    )

    elapsed = time.monotonic() - t_start
    print(f"  Benchmark computation complete in {elapsed:.2f}s")
    print()

    # ----- Results Report -----
    report = format_results_report(
        profile, network, model_preds, icm_windows,
        ew_results, crc_results, kg, elapsed, data,
    )

    return report


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":
    report_text = run_benchmark()

    # Save raw output to reports/ directory
    reports_dir = Path(_PROJECT_ROOT) / "reports"
    reports_dir.mkdir(exist_ok=True)
    raw_output_path = reports_dir / "benchmark_financial_raw_output.txt"
    raw_output_path.write_text(report_text, encoding="utf-8")
    print(f"Raw output saved to: {raw_output_path}")

    report_path = reports_dir / "benchmark_financial.md"
    if report_path.exists():
        print(f"Markdown report at: {report_path}")
    else:
        # Write a basic markdown report if one doesn't exist
        md_lines: list[str] = []
        md_lines.append("# Financial Systemic Risk Benchmark Report")
        md_lines.append("")
        md_lines.append("Generated by `benchmarks/real_world/financial.py`")
        md_lines.append("")
        md_lines.append("```")
        md_lines.append(report_text)
        md_lines.append("```")
        md_lines.append("")
        report_path.write_text("\n".join(md_lines), encoding="utf-8")
        print(f"Report saved to: {report_path}")
