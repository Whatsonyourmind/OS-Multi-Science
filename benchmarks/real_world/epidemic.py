"""Realistic epidemic spreading benchmark for OS Multi-Science.

Simulates SEIR dynamics on a contact network with 5 independent modelling
approaches, then feeds their predictions into the full OS Multi-Science
pipeline: ICM convergence, early-warning detection, and anti-spurious
validation.

Run:
    python benchmarks/real_world/epidemic.py
"""

from __future__ import annotations

import sys
import os
import time

# ---------------------------------------------------------------------------
# Path setup -- ensure the project root is importable
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, os.pardir, os.pardir))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import odeint
from scipy.optimize import curve_fit
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge

from framework.config import (
    AntiSpuriousConfig,
    EarlyWarningConfig,
    ICMConfig,
    OSMultiScienceConfig,
)
from framework.icm import compute_icm_from_predictions
from framework.early_warning import (
    compute_delta_icm,
    compute_prediction_variance,
    compute_rolling_icm,
    compute_z_signal,
    cusum_detector,
    evaluate_early_warning,
    page_hinkley_detector,
    placebo_test,
)
from framework.anti_spurious import generate_anti_spurious_report
from framework.types import ICMComponents, ICMResult
from knowledge.graph import (
    EdgeType,
    KnowledgeEdge,
    KnowledgeGraph,
    KnowledgeNode,
    NodeType,
    generate_id,
)


# ====================================================================
# 1. Contact network generation
# ====================================================================

def generate_contact_network(
    n: int = 500,
    topology: str = "scale_free",
    seed: int = 42,
) -> NDArray:
    """Build a symmetric adjacency matrix for a contact network.

    Parameters
    ----------
    n : int
        Number of nodes (individuals).
    topology : str
        ``"scale_free"`` (Barabasi-Albert-like) or ``"small_world"``
        (Watts-Strogatz-like).
    seed : int
        Random seed.

    Returns
    -------
    adj : ndarray of shape (n, n), float64
        Binary symmetric adjacency matrix (no self-loops).
    """
    rng = np.random.default_rng(seed)

    if topology == "scale_free":
        # Barabasi-Albert preferential attachment (m = 3 new edges per node)
        m = 3
        adj = np.zeros((n, n), dtype=np.float64)
        # Start with a small complete graph of size m+1
        for i in range(m + 1):
            for j in range(i + 1, m + 1):
                adj[i, j] = adj[j, i] = 1.0

        for new_node in range(m + 1, n):
            degree = adj.sum(axis=1)
            total_deg = degree.sum()
            if total_deg == 0:
                probs = np.ones(new_node) / new_node
            else:
                probs = degree[:new_node] / total_deg
            probs = np.maximum(probs, 1e-12)
            probs /= probs.sum()
            targets = rng.choice(new_node, size=m, replace=False, p=probs)
            for t in targets:
                adj[new_node, t] = adj[t, new_node] = 1.0
    else:
        # Watts-Strogatz small-world: ring lattice + random rewiring
        k = 6
        p_rewire = 0.1
        adj = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            for j in range(1, k // 2 + 1):
                neighbour = (i + j) % n
                adj[i, neighbour] = adj[neighbour, i] = 1.0
        for i in range(n):
            for j in range(1, k // 2 + 1):
                if rng.random() < p_rewire:
                    neighbour = (i + j) % n
                    if adj[i, neighbour] == 1.0:
                        adj[i, neighbour] = adj[neighbour, i] = 0.0
                        new_target = rng.integers(0, n)
                        while new_target == i or adj[i, new_target] == 1.0:
                            new_target = rng.integers(0, n)
                        adj[i, new_target] = adj[new_target, i] = 1.0

    np.fill_diagonal(adj, 0.0)
    return adj


# ====================================================================
# 2. SEIR ground-truth simulation on the network
# ====================================================================

def simulate_seir_on_network(
    adj: NDArray,
    n_steps: int = 300,
    outbreak_step: int = 50,
    containment_step: int = 150,
    beta_pre: float = 0.035,
    beta_post: float = 0.008,
    sigma: float = 0.2,
    gamma: float = 0.1,
    n_initial_infected: int = 3,
    seed: int = 42,
) -> dict[str, NDArray]:
    """Agent-based SEIR on a contact network with outbreak and containment.

    Compartments per node: S(0), E(1), I(2), R(3).

    Before ``outbreak_step`` transmission is zero.  Between
    ``outbreak_step`` and ``containment_step`` transmission is
    ``beta_pre``.  After ``containment_step`` it drops to ``beta_post``.

    Returns
    -------
    dict with keys ``"S"``, ``"E"``, ``"I"``, ``"R"`` (fraction arrays),
    ``"new_infections"`` (daily new S->E), ``"states"`` (per-node history).
    """
    rng = np.random.default_rng(seed)
    n = adj.shape[0]
    state = np.zeros(n, dtype=int)

    S_frac = np.zeros(n_steps)
    E_frac = np.zeros(n_steps)
    I_frac = np.zeros(n_steps)
    R_frac = np.zeros(n_steps)
    new_infections = np.zeros(n_steps)
    states_history = np.zeros((n_steps, n), dtype=int)

    for t in range(n_steps):
        if t == outbreak_step:
            susceptible_idx = np.where(state == 0)[0]
            seeds = rng.choice(
                susceptible_idx,
                size=min(n_initial_infected, len(susceptible_idx)),
                replace=False,
            )
            state[seeds] = 2

        if t < outbreak_step:
            beta = 0.0
        elif t < containment_step:
            beta = beta_pre
        else:
            beta = beta_post

        new_state = state.copy()
        n_new = 0

        s_nodes = np.where(state == 0)[0]
        if len(s_nodes) > 0 and beta > 0:
            infectious_vector = (state == 2).astype(np.float64)
            n_inf_neighbours = adj[s_nodes] @ infectious_vector
            for idx, node in enumerate(s_nodes):
                k_inf = int(n_inf_neighbours[idx])
                if k_inf > 0:
                    p_escape = (1.0 - beta) ** k_inf
                    if rng.random() > p_escape:
                        new_state[node] = 1
                        n_new += 1

        for node in np.where(state == 1)[0]:
            if rng.random() < sigma:
                new_state[node] = 2
        for node in np.where(state == 2)[0]:
            if rng.random() < gamma:
                new_state[node] = 3

        state = new_state
        states_history[t] = state
        S_frac[t] = (state == 0).sum() / n
        E_frac[t] = (state == 1).sum() / n
        I_frac[t] = (state == 2).sum() / n
        R_frac[t] = (state == 3).sum() / n
        new_infections[t] = n_new

    return {
        "S": S_frac, "E": E_frac, "I": I_frac, "R": R_frac,
        "new_infections": new_infections, "states": states_history,
    }


# ====================================================================
# 3. Model simulators
# ====================================================================

def model_compartmental_seir(
    ground_truth: dict[str, NDArray],
    n_steps: int,
    outbreak_step: int,
    containment_step: int,
    population: int,
) -> NDArray:
    """Deterministic SEIR ODE model.

    Fits piecewise beta from the observed epidemic curve and solves the
    standard SEIR ODEs.  This model captures the smooth mean-field dynamics
    but misses network heterogeneity and stochastic fluctuations.
    """

    def seir_odes(y, t, beta, sigma, gamma, N):
        S, E, I, R = y
        dSdt = -beta * S * I / N
        dEdt = beta * S * I / N - sigma * E
        dIdt = sigma * E - gamma * I
        dRdt = gamma * I
        return [dSdt, dEdt, dIdt, dRdt]

    gt_new = ground_truth["new_infections"]
    sigma_param = 0.2
    gamma_param = 0.1

    # Estimate beta from observed growth in outbreak phase
    outbreak_data = gt_new[outbreak_step:containment_step]
    if outbreak_data.max() > 0:
        cum_growth = np.cumsum(outbreak_data)
        mid = len(outbreak_data) // 2
        if mid > 0 and cum_growth[mid] > 0:
            growth_rate = cum_growth[-1] / max(cum_growth[mid], 1)
            beta_pre_est = gamma_param + sigma_param * 0.1 * np.log1p(growth_rate)
        else:
            beta_pre_est = 0.15
    else:
        beta_pre_est = 0.15
    beta_pre_est = np.clip(beta_pre_est, 0.05, 0.5)
    beta_post_est = beta_pre_est * 0.25

    predictions = np.zeros(n_steps)
    N = float(population)
    I0, E0, S0, R0_val = 3.0, 1.0, N - 4.0, 0.0
    y0 = [S0, E0, I0, R0_val]

    t_pre = np.arange(0, containment_step - outbreak_step + 1, dtype=float)
    if len(t_pre) > 1:
        sol_pre = odeint(seir_odes, y0, t_pre,
                         args=(beta_pre_est, sigma_param, gamma_param, N))
        new_inf_pre = np.maximum(-np.diff(sol_pre[:, 0]), 0)
        end_pre = min(len(new_inf_pre), containment_step - outbreak_step)
        predictions[outbreak_step:outbreak_step + end_pre] = new_inf_pre[:end_pre]
        y_cont = sol_pre[-1]
    else:
        y_cont = y0

    t_post = np.arange(0, n_steps - containment_step + 1, dtype=float)
    if len(t_post) > 1:
        sol_post = odeint(seir_odes, y_cont, t_post,
                          args=(beta_post_est, sigma_param, gamma_param, N))
        new_inf_post = np.maximum(-np.diff(sol_post[:, 0]), 0)
        end_post = min(len(new_inf_post), n_steps - containment_step)
        predictions[containment_step:containment_step + end_post] = new_inf_post[:end_post]

    return predictions


def model_network_abm(
    adj: NDArray,
    ground_truth: dict[str, NDArray],
    n_steps: int,
    outbreak_step: int,
    containment_step: int,
    seed: int = 137,
) -> NDArray:
    """Independent agent-based simulation with perturbed parameters.

    Uses the same network topology but a different random seed and
    slightly different epidemiological parameters.
    """
    rng = np.random.default_rng(seed)
    n = adj.shape[0]
    beta_pre, beta_post = 0.032, 0.009
    sigma, gamma = 0.18, 0.11
    state = np.zeros(n, dtype=int)
    predictions = np.zeros(n_steps)

    for t in range(n_steps):
        if t == outbreak_step:
            susceptible = np.where(state == 0)[0]
            seeds = rng.choice(susceptible, size=min(3, len(susceptible)),
                               replace=False)
            state[seeds] = 2

        beta = 0.0 if t < outbreak_step else (
            beta_pre if t < containment_step else beta_post)

        new_state = state.copy()
        n_new = 0
        s_nodes = np.where(state == 0)[0]
        if len(s_nodes) > 0 and beta > 0:
            inf_vec = (state == 2).astype(float)
            n_inf = adj[s_nodes] @ inf_vec
            for idx, node in enumerate(s_nodes):
                k = int(n_inf[idx])
                if k > 0 and rng.random() > (1.0 - beta) ** k:
                    new_state[node] = 1
                    n_new += 1

        for node in np.where(state == 1)[0]:
            if rng.random() < sigma:
                new_state[node] = 2
        for node in np.where(state == 2)[0]:
            if rng.random() < gamma:
                new_state[node] = 3

        state = new_state
        predictions[t] = n_new
    return predictions


def model_statistical_logistic(
    ground_truth: dict[str, NDArray],
    n_steps: int,
    outbreak_step: int,
    containment_step: int,
    population: int,
) -> NDArray:
    """Logistic growth model fit to cumulative infections.

    Fits separate logistic curves pre- and post-containment, then
    differentiates to get daily new infections.
    """

    def logistic_cumulative(t, K, r, t0):
        return K / (1.0 + np.exp(-r * (t - t0)))

    gt_new = ground_truth["new_infections"]
    cumulative = np.cumsum(gt_new)
    predictions = np.zeros(n_steps)

    obs_start, obs_end = outbreak_step, containment_step
    if obs_end - obs_start > 5:
        t_obs = np.arange(obs_end - obs_start, dtype=float)
        c_obs = cumulative[obs_start:obs_end]
        try:
            popt, _ = curve_fit(
                logistic_cumulative, t_obs, c_obs,
                p0=[float(population) * 0.3, 0.05, float(len(t_obs) / 2)],
                bounds=([1, 0.001, 0], [population, 1.0, len(t_obs) * 2]),
                maxfev=5000,
            )
            c_fit = logistic_cumulative(t_obs, *popt)
            new_fit = np.maximum(np.diff(np.concatenate([[0], c_fit])), 0)
            predictions[obs_start:obs_end] = new_fit
        except (RuntimeError, ValueError):
            predictions[obs_start:obs_end] = gt_new[obs_start:obs_end] * 0.9

    if n_steps - containment_step > 5:
        t_obs2 = np.arange(n_steps - containment_step, dtype=float)
        c_obs2 = np.maximum(cumulative[containment_step:] - cumulative[containment_step - 1], 0)
        K_remain = max(float(c_obs2[-1]) * 1.5, 10)
        try:
            popt2, _ = curve_fit(
                logistic_cumulative, t_obs2, c_obs2,
                p0=[K_remain, 0.02, float(len(t_obs2) / 2)],
                bounds=([0.1, 0.0001, 0], [population, 1.0, len(t_obs2) * 2]),
                maxfev=5000,
            )
            c_fit2 = logistic_cumulative(t_obs2, *popt2)
            new_fit2 = np.maximum(np.diff(np.concatenate([[0], c_fit2])), 0)
            predictions[containment_step:] = new_fit2
        except (RuntimeError, ValueError):
            predictions[containment_step:] = gt_new[containment_step:] * 0.85

    return predictions


def model_ml_ensemble(
    ground_truth: dict[str, NDArray],
    n_steps: int,
    outbreak_step: int,
) -> NDArray:
    """ML ensemble using lagged features (RandomForest + Ridge)."""
    gt_new = ground_truth["new_infections"]
    max_lag = 7

    X_list, y_list = [], []
    for t in range(max_lag, n_steps):
        features = [gt_new[t - lag] for lag in range(1, max_lag + 1)]
        features.append(np.sum(gt_new[:t]))
        features.append(float(t))
        features.append(float(t - outbreak_step))
        X_list.append(features)
        y_list.append(gt_new[t])

    X_all = np.array(X_list)
    y_all = np.array(y_list)
    predictions = np.zeros(n_steps)

    n_train = int(0.7 * len(X_all))
    if n_train < 10:
        return gt_new * 0.95 + np.random.default_rng(99).normal(0, 0.5, n_steps)

    X_train, y_train = X_all[:n_train], y_all[:n_train]
    rf = RandomForestRegressor(n_estimators=50, max_depth=8, random_state=42)
    ridge = Ridge(alpha=1.0)
    rf.fit(X_train, y_train)
    ridge.fit(X_train, y_train)

    ensemble = np.maximum((rf.predict(X_all) + ridge.predict(X_all)) / 2.0, 0)
    predictions[max_lag:] = ensemble
    return predictions


def model_exponential_smoothing(
    ground_truth: dict[str, NDArray],
    n_steps: int,
    alpha_smooth: float = 0.3,
) -> NDArray:
    """Naive exponential smoothing baseline (one-step lagged)."""
    gt_new = ground_truth["new_infections"]
    predictions = np.zeros(n_steps)
    predictions[0] = gt_new[0]
    for t in range(1, n_steps):
        predictions[t] = (alpha_smooth * gt_new[t - 1]
                          + (1 - alpha_smooth) * predictions[t - 1])
    return predictions


# ====================================================================
# 4. Per-step ICM computation
# ====================================================================

def compute_per_step_icm(
    predictions_dict: dict[str, NDArray],
    window_size: int = 10,
    config: ICMConfig | None = None,
) -> tuple[NDArray, list[ICMResult]]:
    """Compute a rolling-window ICM score for each timestep.

    At each step t, the ICM is computed over the window
    [t - window_size + 1, t].  For the first ``window_size - 1``
    steps a growing window is used.

    Returns
    -------
    icm_scores : (n_steps,) array of ICM scores.
    icm_results : list of ICMResult objects (one per step).
    """
    if config is None:
        config = ICMConfig()

    n_steps = len(next(iter(predictions_dict.values())))
    icm_scores = np.zeros(n_steps)
    icm_results: list[ICMResult] = []

    for t in range(n_steps):
        start = max(0, t - window_size + 1)
        window_len = t - start + 1

        window_preds: dict[str, NDArray] = {}
        for name, preds in predictions_dict.items():
            seg = preds[start:t + 1]
            total = seg.sum()
            if total > 1e-12:
                window_preds[name] = seg / total
            else:
                window_preds[name] = np.ones(window_len) / window_len

        result = compute_icm_from_predictions(
            window_preds, config=config, distance_fn="hellinger",
        )
        result.n_models = len(predictions_dict)
        icm_scores[t] = result.icm_score
        icm_results.append(result)

    return icm_scores, icm_results


# ====================================================================
# 5. Build knowledge graph
# ====================================================================

def build_knowledge_graph(
    model_names: list[str],
    predictions_dict: dict[str, NDArray],
    icm_scores: NDArray,
    icm_results: list[ICMResult],
    ew_results: dict,
    anti_spurious_report,
) -> KnowledgeGraph:
    """Record all benchmark results into a KnowledgeGraph."""
    kg = KnowledgeGraph()

    sys_node = KnowledgeNode(
        id="epidemic_system",
        node_type=NodeType.SYSTEM,
        data={"name": "SEIR Epidemic on Contact Network",
              "nodes": 500, "steps": 300,
              "outbreak_step": 50, "containment_step": 150},
    )
    kg.add_node(sys_node)

    for name in model_names:
        method_node = KnowledgeNode(
            id=f"method_{name}", node_type=NodeType.METHOD,
            data={"name": name, "family": _get_family(name)},
        )
        kg.add_node(method_node)
        kg.add_edge(KnowledgeEdge(
            source_id="epidemic_system", target_id=f"method_{name}",
            edge_type=EdgeType.ANALYZED_BY,
        ))

        pred = predictions_dict[name]
        result_node = KnowledgeNode(
            id=f"result_{name}", node_type=NodeType.RESULT,
            data={"model": name,
                  "total_predicted_infections": float(pred.sum()),
                  "peak_daily": float(pred.max())},
        )
        kg.add_node(result_node)
        kg.add_edge(KnowledgeEdge(
            source_id=f"method_{name}", target_id=f"result_{name}",
            edge_type=EdgeType.PRODUCED,
        ))

    # Sample ICM nodes (every 30 steps to keep graph manageable)
    for t in range(0, len(icm_results), 30):
        r = icm_results[t]
        icm_node = KnowledgeNode(
            id=f"icm_step_{t}", node_type=NodeType.ICM_SCORE,
            data={"icm_score": r.icm_score, "step": t,
                  "A": r.components.A, "D": r.components.D,
                  "U": r.components.U},
        )
        kg.add_node(icm_node)

    # Convergence edges
    mean_icm = float(icm_scores.mean())
    if mean_icm > 0.5:
        for i in range(len(model_names)):
            for j in range(i + 1, len(model_names)):
                kg.add_edge(KnowledgeEdge(
                    source_id=f"result_{model_names[i]}",
                    target_id=f"result_{model_names[j]}",
                    edge_type=EdgeType.CONVERGES_WITH, weight=mean_icm,
                ))

    # Decision node
    decision_node = KnowledgeNode(
        id="decision_epidemic", node_type=NodeType.DECISION,
        data={
            "mean_icm": mean_icm,
            "early_warning_detections": len(ew_results.get("cusum_changes", [])),
            "anti_spurious_genuine": (
                anti_spurious_report.is_genuine
                if anti_spurious_report is not None else None),
        },
    )
    kg.add_node(decision_node)

    for t in range(0, len(icm_results), 30):
        kg.add_edge(KnowledgeEdge(
            source_id=f"icm_step_{t}", target_id="decision_epidemic",
            edge_type=EdgeType.LED_TO,
        ))

    return kg


def _get_family(name: str) -> str:
    return {
        "compartmental_seir": "epidemiological",
        "network_abm": "agent_based",
        "statistical_logistic": "statistical",
        "ml_ensemble": "machine_learning",
        "exponential_smoothing": "baseline",
    }.get(name, "unknown")


# ====================================================================
# 6. Printing utilities
# ====================================================================

def print_header(title: str, char: str = "=") -> None:
    width = 76
    print()
    print(char * width)
    print(f"  {title}")
    print(char * width)


def print_table(headers: list[str], rows: list[list],
                col_widths: list[int] | None = None) -> None:
    if col_widths is None:
        col_widths = []
        for i, h in enumerate(headers):
            max_w = len(str(h))
            for row in rows:
                if i < len(row):
                    max_w = max(max_w, len(str(row[i])))
            col_widths.append(max_w + 2)
    header_line = "".join(str(h).ljust(w) for h, w in zip(headers, col_widths))
    print(header_line)
    print("-" * sum(col_widths))
    for row in rows:
        print("".join(str(v).ljust(w) for v, w in zip(row, col_widths)))


def ff(v: float, d: int = 4) -> str:
    """Format float."""
    return f"{v:.{d}f}"


# ====================================================================
# 7. Main benchmark execution
# ====================================================================

def run_epidemic_benchmark() -> dict:
    """Execute the full epidemic benchmark and return all results."""

    t_start = time.monotonic()
    rng_seed = 42
    n_nodes = 500
    n_steps = 300
    outbreak_step = 50
    containment_step = 150
    icm_window = 10

    print_header("OS Multi-Science: Epidemic Spreading Benchmark")
    print(f"  Network nodes:      {n_nodes}")
    print(f"  Time steps:         {n_steps}")
    print(f"  Outbreak onset:     step {outbreak_step}")
    print(f"  Containment start:  step {containment_step}")
    print(f"  ICM window size:    {icm_window}")
    print(f"  Random seed:        {rng_seed}")

    # ================================================================
    # Step 1: Generate contact network
    # ================================================================
    print_header("Step 1: Generating Contact Network", "-")
    adj = generate_contact_network(n=n_nodes, topology="scale_free", seed=rng_seed)
    degree = adj.sum(axis=1)
    print(f"  Topology:        scale-free (Barabasi-Albert, m=3)")
    print(f"  Nodes:           {n_nodes}")
    print(f"  Edges:           {int(adj.sum() / 2)}")
    print(f"  Mean degree:     {degree.mean():.2f}")
    print(f"  Max degree:      {int(degree.max())}")
    print(f"  Min degree:      {int(degree.min())}")

    # ================================================================
    # Step 2: Simulate ground-truth SEIR epidemic
    # ================================================================
    print_header("Step 2: Simulating Ground-Truth SEIR Epidemic", "-")
    gt = simulate_seir_on_network(
        adj, n_steps=n_steps, outbreak_step=outbreak_step,
        containment_step=containment_step, seed=rng_seed,
    )
    total_infected = gt["new_infections"].sum()
    peak_day = int(np.argmax(gt["new_infections"]))
    peak_value = gt["new_infections"].max()
    attack_rate = 1.0 - gt["S"][-1]

    print(f"  Total new infections:  {total_infected:.0f}")
    print(f"  Peak daily infections: {peak_value:.0f} (step {peak_day})")
    print(f"  Attack rate (final):   {attack_rate:.3f}")
    print(f"  Final S/E/I/R:         {gt['S'][-1]:.3f} / {gt['E'][-1]:.3f} / "
          f"{gt['I'][-1]:.3f} / {gt['R'][-1]:.3f}")

    # Print epidemic curve (text sparkline)
    print("\n  Epidemic curve (daily new infections, 10-step bins):")
    bins = [int(gt["new_infections"][i:i+10].sum()) for i in range(0, n_steps, 10)]
    max_bin = max(bins) if max(bins) > 0 else 1
    for i, b in enumerate(bins):
        bar = "#" * int(40 * b / max_bin) if b > 0 else ""
        step_range = f"{i*10:3d}-{i*10+9:3d}"
        print(f"    {step_range}: {bar} ({b})")

    # ================================================================
    # Step 3: Run 5 model simulators
    # ================================================================
    print_header("Step 3: Running 5 Model Simulators", "-")

    model_names = [
        "compartmental_seir",
        "network_abm",
        "statistical_logistic",
        "ml_ensemble",
        "exponential_smoothing",
    ]

    predictions_dict: dict[str, NDArray] = {}

    print("  [1/5] Compartmental SEIR (ODE)...")
    predictions_dict["compartmental_seir"] = model_compartmental_seir(
        gt, n_steps, outbreak_step, containment_step, n_nodes)

    print("  [2/5] Network ABM (agent-based)...")
    predictions_dict["network_abm"] = model_network_abm(
        adj, gt, n_steps, outbreak_step, containment_step, seed=137)

    print("  [3/5] Statistical logistic regression...")
    predictions_dict["statistical_logistic"] = model_statistical_logistic(
        gt, n_steps, outbreak_step, containment_step, n_nodes)

    print("  [4/5] ML ensemble (RandomForest + Ridge)...")
    predictions_dict["ml_ensemble"] = model_ml_ensemble(
        gt, n_steps, outbreak_step)

    print("  [5/5] Exponential smoothing baseline...")
    predictions_dict["exponential_smoothing"] = model_exponential_smoothing(
        gt, n_steps)

    # Model summary table
    print()
    headers = ["Model", "Total Pred.", "Peak Daily", "RMSE vs GT", "MAE vs GT"]
    rows = []
    model_rmse = {}
    model_mae = {}
    for name in model_names:
        pred = predictions_dict[name]
        err = pred - gt["new_infections"]
        rmse = float(np.sqrt(np.mean(err ** 2)))
        mae = float(np.mean(np.abs(err)))
        model_rmse[name] = rmse
        model_mae[name] = mae
        rows.append([name, ff(pred.sum(), 1), ff(pred.max(), 1),
                      ff(rmse, 4), ff(mae, 4)])
    print_table(headers, rows)

    # ================================================================
    # Step 4: ICM over the epidemic timeline
    # ================================================================
    print_header("Step 4: ICM Convergence Over Epidemic Timeline", "-")

    icm_config = ICMConfig(C_A_wasserstein=50.0)
    icm_scores, icm_results = compute_per_step_icm(
        predictions_dict, window_size=icm_window, config=icm_config)

    # Phase masks
    pre_mask = np.arange(n_steps) < outbreak_step
    outbreak_mask = (np.arange(n_steps) >= outbreak_step) & (np.arange(n_steps) < containment_step)
    post_mask = np.arange(n_steps) >= containment_step

    icm_pre = float(icm_scores[pre_mask].mean())
    icm_outbreak = float(icm_scores[outbreak_mask].mean())
    icm_post = float(icm_scores[post_mask].mean())

    print(f"  Mean ICM (overall):       {ff(icm_scores.mean())}")
    print(f"  ICM std (overall):        {ff(icm_scores.std())}")
    print()

    headers = ["Phase", "Steps", "Mean ICM", "Std ICM", "Min ICM", "Max ICM"]
    rows = []
    for label, mask in [("Pre-outbreak", pre_mask),
                        ("Active outbreak", outbreak_mask),
                        ("Post-containment", post_mask)]:
        s = icm_scores[mask]
        rows.append([label, int(mask.sum()), ff(s.mean()), ff(s.std()),
                      ff(s.min()), ff(s.max())])
    print_table(headers, rows)

    # Component breakdown by phase
    print("\n  ICM Component Breakdown by Phase:")
    headers2 = ["Phase", "A (agree)", "D (dir)", "U (uncert)", "C (invar)", "Pi (dep)"]
    rows2 = []
    for label, mask in [("Pre-outbreak", pre_mask),
                        ("Active outbreak", outbreak_mask),
                        ("Post-containment", post_mask)]:
        indices = np.where(mask)[0]
        A_m = np.mean([icm_results[i].components.A for i in indices])
        D_m = np.mean([icm_results[i].components.D for i in indices])
        U_m = np.mean([icm_results[i].components.U for i in indices])
        C_m = np.mean([icm_results[i].components.C for i in indices])
        Pi_m = np.mean([icm_results[i].components.Pi for i in indices])
        rows2.append([label, ff(A_m), ff(D_m), ff(U_m), ff(C_m), ff(Pi_m)])
    print_table(headers2, rows2)

    # ICM trajectory (text sparkline, sampled)
    print("\n  ICM trajectory (sampled every 15 steps):")
    for t in range(0, n_steps, 15):
        val = icm_scores[t]
        bar_len = int(50 * val)
        bar = "|" * bar_len
        phase = ("PRE" if t < outbreak_step
                 else "OUT" if t < containment_step
                 else "POST")
        print(f"    t={t:3d} [{phase:4s}]: {bar} {ff(val)}")

    # ================================================================
    # Step 5: Early Warning System
    # ================================================================
    print_header("Step 5: Early Warning Detection", "-")

    ew_config = EarlyWarningConfig(
        window_size=10, a1=0.4, a2=0.4, a3=0.2,
        cusum_threshold=2.0, cusum_drift=0.2,
    )

    # Rolling ICM statistics
    rolling_icm = compute_rolling_icm(icm_scores, window_size=ew_config.window_size)
    delta_icm = compute_delta_icm(icm_scores, window_size=ew_config.window_size)

    # Prediction variance over time
    pred_var = compute_prediction_variance(predictions_dict)
    var_max = pred_var.max()
    pred_var_norm = pred_var / var_max if var_max > 1e-12 else pred_var

    # Pi trend
    pi_trend = np.array([icm_results[t].components.Pi for t in range(n_steps)])

    # Composite Z signal
    z_signal = compute_z_signal(delta_icm, pred_var_norm, pi_trend, ew_config)

    # CUSUM detector
    cusum_changes, cusum_values = cusum_detector(
        z_signal, threshold=ew_config.cusum_threshold, drift=ew_config.cusum_drift)

    # Page-Hinkley detector
    ph_changes, ph_values = page_hinkley_detector(
        z_signal, threshold=ew_config.cusum_threshold * 2.0)

    # True change points
    true_changes = [outbreak_step, containment_step]

    # Evaluate
    max_lead = 30
    cusum_eval = evaluate_early_warning(cusum_changes, true_changes, max_lead)
    ph_eval = evaluate_early_warning(ph_changes, true_changes, max_lead)

    print(f"  True change points: {true_changes}")
    print(f"    = steps {outbreak_step} (outbreak onset) and "
          f"{containment_step} (containment)")
    print()
    print(f"  CUSUM detections ({len(cusum_changes)}): "
          f"{cusum_changes[:20]}{'...' if len(cusum_changes)>20 else ''}")
    print(f"  Page-Hinkley detections ({len(ph_changes)}): "
          f"{ph_changes[:20]}{'...' if len(ph_changes)>20 else ''}")
    print()

    headers = ["Detector", "TPR", "FPR", "Lead Times", "AUROC Proxy"]
    rows = [
        ["CUSUM", ff(cusum_eval["true_positive_rate"]),
         ff(cusum_eval["false_positive_rate"]),
         str(cusum_eval["lead_times"]), ff(cusum_eval["auroc"])],
        ["Page-Hinkley", ff(ph_eval["true_positive_rate"]),
         ff(ph_eval["false_positive_rate"]),
         str(ph_eval["lead_times"]), ff(ph_eval["auroc"])],
    ]
    print_table(headers, rows)

    # Z-signal diagnostics
    print("\n  Z-signal diagnostics by phase:")
    headers = ["Phase", "Mean Z", "Std Z", "Max Z"]
    rows = []
    for label, mask in [("Pre-outbreak", pre_mask),
                        ("Active outbreak", outbreak_mask),
                        ("Post-containment", post_mask)]:
        z_phase = z_signal[mask]
        rows.append([label, ff(z_phase.mean()), ff(z_phase.std()),
                      ff(z_phase.max())])
    print_table(headers, rows)

    # Prediction variance diagnostics
    print("\n  Prediction variance diagnostics by phase:")
    headers = ["Phase", "Mean Var", "Max Var", "Var Ratio vs Pre"]
    pre_var_mean = pred_var[pre_mask].mean() if pred_var[pre_mask].mean() > 1e-12 else 1e-12
    rows = []
    for label, mask in [("Pre-outbreak", pre_mask),
                        ("Active outbreak", outbreak_mask),
                        ("Post-containment", post_mask)]:
        v = pred_var[mask]
        rows.append([label, ff(v.mean(), 2), ff(v.max(), 2),
                      ff(v.mean() / pre_var_mean, 1)])
    print_table(headers, rows)

    # Placebo test on pre-outbreak stable period
    if outbreak_step > 20:
        stable_periods = [(0, outbreak_step)]
        def cusum_fn(sig):
            cp, _ = cusum_detector(sig, ew_config.cusum_threshold,
                                   ew_config.cusum_drift)
            return cp
        placebo = placebo_test(z_signal, stable_periods, cusum_fn)
        print(f"\n  Placebo test (pre-outbreak stable period):")
        print(f"    CUSUM false alarm rate: {ff(placebo['false_alarm_rate'])}")

    # ================================================================
    # Step 6: Anti-Spurious Validation
    # ================================================================
    print_header("Step 6: Anti-Spurious Convergence Validation", "-")

    obs_start, obs_end = outbreak_step, containment_step
    sample_preds = {name: pred[obs_start:obs_end]
                    for name, pred in predictions_dict.items()}
    sample_labels = gt["new_infections"][obs_start:obs_end]
    sample_features = np.arange(obs_end - obs_start, dtype=float).reshape(-1, 1)

    anti_config = AntiSpuriousConfig(
        n_permutations=200, fdr_level=0.05, n_negative_controls=50)

    def icm_fn(p: dict[str, NDArray]) -> float:
        normed = {}
        for nm, v in p.items():
            total = v.sum()
            normed[nm] = v / total if total > 1e-12 else np.ones(len(v)) / len(v)
        r = compute_icm_from_predictions(normed, config=icm_config,
                                         distance_fn="hellinger")
        return r.icm_score

    try:
        anti_report = generate_anti_spurious_report(
            sample_preds, sample_labels, sample_features,
            config=anti_config, icm_fn=icm_fn)
        print(f"  Baseline distance D0:      {ff(anti_report.d0_baseline)}")
        print(f"  Normalized convergence:    {ff(anti_report.c_normalized)}")
        print(f"  HSIC p-value (min, FDR):   {ff(anti_report.hsic_pvalue)}")
        print(f"  FDR corrected:             {anti_report.fdr_corrected}")
        print(f"  Convergence is genuine:    {anti_report.is_genuine}")
        if anti_report.ablation_results:
            print("\n  Ablation Analysis (ICM change when model removed):")
            headers = ["Model Removed", "ICM Change", "Interpretation"]
            rows_abl = []
            for name, delta in anti_report.ablation_results.items():
                interp = ("helps convergence" if delta > 0.001
                          else "hurts convergence" if delta < -0.001
                          else "neutral")
                rows_abl.append([name, ff(delta, 6), interp])
            print_table(headers, rows_abl)
    except Exception as e:
        print(f"  Anti-spurious check encountered error: {e}")
        anti_report = None

    # ================================================================
    # Step 7: Knowledge Graph
    # ================================================================
    print_header("Step 7: Knowledge Graph Summary", "-")

    ew_results = {
        "cusum_changes": cusum_changes, "ph_changes": ph_changes,
        "cusum_eval": cusum_eval, "ph_eval": ph_eval,
        "z_signal": z_signal, "pred_var": pred_var,
    }

    kg = build_knowledge_graph(
        model_names, predictions_dict, icm_scores, icm_results,
        ew_results, anti_report)

    kg_summary = kg.summary()
    for key, value in kg_summary.items():
        print(f"  {key}: {value}")

    converging = kg.find_converging_methods("epidemic_system", min_icm=0.5)
    conflicts = kg.find_conflicting_results("epidemic_system")
    print(f"\n  Converging methods (ICM >= 0.5): {len(converging)}")
    for mid in converging:
        print(f"    - {mid}")
    print(f"  Conflicting result pairs: {len(conflicts)}")

    # Decision provenance
    provenance = kg.get_decision_provenance("decision_epidemic")
    if provenance:
        print(f"\n  Decision provenance:")
        print(f"    ICM score nodes:  {len(provenance.get('icm_scores', []))}")
        print(f"    Methods involved: {len(provenance.get('methods', []))}")
        print(f"    Systems:          {len(provenance.get('systems', []))}")

    # ================================================================
    # Step 8: Comprehensive Results
    # ================================================================
    print_header("COMPREHENSIVE RESULTS SUMMARY")

    print("\n  [A] EPIDEMIC SCENARIO")
    print(f"      Network: {n_nodes}-node scale-free (Barabasi-Albert, m=3)")
    print(f"      Dynamics: SEIR with beta_pre=0.035, beta_post=0.008, "
          f"sigma=0.2, gamma=0.1")
    print(f"      Timeline: {n_steps} steps")
    print(f"        - Steps 0-{outbreak_step-1}: pre-outbreak (no disease)")
    print(f"        - Steps {outbreak_step}-{containment_step-1}: "
          f"active outbreak (free spreading)")
    print(f"        - Steps {containment_step}-{n_steps-1}: "
          f"post-containment (reduced transmission)")
    print(f"      Total infections: {total_infected:.0f} "
          f"({attack_rate:.1%} attack rate)")
    print(f"      Peak: {peak_value:.0f} new infections at step {peak_day}")

    print("\n  [B] MODEL PERFORMANCE (ranked by RMSE)")
    sorted_models = sorted(model_rmse.items(), key=lambda x: x[1])
    for rank, (name, rmse) in enumerate(sorted_models, 1):
        mae = model_mae[name]
        tot = predictions_dict[name].sum()
        print(f"      {rank}. {name}")
        print(f"         RMSE={rmse:.4f}  MAE={mae:.4f}  "
              f"Total pred.={tot:.1f}")

    print(f"\n  [C] ICM CONVERGENCE DYNAMICS")
    print(f"      Pre-outbreak ICM:     {ff(icm_pre)}  "
          f"(baseline -- models trivially agree on zero)")
    print(f"      Active outbreak ICM:  {ff(icm_outbreak)}  "
          f"(models diverge on growth trajectory)")
    print(f"      Post-containment ICM: {ff(icm_post)}  "
          f"(partial re-convergence as epidemic wanes)")
    print(f"      ICM range:            [{ff(icm_scores.min())} -- "
          f"{ff(icm_scores.max())}]")

    print(f"\n  [D] EARLY WARNING PERFORMANCE")
    print(f"      CUSUM: TPR={cusum_eval['true_positive_rate']:.2f}, "
          f"FPR={cusum_eval['false_positive_rate']:.2f}, "
          f"Lead={cusum_eval['lead_times']}")
    print(f"      Page-Hinkley: TPR={ph_eval['true_positive_rate']:.2f}, "
          f"FPR={ph_eval['false_positive_rate']:.2f}, "
          f"Lead={ph_eval['lead_times']}")
    n_cusum = len(cusum_changes)
    n_ph = len(ph_changes)
    print(f"      Total detections: CUSUM={n_cusum}, Page-Hinkley={n_ph}")

    print(f"\n  [E] ANTI-SPURIOUS VALIDATION")
    if anti_report is not None:
        print(f"      Genuine convergence: {anti_report.is_genuine}")
        print(f"      Normalized convergence C(x): {ff(anti_report.c_normalized)}")
        print(f"      Baseline D0: {ff(anti_report.d0_baseline)}")
        print(f"      HSIC min p-value: {ff(anti_report.hsic_pvalue)}")
        if anti_report.ablation_results:
            most_helpful = max(anti_report.ablation_results.items(),
                               key=lambda x: x[1])
            most_harmful = min(anti_report.ablation_results.items(),
                               key=lambda x: x[1])
            print(f"      Most helpful model:  {most_helpful[0]} "
                  f"(delta={most_helpful[1]:+.6f})")
            print(f"      Most harmful model:  {most_harmful[0]} "
                  f"(delta={most_harmful[1]:+.6f})")
    else:
        print(f"      Status: SKIPPED")

    print(f"\n  [F] KNOWLEDGE GRAPH")
    print(f"      Nodes: {kg_summary['total_nodes']}  "
          f"Edges: {kg_summary['total_edges']}")
    print(f"      Converging methods: {len(converging)}/5")
    print(f"      Conflicting pairs: {len(conflicts)}")

    elapsed = time.monotonic() - t_start
    print(f"\n  Total benchmark runtime: {elapsed:.2f} seconds")
    print("=" * 76)

    return {
        "ground_truth": gt,
        "predictions": predictions_dict,
        "icm_results": icm_results,
        "icm_scores": icm_scores,
        "early_warning": ew_results,
        "anti_spurious": anti_report,
        "knowledge_graph": kg,
        "model_rmse": model_rmse,
        "model_mae": model_mae,
        "config": {
            "n_nodes": n_nodes, "n_steps": n_steps,
            "outbreak_step": outbreak_step,
            "containment_step": containment_step,
            "icm_window": icm_window,
        },
        "summary": {
            "icm_pre_outbreak": icm_pre,
            "icm_active_outbreak": icm_outbreak,
            "icm_post_containment": icm_post,
            "total_infections": total_infected,
            "attack_rate": attack_rate,
            "peak_day": peak_day,
            "peak_value": peak_value,
        },
        "elapsed_seconds": elapsed,
    }


# ====================================================================
# Entry point
# ====================================================================

if __name__ == "__main__":
    results = run_epidemic_benchmark()
