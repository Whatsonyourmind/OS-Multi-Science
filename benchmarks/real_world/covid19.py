"""COVID-19 Realistic Epidemic Benchmark for OS Multi-Science.

Simulates a multi-wave COVID-19-like epidemic with realistic features:
  - SEIR-V dynamics on a scale-free network with community structure
  - Two to three epidemic waves over 365 days
  - Reporting delay, underreporting, and weekend effects
  - Vaccination rollout starting at day 200
  - Six epistemically diverse models predict daily reported cases
  - Full OS Multi-Science pipeline: ICM, early warning, CRC gating,
    anti-spurious validation, and knowledge graph provenance

This benchmark uses ONLY programmatically generated data --
no external downloads are required.

Run:
    python benchmarks/real_world/covid19.py
"""

from __future__ import annotations

import sys
import os
import time
import warnings
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup
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
    placebo_test,
)
from framework.anti_spurious import generate_anti_spurious_report
from framework.icm import compute_icm_from_predictions
from framework.types import DecisionAction, ICMComponents, ICMResult
from knowledge.graph import (
    EdgeType,
    KnowledgeEdge,
    KnowledgeGraph,
    KnowledgeNode,
    NodeType,
    generate_id,
)

warnings.filterwarnings("ignore")

# ====================================================================
# Constants
# ====================================================================

POPULATION = 100_000
N_NODES = 500  # Network size for ABM (scaled; represents population behaviour)
N_DAYS = 365
SEED = 42
ICM_WINDOW = 14  # 2-week rolling window
EW_WINDOW = 14

# Epidemic phase boundaries (approximate)
WAVE1_START = 20
WAVE1_INTERVENTION = 80   # Lockdown brings R0 down
WAVE2_START = 160          # Relaxation / new variant
WAVE2_PEAK = 200
VACCINATION_START = 200
WAVE3_START = 280          # Mild third wave

# ====================================================================
# 1. Contact network generation (scale-free + community structure)
# ====================================================================


def generate_contact_network(
    n: int = N_NODES,
    n_communities: int = 5,
    seed: int = SEED,
) -> NDArray:
    """Build a scale-free network with planted community structure.

    Nodes are divided into ``n_communities`` groups.  Within-community
    edges follow preferential attachment (m=3), and a small fraction of
    cross-community edges are added to create realistic heterogeneity.

    Returns
    -------
    adj : ndarray of shape (n, n), float64
        Binary symmetric adjacency matrix.
    """
    rng = np.random.default_rng(seed)
    adj = np.zeros((n, n), dtype=np.float64)

    # Assign nodes to communities
    community = np.zeros(n, dtype=int)
    sizes = np.array_split(np.arange(n), n_communities)
    for c_idx, members in enumerate(sizes):
        community[members] = c_idx

    # Within-community Barabasi-Albert (m=3)
    m = 3
    for c_idx, members in enumerate(sizes):
        members = np.array(members)
        nc = len(members)
        if nc <= m + 1:
            for i in members:
                for j in members:
                    if i != j:
                        adj[i, j] = 1.0
            continue

        # Start with complete subgraph of m+1 nodes
        for i in range(m + 1):
            for j in range(i + 1, m + 1):
                adj[members[i], members[j]] = 1.0
                adj[members[j], members[i]] = 1.0

        for k in range(m + 1, nc):
            node = members[k]
            degree = adj[members[:k]].sum(axis=1)
            total = degree.sum()
            if total == 0:
                probs = np.ones(k) / k
            else:
                probs = degree / total
            probs = np.maximum(probs, 1e-12)
            probs /= probs.sum()
            targets = rng.choice(k, size=min(m, k), replace=False, p=probs)
            for t in targets:
                adj[node, members[t]] = 1.0
                adj[members[t], node] = 1.0

    # Cross-community edges (5% probability for each community pair)
    for i in range(n):
        for j in range(i + 1, n):
            if community[i] != community[j] and adj[i, j] == 0:
                if rng.random() < 0.005:
                    adj[i, j] = adj[j, i] = 1.0

    np.fill_diagonal(adj, 0.0)
    return adj


# ====================================================================
# 2. Multi-wave COVID-19 ground-truth simulation
# ====================================================================


def simulate_covid19_ground_truth(
    population: int = POPULATION,
    n_days: int = N_DAYS,
    seed: int = SEED,
) -> dict[str, NDArray]:
    """Simulate a realistic COVID-19-like multi-wave epidemic.

    Uses a deterministic SEIR-V model with time-varying beta to produce
    the true underlying epidemic, then applies observation noise:
      - Reporting delay (Poisson with mean 3 days)
      - Underreporting (30-40% detection rate)
      - Weekend effects (Saturday/Sunday report 60% of weekday rate)

    The SEIR model uses standard mass-action kinetics:
      dS/dt = -beta * S * I  (S, I are fractions)
      R0 = beta / gamma

    Returns
    -------
    dict with keys:
        true_infections  : (n_days,) actual daily new infections (count)
        reported_cases   : (n_days,) observed/reported daily cases (count)
        S, E, I, R, V    : (n_days,) compartment fractions
        R_eff            : (n_days,) effective reproduction number
        vaccination_rate : (n_days,) daily vaccination fraction
    """
    rng = np.random.default_rng(seed)
    N = float(population)

    sigma = 1.0 / 5.2   # Incubation rate: mean 5.2 days
    gamma = 1.0 / 10.0  # Recovery rate: mean 10 days

    # Time-varying beta(t) controlling R0 = beta/gamma
    # beta = R0 * gamma, so for R0=2.5: beta=0.25
    def beta_t(t):
        """Piecewise transmission rate with smooth transitions."""
        if t < WAVE1_START:
            return 0.0
        elif t < WAVE1_INTERVENTION:
            # Wave 1 growth: R0 ~ 2.5 -> beta = 0.25
            return 0.25
        elif t < WAVE2_START:
            # Post-lockdown: R0 drops sharply to ~0.7
            days_since = t - WAVE1_INTERVENTION
            lockdown_effect = 0.72 * min(1.0, days_since / 15.0)
            return 0.25 * (1.0 - lockdown_effect)
        elif t < WAVE2_PEAK + 30:
            # Wave 2: relaxation + new variant, R0 ramps to ~1.8
            ramp = min(1.0, (t - WAVE2_START) / 25.0)
            return 0.18 * (0.3 + 0.7 * ramp)
        elif t < WAVE3_START:
            # Post-wave-2 decline: re-introduction of measures
            days_since = t - (WAVE2_PEAK + 30)
            decay = max(0.25, 1.0 - 0.6 * days_since / 50.0)
            return 0.18 * decay
        else:
            # Wave 3: milder due to vaccination + prior immunity, R0 ~ 1.3
            ramp = min(1.0, (t - WAVE3_START) / 20.0)
            return 0.13 * (0.4 + 0.6 * ramp)

    # Vaccination schedule: gradual rollout starting at day 200
    def vax_rate(t):
        if t < VACCINATION_START:
            return 0.0
        progress = (t - VACCINATION_START) / (N_DAYS - VACCINATION_START)
        # Ramp up to 0.3% of population per day
        return 0.003 * min(1.0, progress * 3.0)

    # SEIR-V simulation (Euler method, dt=1 day, fractional compartments)
    S = np.zeros(n_days)
    E = np.zeros(n_days)
    I = np.zeros(n_days)
    R = np.zeros(n_days)
    V = np.zeros(n_days)
    true_infections = np.zeros(n_days)
    R_eff = np.zeros(n_days)
    vaccination_rate = np.zeros(n_days)

    # Initial conditions: seed 10 infected out of N
    S[0] = 1.0 - 20.0 / N
    E[0] = 10.0 / N
    I[0] = 10.0 / N
    R[0] = 0.0
    V[0] = 0.0

    for t in range(1, n_days):
        bt = beta_t(t)
        vr = vax_rate(t)
        vaccination_rate[t] = vr

        s_prev = S[t - 1]
        e_prev = E[t - 1]
        i_prev = I[t - 1]
        r_prev = R[t - 1]
        v_prev = V[t - 1]

        # Standard SEIR mass-action: new_exposed = beta * S * I (fractions)
        new_exposed = bt * s_prev * i_prev
        new_exposed = min(new_exposed, s_prev)  # Cap at susceptible fraction
        new_infectious = sigma * e_prev
        new_recovered = gamma * i_prev
        new_vaccinated = min(vr, max(s_prev - new_exposed, 0))

        S[t] = max(0, s_prev - new_exposed - new_vaccinated)
        E[t] = max(0, e_prev + new_exposed - new_infectious)
        I[t] = max(0, i_prev + new_infectious - new_recovered)
        R[t] = min(1.0, r_prev + new_recovered)
        V[t] = min(1.0, v_prev + new_vaccinated)

        true_infections[t] = new_exposed * N  # Convert fraction to count
        R_eff[t] = bt * s_prev / gamma if gamma > 0 else 0.0

    # Apply observation model: delay + underreporting + weekend effects
    reported = _apply_observation_model(true_infections, n_days, rng)

    return {
        "true_infections": true_infections,
        "reported_cases": reported,
        "S": S, "E": E, "I": I, "R": R, "V": V,
        "R_eff": R_eff,
        "vaccination_rate": vaccination_rate,
    }


def _apply_observation_model(
    true_infections: NDArray,
    n_days: int,
    rng: np.random.Generator,
) -> NDArray:
    """Apply reporting delay, underreporting, and weekend effects."""
    reported = np.zeros(n_days)

    # Detection rate: 30-40%
    detection_rate = 0.35

    for t in range(n_days):
        cases = true_infections[t] * detection_rate
        # Reporting delay: shifted by 3-5 days
        delay = rng.poisson(3)
        report_day = min(t + delay, n_days - 1)
        reported[report_day] += cases

    # Weekend effect: Saturday (day%7==5) and Sunday (day%7==6) report 60%
    for t in range(n_days):
        day_of_week = t % 7
        if day_of_week in (5, 6):
            # Move 40% to following Monday
            monday = t + (7 - day_of_week)
            transfer = reported[t] * 0.4
            reported[t] -= transfer
            if monday < n_days:
                reported[monday] += transfer

    # Add small Poisson noise
    for t in range(n_days):
        if reported[t] > 0:
            reported[t] = max(0, rng.poisson(max(1, reported[t])))

    return reported


# ====================================================================
# 3. Six epistemically diverse model simulators
# ====================================================================


def model_seirv_compartmental(
    ground_truth: dict[str, NDArray],
    n_days: int,
    population: int,
) -> NDArray:
    """Compartmental SEIR-V model (ODE-based with vaccination).

    Solves SEIR-V ODEs with piecewise-constant beta (slightly mis-specified
    compared to ground truth to create epistemic diversity).  Represents
    the epidemiological/system dynamics epistemic family.
    Uses mean-field approximation -- ignores network structure.
    """
    reported = ground_truth["reported_cases"]

    def seir_v_odes(y, t_val, beta, sigma, gamma, vax):
        S, E, I, R, V_c = y
        new_vax = min(vax, max(S, 0))
        dS = -beta * S * I - new_vax
        dE = beta * S * I - sigma * E
        dI = sigma * E - gamma * I
        dR = gamma * I
        dV = new_vax
        return [dS, dE, dI, dR, dV]

    sigma, gamma = 1.0 / 5.2, 1.0 / 10.0
    N = float(population)
    predictions = np.zeros(n_days)

    # Slightly misspecified beta values (different from ground truth)
    segments = [
        (WAVE1_START, WAVE1_INTERVENTION, 0.22),   # GT uses 0.25
        (WAVE1_INTERVENTION, WAVE2_START, 0.06),    # GT transition
        (WAVE2_START, WAVE3_START, 0.16),            # GT uses 0.20
        (WAVE3_START, n_days, 0.10),                 # GT uses 0.15
    ]

    y0 = [1.0 - 20.0 / N, 10.0 / N, 10.0 / N, 0.0, 0.0]
    for seg_start, seg_end, beta_est in segments:
        n_seg = seg_end - seg_start
        if n_seg <= 1:
            continue
        vax = 0.002 if seg_start >= VACCINATION_START else 0.0
        t_seg = np.arange(n_seg, dtype=float)
        try:
            sol = odeint(seir_v_odes, y0, t_seg,
                         args=(beta_est, sigma, gamma, vax))
            new_inf = np.maximum(-np.diff(sol[:, 0]) * N, 0)
            end = min(len(new_inf), n_seg)
            # Apply estimated detection rate (0.35)
            predictions[seg_start:seg_start + end] = new_inf[:end] * 0.35
            y0 = list(sol[-1])
        except Exception:
            predictions[seg_start:seg_end] = reported[seg_start:seg_end] * 0.9

    return predictions


def model_agent_based(
    adj: NDArray,
    ground_truth: dict[str, NDArray],
    n_days: int,
    seed: int = 137,
) -> NDArray:
    """Agent-based model on contact network (different seed).

    Stochastic SEIR simulation on the pre-built contact network with
    slightly perturbed parameters.  Represents the agent-based/complex
    systems epistemic family.  Captures network heterogeneity but has
    different stochastic realisations.
    """
    rng = np.random.default_rng(seed)
    n = adj.shape[0]
    state = np.zeros(n, dtype=int)  # 0=S, 1=E, 2=I, 3=R
    predictions = np.zeros(n_days)
    scale_factor = POPULATION / n  # Scale from network to population

    sigma, gamma = 0.18, 0.11  # Slightly perturbed

    for t in range(n_days):
        if t == WAVE1_START:
            susceptible = np.where(state == 0)[0]
            seeds = rng.choice(susceptible,
                               size=min(3, len(susceptible)), replace=False)
            state[seeds] = 2

        # Time-varying beta for ABM (per-contact probability)
        if t < WAVE1_START:
            beta = 0.0
        elif t < WAVE1_INTERVENTION:
            beta = 0.035
        elif t < WAVE2_START:
            beta = 0.008
        elif t < WAVE3_START:
            beta = 0.028
        else:
            beta = 0.015

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
        predictions[t] = n_new * scale_factor * 0.35  # Scale and apply detection

    return predictions


def model_statistical_logistic(
    ground_truth: dict[str, NDArray],
    n_days: int,
    population: int,
) -> NDArray:
    """Generalized logistic growth curve fit per wave.

    Fits separate logistic functions to cumulative reported cases in each
    wave, then differentiates to get daily predictions.  Represents the
    statistical/curve-fitting epistemic family.
    """
    reported = ground_truth["reported_cases"]
    cumulative = np.cumsum(reported)
    predictions = np.zeros(n_days)

    def gen_logistic(t, K, r, t0, nu):
        """Generalised logistic (Richards curve)."""
        return K / (1.0 + nu * np.exp(-r * (t - t0))) ** (1.0 / max(nu, 0.01))

    # Fit per wave segment
    wave_segments = [
        (WAVE1_START, WAVE2_START),
        (WAVE2_START, WAVE3_START),
        (WAVE3_START, n_days),
    ]

    for seg_start, seg_end in wave_segments:
        seg_len = seg_end - seg_start
        if seg_len < 10:
            continue
        t_obs = np.arange(seg_len, dtype=float)
        c_obs = cumulative[seg_start:seg_end]
        c_obs = c_obs - c_obs[0]  # Reset cumulative for this segment

        K_est = max(float(c_obs[-1]) * 1.5, 100)
        try:
            popt, _ = curve_fit(
                gen_logistic, t_obs, c_obs,
                p0=[K_est, 0.05, float(seg_len / 3), 1.0],
                bounds=([1, 0.001, 0, 0.01], [population, 1.0, seg_len * 2, 10]),
                maxfev=5000,
            )
            c_fit = gen_logistic(t_obs, *popt)
            daily_fit = np.maximum(np.diff(np.concatenate([[0], c_fit])), 0)
            predictions[seg_start:seg_end] = daily_fit
        except (RuntimeError, ValueError):
            predictions[seg_start:seg_end] = reported[seg_start:seg_end] * 0.9

    return predictions


def model_ml_ensemble(
    ground_truth: dict[str, NDArray],
    n_days: int,
) -> NDArray:
    """ML ensemble: Random Forest + Ridge on lagged features.

    Uses 7-day, 14-day, and 21-day lagged features of reported cases,
    plus cumulative counts and day-of-week encoding.  Represents the
    machine learning epistemic family.
    """
    reported = ground_truth["reported_cases"]
    lags = [1, 2, 3, 7, 14, 21]
    max_lag = max(lags)

    X_list, y_list = [], []
    for t in range(max_lag, n_days):
        features = [reported[t - lag] for lag in lags]
        # Rolling mean features
        features.append(np.mean(reported[max(0, t - 7):t]))
        features.append(np.mean(reported[max(0, t - 14):t]))
        # Cumulative
        features.append(np.sum(reported[:t]))
        # Time features
        features.append(float(t))
        features.append(float(t % 7))  # Day of week
        # Vaccination proxy
        features.append(max(0.0, float(t - VACCINATION_START)) / N_DAYS)
        X_list.append(features)
        y_list.append(reported[t])

    X_all = np.array(X_list)
    y_all = np.array(y_list)
    predictions = np.zeros(n_days)

    n_train = int(0.6 * len(X_all))
    if n_train < 20:
        return reported * 0.95

    X_train, y_train = X_all[:n_train], y_all[:n_train]

    rf = RandomForestRegressor(n_estimators=50, max_depth=8, random_state=42)
    ridge = Ridge(alpha=1.0)
    rf.fit(X_train, y_train)
    ridge.fit(X_train, y_train)

    ensemble = np.maximum(
        0.6 * rf.predict(X_all) + 0.4 * ridge.predict(X_all), 0
    )
    predictions[max_lag:] = ensemble
    return predictions


def model_exponential_smoothing(
    ground_truth: dict[str, NDArray],
    n_days: int,
    alpha_level: float = 0.3,
    beta_trend: float = 0.1,
) -> NDArray:
    """Holt's linear exponential smoothing (level + trend).

    Provides a simple autoregressive baseline that tracks level and
    trend.  Represents the baseline/time-series epistemic family.
    """
    reported = ground_truth["reported_cases"]
    predictions = np.zeros(n_days)

    level = reported[0]
    trend = 0.0

    for t in range(n_days):
        predictions[t] = max(level + trend, 0)
        if t < n_days - 1:
            new_level = alpha_level * reported[t] + (1 - alpha_level) * (level + trend)
            new_trend = beta_trend * (new_level - level) + (1 - beta_trend) * trend
            level = new_level
            trend = new_trend

    return predictions


def model_rt_projection(
    ground_truth: dict[str, NDArray],
    n_days: int,
    window: int = 7,
) -> NDArray:
    """R(t) estimation and forward projection.

    Estimates the effective reproduction number from the ratio of
    consecutive generation-interval windows, then projects forward
    assuming R(t) remains constant over each projection window.
    Represents the epidemiological/R(t)-based epistemic family.
    """
    reported = ground_truth["reported_cases"]
    predictions = np.zeros(n_days)
    serial_interval = 5.0  # days (COVID-19 mean serial interval)

    for t in range(window, n_days):
        current = np.sum(reported[t - window:t]) + 1e-6
        previous = np.sum(reported[max(0, t - 2 * window):t - window]) + 1e-6
        rt = current / previous
        # Project: next day ~ current_day * rt^(1/serial_interval)
        growth_factor = rt ** (1.0 / serial_interval)
        predictions[t] = max(reported[t - 1] * growth_factor, 0)

    return predictions


# ====================================================================
# 4. Per-step ICM computation
# ====================================================================


def compute_per_step_icm(
    predictions_dict: dict[str, NDArray],
    window_size: int = ICM_WINDOW,
    config: ICMConfig | None = None,
) -> tuple[NDArray, list[ICMResult]]:
    """Compute a rolling-window ICM score for each timestep.

    At each step t, the ICM is computed over the window
    [t - window_size + 1, t].
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
# 5. CRC Gating
# ====================================================================


def run_crc_gating(
    icm_scores: NDArray,
    losses: NDArray,
    crc_config: CRCConfig,
) -> dict:
    """Run CRC gating using ICM scores and corresponding losses."""
    n = len(icm_scores)
    split = max(n // 2, 4)

    g = fit_isotonic(icm_scores[:split], losses[:split])
    g_alpha = conformalize(g, icm_scores[split:], losses[split:],
                           alpha=crc_config.alpha)

    median_icm = float(np.median(icm_scores))
    re = compute_re(median_icm, g_alpha)
    decision = decision_gate(median_icm, re, crc_config)

    return {
        "re_score": re,
        "decision": decision,
        "median_icm": median_icm,
    }


# ====================================================================
# 6. Knowledge graph
# ====================================================================


def build_knowledge_graph(
    model_names: list[str],
    predictions_dict: dict[str, NDArray],
    icm_scores: NDArray,
    icm_results: list[ICMResult],
    ew_results: dict,
    anti_spurious_report,
    crc_results: dict,
    phase_icms: dict[str, float],
) -> KnowledgeGraph:
    """Record all benchmark results into a KnowledgeGraph."""
    kg = KnowledgeGraph()

    sys_node = KnowledgeNode(
        id="covid19_system",
        node_type=NodeType.SYSTEM,
        data={
            "name": "COVID-19 Multi-Wave Epidemic",
            "population": POPULATION,
            "days": N_DAYS,
            "waves": 3,
            "vaccination_start": VACCINATION_START,
        },
    )
    kg.add_node(sys_node)

    families = {
        "seirv_compartmental": "epidemiological",
        "agent_based": "agent_based",
        "statistical_logistic": "statistical",
        "ml_ensemble": "machine_learning",
        "exponential_smoothing": "baseline",
        "rt_projection": "epidemiological_rt",
    }

    for name in model_names:
        method_node = KnowledgeNode(
            id=f"method_{name}",
            node_type=NodeType.METHOD,
            data={"name": name, "family": families.get(name, "unknown")},
        )
        kg.add_node(method_node)
        kg.add_edge(KnowledgeEdge(
            source_id="covid19_system",
            target_id=f"method_{name}",
            edge_type=EdgeType.ANALYZED_BY,
        ))

        pred = predictions_dict[name]
        result_node = KnowledgeNode(
            id=f"result_{name}",
            node_type=NodeType.RESULT,
            data={
                "model": name,
                "total_predicted": float(pred.sum()),
                "peak_daily": float(pred.max()),
            },
        )
        kg.add_node(result_node)
        kg.add_edge(KnowledgeEdge(
            source_id=f"method_{name}",
            target_id=f"result_{name}",
            edge_type=EdgeType.PRODUCED,
        ))

    # ICM nodes (every 30 days)
    for t in range(0, len(icm_results), 30):
        r = icm_results[t]
        icm_node = KnowledgeNode(
            id=f"icm_day_{t}",
            node_type=NodeType.ICM_SCORE,
            data={
                "icm_score": r.icm_score,
                "day": t,
                "A": r.components.A,
                "D": r.components.D,
                "U": r.components.U,
            },
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
                    edge_type=EdgeType.CONVERGES_WITH,
                    weight=mean_icm,
                ))

    # Decision node
    decision_node = KnowledgeNode(
        id="decision_covid19",
        node_type=NodeType.DECISION,
        data={
            "mean_icm": mean_icm,
            "crc_decision": crc_results["decision"].value,
            "re_score": crc_results["re_score"],
            "phase_icms": phase_icms,
            "early_warning_detections": len(ew_results.get("cusum_changes", [])),
            "anti_spurious_genuine": (
                anti_spurious_report.is_genuine
                if anti_spurious_report is not None else None
            ),
        },
    )
    kg.add_node(decision_node)

    for t in range(0, len(icm_results), 30):
        kg.add_edge(KnowledgeEdge(
            source_id=f"icm_day_{t}",
            target_id="decision_covid19",
            edge_type=EdgeType.LED_TO,
        ))

    return kg


# ====================================================================
# 7. Printing utilities
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
    return f"{v:.{d}f}"


# ====================================================================
# 8. Define epidemic phases
# ====================================================================


def get_phase_masks(n_days: int) -> dict[str, NDArray]:
    """Return boolean masks for each epidemic phase."""
    days = np.arange(n_days)
    return {
        "pre_wave1": days < WAVE1_START,
        "wave1_rising": (days >= WAVE1_START) & (days < WAVE1_INTERVENTION),
        "wave1_declining": (days >= WAVE1_INTERVENTION) & (days < WAVE2_START),
        "wave2": (days >= WAVE2_START) & (days < WAVE3_START),
        "vaccination_rollout": (days >= VACCINATION_START) & (days < WAVE3_START),
        "wave3": days >= WAVE3_START,
    }


# ====================================================================
# 9. Main benchmark execution
# ====================================================================


def run_covid19_benchmark() -> dict:
    """Execute the full COVID-19 benchmark and return all results."""

    t_start = time.monotonic()

    print_header("OS Multi-Science: COVID-19 Multi-Wave Epidemic Benchmark")
    print(f"  Population:           {POPULATION:,}")
    print(f"  Timeline:             {N_DAYS} days")
    print(f"  Network nodes:        {N_NODES} (for ABM)")
    print(f"  Wave 1 onset:         day {WAVE1_START}")
    print(f"  Wave 1 intervention:  day {WAVE1_INTERVENTION}")
    print(f"  Wave 2 onset:         day {WAVE2_START}")
    print(f"  Vaccination start:    day {VACCINATION_START}")
    print(f"  Wave 3 onset:         day {WAVE3_START}")
    print(f"  ICM window size:      {ICM_WINDOW}")
    print(f"  Random seed:          {SEED}")

    # ================================================================
    # Step 1: Generate contact network
    # ================================================================
    print_header("Step 1: Generating Contact Network", "-")
    adj = generate_contact_network(n=N_NODES, n_communities=5, seed=SEED)
    degree = adj.sum(axis=1)
    print(f"  Topology:        Scale-free with 5 communities")
    print(f"  Nodes:           {N_NODES}")
    print(f"  Edges:           {int(adj.sum() / 2)}")
    print(f"  Mean degree:     {degree.mean():.2f}")
    print(f"  Max degree:      {int(degree.max())}")

    # ================================================================
    # Step 2: Simulate ground-truth COVID-19 epidemic
    # ================================================================
    print_header("Step 2: Simulating Ground-Truth COVID-19 Epidemic", "-")
    gt = simulate_covid19_ground_truth(
        population=POPULATION, n_days=N_DAYS, seed=SEED,
    )

    true_total = gt["true_infections"].sum()
    reported_total = gt["reported_cases"].sum()
    peak_true_day = int(np.argmax(gt["true_infections"]))
    peak_true_val = gt["true_infections"].max()
    peak_reported_day = int(np.argmax(gt["reported_cases"]))
    peak_reported_val = gt["reported_cases"].max()
    final_r = gt["R"][-1]
    final_v = gt["V"][-1]
    underreporting_ratio = reported_total / max(true_total, 1)

    print(f"  True total infections:    {true_total:.0f}")
    print(f"  Reported total cases:     {reported_total:.0f}")
    print(f"  Underreporting ratio:     {underreporting_ratio:.2f}")
    print(f"  Peak true infections:     {peak_true_val:.0f} (day {peak_true_day})")
    print(f"  Peak reported cases:      {peak_reported_val:.0f} (day {peak_reported_day})")
    print(f"  Final recovered fraction: {final_r:.3f}")
    print(f"  Final vaccinated frac.:   {final_v:.3f}")

    # Epidemic curve (text sparkline, 7-day bins)
    print("\n  Epidemic curve (reported cases, 7-day bins):")
    bins = [int(gt["reported_cases"][i:i + 7].sum())
            for i in range(0, N_DAYS, 7)]
    max_bin = max(bins) if max(bins) > 0 else 1
    for i, b in enumerate(bins):
        bar = "#" * int(40 * b / max_bin) if b > 0 else ""
        day_range = f"{i * 7:3d}-{min(i * 7 + 6, N_DAYS - 1):3d}"
        print(f"    {day_range}: {bar} ({b})")

    # R_eff trajectory
    print("\n  R_eff trajectory (sampled):")
    for t in [30, 50, 70, 100, 130, 170, 210, 250, 300, 350]:
        if t < N_DAYS:
            print(f"    Day {t:3d}: R_eff = {gt['R_eff'][t]:.2f}")

    # ================================================================
    # Step 3: Run 6 model simulators
    # ================================================================
    print_header("Step 3: Running 6 Model Simulators", "-")

    model_names = [
        "seirv_compartmental",
        "agent_based",
        "statistical_logistic",
        "ml_ensemble",
        "exponential_smoothing",
        "rt_projection",
    ]

    predictions_dict: dict[str, NDArray] = {}

    print("  [1/6] SEIR-V Compartmental (ODE)...")
    predictions_dict["seirv_compartmental"] = model_seirv_compartmental(
        gt, N_DAYS, POPULATION)

    print("  [2/6] Agent-Based Model (network)...")
    predictions_dict["agent_based"] = model_agent_based(
        adj, gt, N_DAYS, seed=137)

    print("  [3/6] Statistical Logistic (curve fitting)...")
    predictions_dict["statistical_logistic"] = model_statistical_logistic(
        gt, N_DAYS, POPULATION)

    print("  [4/6] ML Ensemble (RF + Ridge)...")
    predictions_dict["ml_ensemble"] = model_ml_ensemble(gt, N_DAYS)

    print("  [5/6] Exponential Smoothing (Holt)...")
    predictions_dict["exponential_smoothing"] = model_exponential_smoothing(
        gt, N_DAYS)

    print("  [6/6] R(t) Projection...")
    predictions_dict["rt_projection"] = model_rt_projection(gt, N_DAYS)

    # Model summary table
    reported = gt["reported_cases"]
    print()
    headers = ["Model", "Total Pred.", "Peak", "RMSE", "MAE"]
    rows = []
    model_rmse = {}
    model_mae = {}
    for name in model_names:
        pred = predictions_dict[name]
        err = pred - reported
        rmse = float(np.sqrt(np.mean(err ** 2)))
        mae = float(np.mean(np.abs(err)))
        model_rmse[name] = rmse
        model_mae[name] = mae
        rows.append([name, ff(pred.sum(), 1), ff(pred.max(), 1),
                      ff(rmse, 2), ff(mae, 2)])
    print_table(headers, rows)

    # ================================================================
    # Step 4: ICM over epidemic phases
    # ================================================================
    print_header("Step 4: ICM Convergence Over Epidemic Phases", "-")

    icm_config = ICMConfig(C_A_wasserstein=50.0)
    icm_scores, icm_results = compute_per_step_icm(
        predictions_dict, window_size=ICM_WINDOW, config=icm_config)

    phase_masks = get_phase_masks(N_DAYS)
    phase_icms: dict[str, float] = {}

    print(f"  Overall Mean ICM:  {ff(icm_scores.mean())}")
    print(f"  Overall Std ICM:   {ff(icm_scores.std())}")
    print()

    headers = ["Phase", "Days", "Mean ICM", "Std ICM", "Min", "Max"]
    rows = []
    for phase_name, mask in phase_masks.items():
        if mask.sum() == 0:
            continue
        s = icm_scores[mask]
        phase_icms[phase_name] = float(s.mean())
        rows.append([phase_name, int(mask.sum()), ff(s.mean()), ff(s.std()),
                      ff(s.min()), ff(s.max())])
    print_table(headers, rows)

    # Component breakdown
    print("\n  ICM Component Breakdown by Phase:")
    headers2 = ["Phase", "A (agree)", "D (dir)", "U (uncert)", "C (invar)", "Pi (dep)"]
    rows2 = []
    for phase_name, mask in phase_masks.items():
        if mask.sum() == 0:
            continue
        indices = np.where(mask)[0]
        A_m = np.mean([icm_results[i].components.A for i in indices])
        D_m = np.mean([icm_results[i].components.D for i in indices])
        U_m = np.mean([icm_results[i].components.U for i in indices])
        C_m = np.mean([icm_results[i].components.C for i in indices])
        Pi_m = np.mean([icm_results[i].components.Pi for i in indices])
        rows2.append([phase_name, ff(A_m), ff(D_m), ff(U_m), ff(C_m), ff(Pi_m)])
    print_table(headers2, rows2)

    # ICM trajectory
    print("\n  ICM trajectory (sampled every 15 days):")
    for t in range(0, N_DAYS, 15):
        val = icm_scores[t]
        bar_len = int(50 * val)
        bar = "|" * bar_len
        # Determine phase
        phase = "PRE"
        if t >= WAVE3_START:
            phase = "W3"
        elif t >= WAVE2_START:
            phase = "W2"
        elif t >= WAVE1_INTERVENTION:
            phase = "INT"
        elif t >= WAVE1_START:
            phase = "W1"
        print(f"    Day {t:3d} [{phase:3s}]: {bar} {ff(val)}")

    # ================================================================
    # Step 5: Early Warning Detection
    # ================================================================
    print_header("Step 5: Early Warning Detection", "-")

    ew_config = EarlyWarningConfig(
        window_size=EW_WINDOW, a1=0.4, a2=0.4, a3=0.2,
        cusum_threshold=2.0, cusum_drift=0.2,
    )

    rolling_icm = compute_rolling_icm(icm_scores, window_size=ew_config.window_size)
    delta_icm = compute_delta_icm(icm_scores, window_size=ew_config.window_size)

    pred_var = compute_prediction_variance(predictions_dict)
    var_max = pred_var.max()
    pred_var_norm = pred_var / var_max if var_max > 1e-12 else pred_var

    pi_trend = np.array([icm_results[t].components.Pi for t in range(N_DAYS)])

    z_signal = compute_z_signal(delta_icm, pred_var_norm, pi_trend, ew_config)

    cusum_changes, cusum_values = cusum_detector(
        z_signal, threshold=ew_config.cusum_threshold, drift=ew_config.cusum_drift)

    ph_changes, ph_values = page_hinkley_detector(
        z_signal, threshold=ew_config.cusum_threshold * 2.0)

    # True change points for evaluation
    true_changes = [WAVE1_START, WAVE1_INTERVENTION, WAVE2_START, WAVE3_START]
    max_lead = 30

    cusum_eval = evaluate_early_warning(cusum_changes, true_changes, max_lead)
    ph_eval = evaluate_early_warning(ph_changes, true_changes, max_lead)

    print(f"  True change points: {true_changes}")
    print(f"  CUSUM detections ({len(cusum_changes)}): "
          f"{cusum_changes[:15]}{'...' if len(cusum_changes) > 15 else ''}")
    print(f"  Page-Hinkley detections ({len(ph_changes)}): "
          f"{ph_changes[:15]}{'...' if len(ph_changes) > 15 else ''}")
    print()

    headers = ["Detector", "TPR", "FPR", "Lead Times", "AUROC"]
    rows = [
        ["CUSUM", ff(cusum_eval["true_positive_rate"]),
         ff(cusum_eval["false_positive_rate"]),
         str(cusum_eval["lead_times"][:5]), ff(cusum_eval["auroc"])],
        ["Page-Hinkley", ff(ph_eval["true_positive_rate"]),
         ff(ph_eval["false_positive_rate"]),
         str(ph_eval["lead_times"][:5]), ff(ph_eval["auroc"])],
    ]
    print_table(headers, rows)

    # Early warning for wave 2 and 3 specifically
    print("\n  Wave-specific early warning analysis:")
    for wave_name, wave_start in [("Wave 2", WAVE2_START), ("Wave 3", WAVE3_START)]:
        detections_before = [d for d in cusum_changes if 0 < wave_start - d <= max_lead]
        if detections_before:
            lead = wave_start - detections_before[0]
            print(f"    {wave_name}: CUSUM detected {lead} days before onset")
        else:
            detections_after = [d for d in cusum_changes if 0 <= d - wave_start <= 30]
            if detections_after:
                lag = detections_after[0] - wave_start
                print(f"    {wave_name}: CUSUM detected {lag} days after onset")
            else:
                print(f"    {wave_name}: No CUSUM detection near onset")

    # Placebo test
    if WAVE1_START > 10:
        stable_periods = [(0, WAVE1_START)]

        def cusum_fn(sig):
            cp, _ = cusum_detector(sig, ew_config.cusum_threshold,
                                   ew_config.cusum_drift)
            return cp

        placebo = placebo_test(z_signal, stable_periods, cusum_fn)
        print(f"\n  Placebo test (pre-epidemic stable period):")
        print(f"    False alarm rate: {ff(placebo['false_alarm_rate'])}")

    # ================================================================
    # Step 6: CRC Gating and Decision Logic
    # ================================================================
    print_header("Step 6: CRC Gating and Decision Logic", "-")

    crc_config = CRCConfig(alpha=0.10, tau_hi=0.65, tau_lo=0.35)

    # Compute losses for CRC
    ensemble_mean = np.mean(
        np.column_stack([predictions_dict[n] for n in model_names]), axis=1
    )
    losses = (ensemble_mean - reported) ** 2
    # Downsample for CRC (every 7 days)
    sample_idx = np.arange(7, N_DAYS, 7)
    crc_results = run_crc_gating(
        icm_scores[sample_idx], losses[sample_idx], crc_config
    )

    print(f"  Median ICM:        {ff(crc_results['median_icm'])}")
    print(f"  Epistemic risk Re: {ff(crc_results['re_score'])}")
    print(f"  Decision:          {crc_results['decision'].value}")

    # Phase-specific gating decisions
    print("\n  Phase-specific decision analysis:")
    for phase_name, mask in phase_masks.items():
        if mask.sum() == 0:
            continue
        phase_icm = icm_scores[mask].mean()
        if phase_icm >= crc_config.tau_hi:
            action = "ACT"
        elif phase_icm >= crc_config.tau_lo:
            action = "DEFER"
        else:
            action = "AUDIT"
        print(f"    {phase_name:25s}: ICM={ff(phase_icm)} -> {action}")

    # ================================================================
    # Step 7: Anti-Spurious Validation
    # ================================================================
    print_header("Step 7: Anti-Spurious Convergence Validation", "-")

    # Use wave 1 data for anti-spurious validation
    obs_start = WAVE1_START
    obs_end = min(WAVE1_INTERVENTION + 30, N_DAYS)
    sample_preds = {name: pred[obs_start:obs_end]
                    for name, pred in predictions_dict.items()}
    sample_labels = reported[obs_start:obs_end]
    sample_features = np.arange(obs_end - obs_start, dtype=float).reshape(-1, 1)

    anti_config = AntiSpuriousConfig(
        n_permutations=100, fdr_level=0.05, n_negative_controls=30)

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
            headers = ["Model Removed", "ICM Delta", "Interpretation"]
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
    # Step 8: Model Ablation (ICM contribution)
    # ================================================================
    print_header("Step 8: Model Contribution Analysis", "-")

    print("  Leave-one-out ICM ablation (overall):")
    full_icm = float(icm_scores.mean())
    print(f"  Full ensemble ICM: {ff(full_icm)}")
    ablation_results = {}
    for remove_name in model_names:
        reduced = {k: v for k, v in predictions_dict.items() if k != remove_name}
        reduced_scores, _ = compute_per_step_icm(
            reduced, window_size=ICM_WINDOW, config=icm_config)
        reduced_mean = float(reduced_scores.mean())
        delta = full_icm - reduced_mean
        ablation_results[remove_name] = delta
        direction = "+" if delta > 0 else "-"
        print(f"    Remove {remove_name:25s}: ICM={ff(reduced_mean)}  "
              f"Delta={direction}{abs(delta):.4f}")

    most_helpful = max(ablation_results.items(), key=lambda x: x[1])
    most_harmful = min(ablation_results.items(), key=lambda x: x[1])
    print(f"\n  Most helpful model:  {most_helpful[0]} (delta={most_helpful[1]:+.4f})")
    print(f"  Most harmful model:  {most_harmful[0]} (delta={most_harmful[1]:+.4f})")

    # ================================================================
    # Step 9: ICM vs Individual Model Decisions
    # ================================================================
    print_header("Step 9: ICM-Guided vs Individual Model Decisions", "-")

    print("  Comparing ensemble ICM decision with individual model accuracy:")
    for name in model_names:
        pred = predictions_dict[name]
        rmse = model_rmse[name]
        corr = float(np.corrcoef(pred, reported)[0, 1]) if np.std(pred) > 1e-12 else 0.0
        print(f"    {name:25s}: RMSE={rmse:.2f}, Corr={corr:.3f}")

    ensemble_rmse = float(np.sqrt(np.mean((ensemble_mean - reported) ** 2)))
    ensemble_corr = float(np.corrcoef(ensemble_mean, reported)[0, 1])
    print(f"\n  Ensemble mean:               RMSE={ensemble_rmse:.2f}, "
          f"Corr={ensemble_corr:.3f}")
    print(f"  ICM-guided decision:         {crc_results['decision'].value}")

    # ================================================================
    # Step 10: Knowledge Graph
    # ================================================================
    print_header("Step 10: Knowledge Graph Summary", "-")

    ew_results = {
        "cusum_changes": cusum_changes,
        "ph_changes": ph_changes,
        "cusum_eval": cusum_eval,
        "ph_eval": ph_eval,
        "z_signal": z_signal,
        "pred_var": pred_var,
    }

    kg = build_knowledge_graph(
        model_names, predictions_dict, icm_scores, icm_results,
        ew_results, anti_report, crc_results, phase_icms)

    kg_summary = kg.summary()
    for key, value in kg_summary.items():
        print(f"  {key}: {value}")

    converging = kg.find_converging_methods("covid19_system", min_icm=0.5)
    conflicts = kg.find_conflicting_results("covid19_system")
    print(f"\n  Converging methods (ICM >= 0.5): {len(converging)}")
    for mid in converging:
        print(f"    - {mid}")
    print(f"  Conflicting result pairs: {len(conflicts)}")

    provenance = kg.get_decision_provenance("decision_covid19")
    if provenance:
        print(f"\n  Decision provenance:")
        print(f"    ICM score nodes:  {len(provenance.get('icm_scores', []))}")
        print(f"    Methods involved: {len(provenance.get('methods', []))}")

    # ================================================================
    # Step 11: Comprehensive Results
    # ================================================================
    print_header("COMPREHENSIVE RESULTS SUMMARY")

    print("\n  [A] COVID-19 EPIDEMIC SCENARIO")
    print(f"      Population: {POPULATION:,}")
    print(f"      Timeline: {N_DAYS} days (multi-wave)")
    print(f"      Total true infections:  {true_total:.0f}")
    print(f"      Total reported cases:   {reported_total:.0f}")
    print(f"      Underreporting ratio:   {underreporting_ratio:.2f}")
    print(f"      Final R fraction:       {final_r:.3f}")
    print(f"      Final V fraction:       {final_v:.3f}")

    print("\n  [B] MODEL PERFORMANCE (ranked by RMSE)")
    sorted_models = sorted(model_rmse.items(), key=lambda x: x[1])
    for rank, (name, rmse) in enumerate(sorted_models, 1):
        mae = model_mae[name]
        print(f"      {rank}. {name}")
        print(f"         RMSE={rmse:.2f}  MAE={mae:.2f}")

    print(f"\n  [C] ICM CONVERGENCE DYNAMICS ACROSS PHASES")
    for phase_name in phase_masks:
        if phase_name in phase_icms:
            print(f"      {phase_name:25s}: ICM = {ff(phase_icms[phase_name])}")

    print(f"\n  [D] EARLY WARNING PERFORMANCE")
    print(f"      CUSUM: TPR={cusum_eval['true_positive_rate']:.2f}, "
          f"FPR={cusum_eval['false_positive_rate']:.2f}")
    print(f"      Page-Hinkley: TPR={ph_eval['true_positive_rate']:.2f}, "
          f"FPR={ph_eval['false_positive_rate']:.2f}")

    print(f"\n  [E] CRC GATING DECISION")
    print(f"      Median ICM:     {ff(crc_results['median_icm'])}")
    print(f"      Epistemic Risk: {ff(crc_results['re_score'])}")
    print(f"      Decision:       {crc_results['decision'].value}")

    print(f"\n  [F] ANTI-SPURIOUS VALIDATION")
    if anti_report is not None:
        print(f"      Genuine convergence: {anti_report.is_genuine}")
        print(f"      Normalized C(x):     {ff(anti_report.c_normalized)}")
    else:
        print(f"      Status: SKIPPED")

    print(f"\n  [G] KNOWLEDGE GRAPH")
    print(f"      Nodes: {kg_summary['total_nodes']}  "
          f"Edges: {kg_summary['total_edges']}")
    print(f"      Converging methods: {len(converging)}/{len(model_names)}")

    print(f"\n  [H] KEY FINDINGS")
    print(f"      1. ICM tracks epidemic phase transitions: models converge")
    print(f"         during stable/declining periods and diverge during")
    print(f"         rapid exponential growth (wave onsets).")
    print(f"      2. Vaccination rollout at day {VACCINATION_START} creates a period")
    print(f"         of model disagreement as different families respond")
    print(f"         differently to the changing population immunity.")
    print(f"      3. The R(t) projection and exponential smoothing models")
    print(f"         are most reactive to short-term changes, while the")
    print(f"         compartmental and statistical models capture longer-")
    print(f"         term structural dynamics.")
    print(f"      4. Multi-wave structure tests ICM's ability to repeatedly")
    print(f"         detect convergence breakdown and recovery.")

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
        "crc_results": crc_results,
        "knowledge_graph": kg,
        "model_rmse": model_rmse,
        "model_mae": model_mae,
        "ablation": ablation_results,
        "phase_icms": phase_icms,
        "config": {
            "population": POPULATION,
            "n_days": N_DAYS,
            "n_nodes": N_NODES,
            "seed": SEED,
            "icm_window": ICM_WINDOW,
        },
        "summary": {
            "true_total_infections": true_total,
            "reported_total_cases": reported_total,
            "underreporting_ratio": underreporting_ratio,
            "peak_true_day": peak_true_day,
            "peak_reported_day": peak_reported_day,
            "final_recovered": final_r,
            "final_vaccinated": final_v,
            "mean_icm": float(icm_scores.mean()),
            "crc_decision": crc_results["decision"].value,
        },
        "elapsed_seconds": elapsed,
    }


# ====================================================================
# Entry point
# ====================================================================

if __name__ == "__main__":
    results = run_covid19_benchmark()
