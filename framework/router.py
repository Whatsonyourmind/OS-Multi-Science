"""Intelligent Router -- selects a diverse, well-fitting method kit."""

from __future__ import annotations

import numpy as np

from framework.catalog import epistemic_distance, get_catalog
from framework.config import RouterConfig
from framework.types import (
    AgentType,
    Controllability,
    DataRegime,
    Dynamics,
    EpistemicFamily,
    Feedback,
    MethodProfile,
    NetworkType,
    Observability,
    Role,
    RouterSelection,
    Scale,
    SystemProfile,
)


# ---------------------------------------------------------------------------
# Fit scoring
# ---------------------------------------------------------------------------

def compute_fit(profile: SystemProfile, method: MethodProfile) -> float:
    """Score how well *method* fits *profile*.  Returns a float in [0, 1].

    Sub-scores (equal weight by default):
      1. Role match         -- fraction of primary roles supported
      2. Scale compatibility -- heuristic per method family
      3. Data compatibility  -- method data needs vs available data regime
      4. Network compat.     -- does the method need / leverage graphs?
      5. Complexity match    -- simple methods preferred for simple systems
    """
    scores: list[float] = []

    # 1. Role match: fraction of primary roles covered
    if profile.primary_roles:
        covered = sum(
            1 for r in profile.primary_roles if r in method.roles_supported
        )
        scores.append(covered / len(profile.primary_roles))
    else:
        scores.append(0.5)

    # 2. Scale compatibility
    scale_score = _scale_fit(profile.scale, method)
    scores.append(scale_score)

    # 3. Data regime compatibility
    data_score = _data_fit(profile.data_regime, method)
    scores.append(data_score)

    # 4. Network compatibility
    net_score = _network_fit(profile.network, method)
    scores.append(net_score)

    # 5. Complexity match
    complexity_score = _complexity_fit(profile, method)
    scores.append(complexity_score)

    return float(np.clip(np.mean(scores), 0.0, 1.0))


def _scale_fit(scale: Scale, method: MethodProfile) -> float:
    """Heuristic: ABM good for micro, SD for macro, ML flexible."""
    family = method.family
    if scale == Scale.MICRO:
        if family == EpistemicFamily.AGENT_BASED:
            return 1.0
        if family in (EpistemicFamily.SYSTEM_DYNAMICS, EpistemicFamily.EPIDEMIOLOGICAL):
            return 0.3
        return 0.6
    if scale == Scale.MACRO:
        if family in (EpistemicFamily.SYSTEM_DYNAMICS, EpistemicFamily.EPIDEMIOLOGICAL,
                       EpistemicFamily.STATISTICAL):
            return 1.0
        if family == EpistemicFamily.AGENT_BASED:
            return 0.4
        return 0.6
    if scale == Scale.MULTI_SCALE:
        # Most methods get partial credit; network/ML get more
        if family in (EpistemicFamily.NETWORK, EpistemicFamily.MACHINE_LEARNING):
            return 0.9
        if family == EpistemicFamily.AGENT_BASED:
            return 0.8
        return 0.6
    # MESO -- neutral
    return 0.7


def _data_fit(regime: DataRegime, method: MethodProfile) -> float:
    """ML/DL need rich data; ABM/SD work with scarce."""
    needs = method.data_requirements
    if regime in (DataRegime.RICH, DataRegime.STREAMING):
        return 1.0  # plenty of data -- everything works
    if regime == DataRegime.MODERATE:
        if needs in ("rich",):
            return 0.4
        return 0.8
    # SCARCE
    if needs in ("rich",):
        return 0.1
    if needs in ("moderate_to_rich",):
        return 0.3
    return 0.9


def _network_fit(network: NetworkType, method: MethodProfile) -> float:
    """GNN / Network science need graphs; penalise if none."""
    family = method.family
    if network == NetworkType.NONE:
        if family == EpistemicFamily.NETWORK:
            return 0.1
        if method.name == "GNN":
            return 0.1
        return 0.8
    if network in (NetworkType.DENSE, NetworkType.MULTILAYER):
        if family == EpistemicFamily.NETWORK or method.name == "GNN":
            return 1.0
        return 0.6
    # SPARSE
    if family == EpistemicFamily.NETWORK or method.name == "GNN":
        return 0.8
    return 0.7


def _complexity_fit(profile: SystemProfile, method: MethodProfile) -> float:
    """Simple systems favour interpretable methods; complex ones favour ML."""
    complexity_indicators = 0
    if profile.dynamics in (Dynamics.FAST, Dynamics.CHAOTIC):
        complexity_indicators += 1
    if profile.feedback in (Feedback.NONLINEAR, Feedback.STRONG):
        complexity_indicators += 1
    if profile.network in (NetworkType.DENSE, NetworkType.MULTILAYER):
        complexity_indicators += 1
    if profile.agents in (AgentType.MANY, AgentType.STRATEGIC):
        complexity_indicators += 1

    high_complexity = complexity_indicators >= 3

    if high_complexity:
        # Reward powerful methods
        if method.computational_cost == "high":
            return 0.9
        if method.computational_cost == "medium":
            return 0.7
        return 0.5
    else:
        # Reward simple / interpretable methods
        if method.interpretability == "high":
            return 0.9
        if method.interpretability == "medium":
            return 0.7
        return 0.5


# ---------------------------------------------------------------------------
# Diversity matrix
# ---------------------------------------------------------------------------

def compute_diversity_matrix(methods: list[MethodProfile]) -> np.ndarray:
    """Pairwise epistemic distance matrix (symmetric, zero diagonal)."""
    n = len(methods)
    mat = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = epistemic_distance(methods[i], methods[j])
            mat[i, j] = d
            mat[j, i] = d
    return mat


# ---------------------------------------------------------------------------
# Kit selection (greedy diversity-weighted)
# ---------------------------------------------------------------------------

def select_kit(
    profile: SystemProfile,
    catalog: list[MethodProfile] | None = None,
    config: RouterConfig | None = None,
) -> RouterSelection:
    """Select a diverse, high-fit method kit for the given system profile.

    Algorithm
    ---------
    1. Score every method by fit.
    2. Seed with the highest-fit method.
    3. Greedily add the method that maximises
       ``(1 - w) * fit + w * mean_diversity_to_selected``
       where *w* = ``config.diversity_weight``.
    4. Stop at *max_methods* or when marginal gain < 0.05.
    5. Enforce *min_methods* and *require_interpretable*.
    """
    if catalog is None:
        catalog = get_catalog()
    if config is None:
        config = RouterConfig()

    w = config.diversity_weight

    # Step 1 -- score every method
    fit_scores: dict[str, float] = {}
    for m in catalog:
        fit_scores[m.name] = compute_fit(profile, m)

    # Step 2 -- seed
    sorted_methods = sorted(catalog, key=lambda m: fit_scores[m.name], reverse=True)
    selected: list[MethodProfile] = [sorted_methods[0]]
    remaining = list(sorted_methods[1:])

    # Step 3 -- greedy expansion
    prev_score = fit_scores[selected[0].name]

    while remaining and len(selected) < config.max_methods:
        best_candidate = None
        best_combined = -1.0

        for candidate in remaining:
            fit_val = fit_scores[candidate.name]
            # Average epistemic distance to already-selected methods
            avg_div = float(np.mean(
                [epistemic_distance(candidate, s) for s in selected]
            ))
            combined = (1.0 - w) * fit_val + w * avg_div
            if combined > best_combined:
                best_combined = combined
                best_candidate = candidate

        # Step 4 -- stopping criterion (after min satisfied)
        if len(selected) >= config.min_methods:
            marginal = best_combined - prev_score
            if marginal < -0.05:
                break

        if best_candidate is not None:
            selected.append(best_candidate)
            remaining.remove(best_candidate)
            prev_score = best_combined

    # Ensure min_methods
    while len(selected) < config.min_methods and remaining:
        next_best = max(remaining, key=lambda m: fit_scores[m.name])
        selected.append(next_best)
        remaining.remove(next_best)

    # Step 5 -- ensure at least one interpretable method
    if config.require_interpretable:
        has_interpretable = any(m.interpretability == "high" for m in selected)
        if not has_interpretable:
            interpretable = [
                m for m in remaining if m.interpretability == "high"
            ]
            if interpretable:
                best_interp = max(
                    interpretable, key=lambda m: fit_scores[m.name]
                )
                # Replace lowest-fit non-interpretable if at max
                if len(selected) >= config.max_methods:
                    worst = min(selected, key=lambda m: fit_scores[m.name])
                    selected.remove(worst)
                selected.append(best_interp)

    # Build results
    div_matrix = compute_diversity_matrix(selected)
    n = len(selected)
    if n > 1:
        avg_div = float(div_matrix.sum() / (n * (n - 1)))
    else:
        avg_div = 0.0

    justifications = _build_justifications(profile, selected, fit_scores)

    return RouterSelection(
        selected_methods=selected,
        fit_scores={m.name: fit_scores[m.name] for m in selected},
        diversity_matrix=div_matrix,
        justifications=justifications,
        total_fit=sum(fit_scores[m.name] for m in selected),
        avg_diversity=avg_div,
    )


def _build_justifications(
    profile: SystemProfile,
    selected: list[MethodProfile],
    fit_scores: dict[str, float],
) -> dict[str, str]:
    """One-line justification per selected method."""
    justifications: dict[str, str] = {}
    for m in selected:
        roles = ", ".join(r.value for r in m.roles_supported)
        justifications[m.name] = (
            f"Fit={fit_scores[m.name]:.2f}. "
            f"Covers roles: {roles}. "
            f"Family: {m.family.value}. "
            f"Interpretability: {m.interpretability}."
        )
    return justifications


# ---------------------------------------------------------------------------
# Decision cards
# ---------------------------------------------------------------------------

def generate_decision_cards(
    selection: RouterSelection,
    profile: SystemProfile,
) -> list[dict]:
    """Create a decision card dict for each selected method.

    Each card contains:
      - method: name
      - why_selected: justification text
      - roles_covered: list of role values
      - key_assumptions: list of assumption strings
      - failure_modes: list of weakness strings
      - contribution: what this method uniquely adds to the kit
    """
    cards: list[dict] = []
    all_roles_covered: set[str] = set()

    for m in selection.selected_methods:
        roles_here = [r.value for r in m.roles_supported]
        new_roles = set(roles_here) - all_roles_covered
        all_roles_covered.update(roles_here)

        contribution = (
            f"Adds new epistemic perspectives: {', '.join(new_roles)}"
            if new_roles
            else (
                f"Reinforces existing coverage with a "
                f"{m.family.value} viewpoint"
            )
        )

        cards.append({
            "method": m.name,
            "why_selected": selection.justifications.get(m.name, ""),
            "roles_covered": roles_here,
            "key_assumptions": m.assumptions,
            "failure_modes": m.weaknesses,
            "contribution": contribution,
        })

    return cards
