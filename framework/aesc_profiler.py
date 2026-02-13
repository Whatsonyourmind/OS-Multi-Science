"""AESC System Profiler -- builds and analyses SystemProfile instances."""

from __future__ import annotations

from framework.types import (
    AgentType,
    Controllability,
    DataRegime,
    Dynamics,
    Feedback,
    NetworkType,
    Observability,
    Role,
    Scale,
    SystemProfile,
)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_profile(
    name: str,
    description: str,
    *,
    scale: Scale = Scale.MESO,
    dynamics: Dynamics = Dynamics.SLOW,
    network: NetworkType = NetworkType.SPARSE,
    agents: AgentType = AgentType.MANY,
    feedback: Feedback = Feedback.MODERATE,
    data_regime: DataRegime = DataRegime.MODERATE,
    controllability: Controllability = Controllability.PARTIAL,
    observability: Observability = Observability.PARTIAL,
    conservation_laws: list[str] | None = None,
    regulatory_context: list[str] | None = None,
) -> SystemProfile:
    """Create a SystemProfile with sensible defaults, then auto-infer roles."""
    profile = SystemProfile(
        name=name,
        description=description,
        scale=scale,
        dynamics=dynamics,
        network=network,
        agents=agents,
        feedback=feedback,
        data_regime=data_regime,
        controllability=controllability,
        observability=observability,
        conservation_laws=conservation_laws or [],
        regulatory_context=regulatory_context or [],
    )
    primary, secondary = infer_roles(profile)
    profile.primary_roles = primary
    profile.secondary_roles = secondary
    return profile


# ---------------------------------------------------------------------------
# Role inference
# ---------------------------------------------------------------------------

def infer_roles(profile: SystemProfile) -> tuple[list[Role], list[Role]]:
    """Infer primary and secondary epistemic roles from system characteristics.

    Scoring heuristic -- each role gets a score; the top scorers become
    primary, the rest secondary (if score > 0).
    """
    scores: dict[Role, float] = {r: 0.0 for r in Role}

    # STRUCTURE: dense or multilayer network
    if profile.network in (NetworkType.DENSE, NetworkType.MULTILAYER):
        scores[Role.STRUCTURE] += 1.0
    elif profile.network == NetworkType.SPARSE:
        scores[Role.STRUCTURE] += 0.3

    # BEHAVIOR: fast or chaotic dynamics
    if profile.dynamics in (Dynamics.FAST, Dynamics.CHAOTIC):
        scores[Role.BEHAVIOR] += 1.0
    elif profile.dynamics == Dynamics.SLOW:
        scores[Role.BEHAVIOR] += 0.3
    # Nonlinear / strong feedback boosts BEHAVIOR
    if profile.feedback in (Feedback.NONLINEAR, Feedback.STRONG):
        scores[Role.BEHAVIOR] += 0.4

    # FORECAST: rich data or streaming
    if profile.data_regime in (DataRegime.RICH, DataRegime.STREAMING):
        scores[Role.FORECAST] += 1.0
    elif profile.data_regime == DataRegime.MODERATE:
        scores[Role.FORECAST] += 0.4

    # INTERVENTION: high controllability
    if profile.controllability == Controllability.HIGH:
        scores[Role.INTERVENTION] += 1.0
    elif profile.controllability == Controllability.PARTIAL:
        scores[Role.INTERVENTION] += 0.4

    # CAUSAL_ID: partial observability + strategic agents
    if profile.observability in (Observability.PARTIAL, Observability.INPUTS_ONLY):
        scores[Role.CAUSAL_ID] += 0.5
    if profile.agents == AgentType.STRATEGIC:
        scores[Role.CAUSAL_ID] += 0.5
    # Delayed feedback also hints at causal identification difficulty
    if profile.feedback == Feedback.DELAYED:
        scores[Role.CAUSAL_ID] += 0.3

    # Split into primary (score >= 0.7) and secondary (0 < score < 0.7)
    primary: list[Role] = []
    secondary: list[Role] = []
    for role, score in scores.items():
        if score >= 0.7:
            primary.append(role)
        elif score > 0:
            secondary.append(role)

    # Guarantee at least one primary role
    if not primary and secondary:
        best = max(secondary, key=lambda r: scores[r])
        secondary.remove(best)
        primary.append(best)

    return primary, secondary


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------

def profile_to_dict(profile: SystemProfile) -> dict:
    """Serialize a SystemProfile to a plain dict (JSON-safe)."""
    return {
        "name": profile.name,
        "description": profile.description,
        "scale": profile.scale.value,
        "dynamics": profile.dynamics.value,
        "network": profile.network.value,
        "agents": profile.agents.value,
        "feedback": profile.feedback.value,
        "data_regime": profile.data_regime.value,
        "controllability": profile.controllability.value,
        "observability": profile.observability.value,
        "conservation_laws": profile.conservation_laws,
        "regulatory_context": profile.regulatory_context,
        "primary_roles": [r.value for r in profile.primary_roles],
        "secondary_roles": [r.value for r in profile.secondary_roles],
    }


def profile_summary(profile: SystemProfile) -> str:
    """Return a human-readable one-paragraph summary of the profile."""
    primary = ", ".join(r.value for r in profile.primary_roles) or "none"
    secondary = ", ".join(r.value for r in profile.secondary_roles) or "none"
    return (
        f"System '{profile.name}': {profile.description}. "
        f"It operates at {profile.scale.value} scale with "
        f"{profile.dynamics.value} dynamics and "
        f"{profile.network.value} network structure. "
        f"Agents are {profile.agents.value}, feedback is "
        f"{profile.feedback.value}, and data regime is "
        f"{profile.data_regime.value}. "
        f"Controllability is {profile.controllability.value} "
        f"with {profile.observability.value} observability. "
        f"Primary epistemic roles: {primary}. "
        f"Secondary roles: {secondary}."
    )


# ---------------------------------------------------------------------------
# Pre-built example profiles
# ---------------------------------------------------------------------------

FINANCIAL_ENERGY_SYSTEM = create_profile(
    name="Financial-Energy Coupled Market",
    description=(
        "Coupled financial and energy markets with strategic traders, "
        "regulatory constraints, and rich high-frequency data streams"
    ),
    scale=Scale.MULTI_SCALE,
    dynamics=Dynamics.FAST,
    network=NetworkType.MULTILAYER,
    agents=AgentType.STRATEGIC,
    feedback=Feedback.NONLINEAR,
    data_regime=DataRegime.STREAMING,
    controllability=Controllability.PARTIAL,
    observability=Observability.PARTIAL,
    conservation_laws=["energy_balance", "capital_conservation"],
    regulatory_context=["market_regulation", "emissions_caps"],
)

EPIDEMIC_NETWORK = create_profile(
    name="Epidemic on Social Network",
    description=(
        "Infectious disease spreading through a heterogeneous social "
        "contact network with partial surveillance data"
    ),
    scale=Scale.MACRO,
    dynamics=Dynamics.FAST,
    network=NetworkType.DENSE,
    agents=AgentType.MANY,
    feedback=Feedback.NONLINEAR,
    data_regime=DataRegime.MODERATE,
    controllability=Controllability.PARTIAL,
    observability=Observability.PARTIAL,
    conservation_laws=["population_conservation"],
    regulatory_context=["public_health_mandates"],
)

SUPPLY_CHAIN = create_profile(
    name="Multi-tier Supply Chain",
    description=(
        "Global multi-layer supplier network with inventory constraints, "
        "lead-time delays, and limited visibility beyond tier-1 suppliers"
    ),
    scale=Scale.MESO,
    dynamics=Dynamics.SLOW,
    network=NetworkType.MULTILAYER,
    agents=AgentType.STRATEGIC,
    feedback=Feedback.DELAYED,
    data_regime=DataRegime.MODERATE,
    controllability=Controllability.PARTIAL,
    observability=Observability.PARTIAL,
    conservation_laws=["material_balance"],
    regulatory_context=["trade_regulations", "customs"],
)
