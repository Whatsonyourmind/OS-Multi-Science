"""Discipline catalog -- 12 scientific methods for the Router to select from."""

from __future__ import annotations

from framework.types import EpistemicFamily, MethodProfile, Role


# ---------------------------------------------------------------------------
# Catalog entries
# ---------------------------------------------------------------------------

def _build_catalog() -> list[MethodProfile]:
    """Construct the full 12-method catalog."""
    return [
        # 1. Agent-Based Modeling
        MethodProfile(
            name="ABM",
            family=EpistemicFamily.AGENT_BASED,
            description=(
                "Agent-based modeling: heterogeneous agents following local "
                "rules produce emergent macro-level phenomena."
            ),
            assumptions=[
                "agents_follow_rules",
                "micro_interactions_drive_macro",
                "bounded_rationality",
            ],
            strengths=[
                "captures_heterogeneity",
                "emergent_behavior",
                "flexible_topology",
            ],
            weaknesses=[
                "computationally_expensive",
                "calibration_difficult",
                "parameter_sensitivity",
            ],
            roles_supported=[Role.BEHAVIOR, Role.STRUCTURE],
            data_requirements="scarce_to_moderate",
            computational_cost="high",
            interpretability="medium",
        ),
        # 2. System Dynamics
        MethodProfile(
            name="System Dynamics",
            family=EpistemicFamily.SYSTEM_DYNAMICS,
            description=(
                "Stock-and-flow models capturing feedback loops and delays "
                "in aggregate system behavior."
            ),
            assumptions=[
                "aggregate_representation",
                "continuous_flows",
                "feedback_loops_dominant",
            ],
            strengths=[
                "captures_feedback",
                "long_horizon_simulation",
                "interpretable_structure",
            ],
            weaknesses=[
                "no_agent_heterogeneity",
                "aggregation_bias",
                "hard_to_validate",
            ],
            roles_supported=[Role.BEHAVIOR, Role.FORECAST],
            data_requirements="scarce_to_moderate",
            computational_cost="low",
            interpretability="high",
        ),
        # 3. Network Science
        MethodProfile(
            name="Network Science",
            family=EpistemicFamily.NETWORK,
            description=(
                "Graph-theoretic analysis of relational structure: centrality, "
                "community detection, diffusion on networks."
            ),
            assumptions=[
                "relational_structure_matters",
                "graph_topology_informative",
            ],
            strengths=[
                "reveals_topology",
                "identifies_key_nodes",
                "scalable_algorithms",
            ],
            weaknesses=[
                "static_snapshot_bias",
                "edge_definition_sensitivity",
            ],
            roles_supported=[Role.STRUCTURE, Role.BEHAVIOR],
            data_requirements="moderate",
            computational_cost="medium",
            interpretability="high",
        ),
        # 4. Epidemiological Models (SIR/SIS)
        MethodProfile(
            name="Epidemiological Models",
            family=EpistemicFamily.EPIDEMIOLOGICAL,
            description=(
                "Compartmental models (SIR, SIS, SEIR) for contagion and "
                "spreading dynamics on populations."
            ),
            assumptions=[
                "compartmental_homogeneity",
                "well_mixed_population",
                "constant_rates",
            ],
            strengths=[
                "analytical_tractability",
                "interpretable_parameters",
                "threshold_analysis",
            ],
            weaknesses=[
                "homogeneity_assumption",
                "limited_behavioral_feedback",
            ],
            roles_supported=[Role.BEHAVIOR, Role.FORECAST],
            data_requirements="scarce_to_moderate",
            computational_cost="low",
            interpretability="high",
        ),
        # 5. Econometric / Statistical
        MethodProfile(
            name="Econometric/Statistical",
            family=EpistemicFamily.STATISTICAL,
            description=(
                "Panel regression, VAR, survival analysis, and classical "
                "statistical inference for causal and predictive tasks."
            ),
            assumptions=[
                "linearity_or_known_functional_form",
                "stationarity",
                "exogeneity",
            ],
            strengths=[
                "well_understood_theory",
                "confidence_intervals",
                "causal_identification",
            ],
            weaknesses=[
                "linearity_limits",
                "stationarity_requirement",
                "endogeneity_bias",
            ],
            roles_supported=[Role.FORECAST, Role.CAUSAL_ID],
            data_requirements="moderate_to_rich",
            computational_cost="low",
            interpretability="high",
        ),
        # 6. Gradient Boosting (XGBoost / LightGBM)
        MethodProfile(
            name="Gradient Boosting",
            family=EpistemicFamily.MACHINE_LEARNING,
            description=(
                "Ensemble of decision trees trained via gradient boosting "
                "(XGBoost, LightGBM) for tabular prediction."
            ),
            assumptions=[
                "tabular_features_available",
                "iid_or_weakly_dependent_samples",
            ],
            strengths=[
                "state_of_art_tabular",
                "feature_importance",
                "handles_missing_data",
            ],
            weaknesses=[
                "limited_sequence_modeling",
                "no_native_uncertainty",
                "opaque_interactions",
            ],
            roles_supported=[Role.FORECAST],
            data_requirements="rich",
            computational_cost="medium",
            interpretability="medium",
        ),
        # 7. Deep Learning (LSTM / Transformer)
        MethodProfile(
            name="Deep Learning",
            family=EpistemicFamily.MACHINE_LEARNING,
            description=(
                "Recurrent and attention-based neural networks (LSTM, "
                "Transformer) for sequential and temporal data."
            ),
            assumptions=[
                "large_training_set",
                "temporal_patterns_learnable",
            ],
            strengths=[
                "captures_long_range_dependencies",
                "flexible_architecture",
                "automatic_feature_learning",
            ],
            weaknesses=[
                "data_hungry",
                "low_interpretability",
                "computationally_expensive",
            ],
            roles_supported=[Role.FORECAST, Role.BEHAVIOR],
            data_requirements="rich",
            computational_cost="high",
            interpretability="low",
        ),
        # 8. Graph Neural Networks (GNN)
        MethodProfile(
            name="GNN",
            family=EpistemicFamily.MACHINE_LEARNING,
            description=(
                "Graph neural networks that learn over relational structure: "
                "node/edge/graph-level prediction."
            ),
            assumptions=[
                "graph_structure_available",
                "message_passing_sufficient",
            ],
            strengths=[
                "leverages_topology",
                "inductive_on_new_graphs",
                "combines_features_and_structure",
            ],
            weaknesses=[
                "over_smoothing",
                "needs_graph_data",
                "limited_interpretability",
            ],
            roles_supported=[Role.STRUCTURE, Role.FORECAST],
            data_requirements="rich",
            computational_cost="high",
            interpretability="low",
        ),
        # 9. Operations Research (MPC, optimisation)
        MethodProfile(
            name="Operations Research",
            family=EpistemicFamily.OPTIMIZATION,
            description=(
                "Model predictive control, linear/integer programming, and "
                "stochastic optimization for decision-making."
            ),
            assumptions=[
                "objective_function_known",
                "constraints_specifiable",
                "controllable_variables",
            ],
            strengths=[
                "optimal_decisions",
                "constraint_handling",
                "well_developed_solvers",
            ],
            weaknesses=[
                "model_mismatch_risk",
                "scalability_limits",
                "assumes_known_model",
            ],
            roles_supported=[Role.INTERVENTION],
            data_requirements="moderate",
            computational_cost="medium",
            interpretability="high",
        ),
        # 10. Causal Inference (DiD, IV, synthetic control)
        MethodProfile(
            name="Causal Inference",
            family=EpistemicFamily.CAUSAL,
            description=(
                "Difference-in-differences, instrumental variables, and "
                "synthetic control for causal effect estimation."
            ),
            assumptions=[
                "parallel_trends_or_exclusion_restriction",
                "no_unmeasured_confounders",
                "SUTVA",
            ],
            strengths=[
                "causal_claims",
                "policy_relevant",
                "well_understood_assumptions",
            ],
            weaknesses=[
                "strong_assumptions",
                "limited_to_specific_designs",
                "external_validity",
            ],
            roles_supported=[Role.CAUSAL_ID, Role.INTERVENTION],
            data_requirements="moderate_to_rich",
            computational_cost="low",
            interpretability="high",
        ),
        # 11. Topological Data Analysis (persistent homology)
        MethodProfile(
            name="TDA",
            family=EpistemicFamily.TOPOLOGICAL,
            description=(
                "Persistent homology and topological summaries that capture "
                "shape and connectivity invariants of data."
            ),
            assumptions=[
                "topology_captures_signal",
                "appropriate_filtration",
            ],
            strengths=[
                "coordinate_free",
                "robust_to_noise",
                "captures_global_shape",
            ],
            weaknesses=[
                "high_computational_cost",
                "interpretation_requires_expertise",
                "limited_predictive_power_alone",
            ],
            roles_supported=[Role.STRUCTURE],
            data_requirements="moderate",
            computational_cost="high",
            interpretability="medium",
        ),
        # 12. Baseline / Naive
        MethodProfile(
            name="Baseline",
            family=EpistemicFamily.BASELINE,
            description=(
                "Simple baselines: linear trend extrapolation, last-value, "
                "historical mean. Essential reference points."
            ),
            assumptions=["stationarity_or_linear_trend"],
            strengths=[
                "trivially_interpretable",
                "near_zero_cost",
                "sanity_check",
            ],
            weaknesses=[
                "no_complex_dynamics",
                "poor_on_nonlinear_systems",
            ],
            roles_supported=[Role.FORECAST],
            data_requirements="scarce",
            computational_cost="low",
            interpretability="high",
        ),
    ]


# Module-level singleton
_CATALOG: list[MethodProfile] | None = None


def get_catalog() -> list[MethodProfile]:
    """Return the full 12-method catalog (cached)."""
    global _CATALOG  # noqa: PLW0603
    if _CATALOG is None:
        _CATALOG = _build_catalog()
    return list(_CATALOG)


# ---------------------------------------------------------------------------
# Queries
# ---------------------------------------------------------------------------

def get_methods_by_role(role: Role) -> list[MethodProfile]:
    """Return methods that support the given epistemic role."""
    return [m for m in get_catalog() if role in m.roles_supported]


def get_methods_by_family(family: EpistemicFamily) -> list[MethodProfile]:
    """Return methods belonging to the given epistemic family."""
    return [m for m in get_catalog() if m.family == family]


# ---------------------------------------------------------------------------
# Epistemic distance
# ---------------------------------------------------------------------------

def epistemic_distance(m1: MethodProfile, m2: MethodProfile) -> float:
    """Measure epistemic distance between two methods in [0, 1].

    Components (weights sum to 1.0):
      - Family difference   : 0.4  (1 if different family, 0 if same)
      - Assumption overlap   : 0.3  (Jaccard distance of assumption sets)
      - Role overlap         : 0.3  (Jaccard distance of role sets)
    """
    # Family component
    family_dist = 0.0 if m1.family == m2.family else 1.0

    # Assumption component (Jaccard distance)
    a1 = set(m1.assumptions)
    a2 = set(m2.assumptions)
    if a1 or a2:
        assumption_dist = 1.0 - len(a1 & a2) / len(a1 | a2)
    else:
        assumption_dist = 0.0

    # Role component (Jaccard distance)
    r1 = set(m1.roles_supported)
    r2 = set(m2.roles_supported)
    if r1 or r2:
        role_dist = 1.0 - len(r1 & r2) / len(r1 | r2)
    else:
        role_dist = 0.0

    return 0.4 * family_dist + 0.3 * assumption_dist + 0.3 * role_dist
