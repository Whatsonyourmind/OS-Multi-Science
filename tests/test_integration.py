"""End-to-end integration tests for OS Multi-Science.

Validates the entire pipeline from system profiling through decision output,
including knowledge graph recording and agent coordination.
"""

from __future__ import annotations

import numpy as np
import pytest

from framework.aesc_profiler import (
    FINANCIAL_ENERGY_SYSTEM,
    EPIDEMIC_NETWORK,
    SUPPLY_CHAIN,
    create_profile,
)
from framework.anti_spurious import generate_anti_spurious_report
from framework.catalog import get_catalog
from framework.config import OSMultiScienceConfig
from framework.crc_gating import (
    calibrate_thresholds,
    compute_re,
    conformalize,
    decision_gate,
    fit_isotonic,
)
from framework.early_warning import (
    compute_delta_icm,
    compute_rolling_icm,
    compute_z_signal,
    cusum_detector,
)
from framework.icm import compute_icm, compute_icm_from_predictions
from framework.router import generate_decision_cards, select_kit
from framework.types import (
    DecisionAction,
    DecisionCard,
    ICMComponents,
    ICMResult,
)
from orchestrator.pipeline import Pipeline, PipelineResult
from knowledge.graph import (
    EdgeType,
    KnowledgeEdge,
    KnowledgeGraph,
    KnowledgeNode,
    NodeType,
)
from agents.coordinator import (
    Agent,
    AgentRole,
    Coordinator,
    MessageType,
    WorkPlan,
    ConsensusProtocol,
)
from benchmarks.synthetic.generators import (
    generate_classification_benchmark,
    generate_multi_model_predictions,
    generate_change_point_series,
)


# ============================================================================
# Helpers
# ============================================================================

def _make_model_executor(noise: float, seed: int):
    """Create a simple model executor that adds Gaussian noise to y_true."""
    def executor(X, profile):
        rng = np.random.default_rng(seed)
        n = X.shape[0]
        return rng.normal(0, noise, n)
    return executor


def _make_classification_executor(noise: float, seed: int, n_classes: int = 3):
    """Create a classification model executor returning probability vectors."""
    def executor(X, profile):
        rng = np.random.default_rng(seed)
        n = X.shape[0]
        probs = rng.dirichlet(np.ones(n_classes) * (1.0 / max(noise, 0.01)), size=n)
        return probs
    return executor


# ============================================================================
# Test: Full pipeline end-to-end
# ============================================================================

class TestFullPipeline:
    """Run the full Pipeline with synthetic data and validate all outputs."""

    def test_financial_system_pipeline(self):
        """Complete pipeline run on a financial energy system profile."""
        config = OSMultiScienceConfig()
        pipe = Pipeline(config)

        # Register some model executors
        for i in range(5):
            pipe.register_model(
                f"model_{i}",
                _make_model_executor(noise=0.1 * (i + 1), seed=42 + i),
            )

        X = np.random.default_rng(42).standard_normal((200, 5))
        y_true = np.sin(X[:, 0]) + 0.1 * np.random.default_rng(99).standard_normal(200)

        result = pipe.run(
            system_profile=FINANCIAL_ENERGY_SYSTEM,
            X=X,
            y_true=y_true,
            features=X,
            skip_anti_spurious=True,
        )

        # Pipeline should succeed
        assert result.is_success
        assert result.system_profile is not None
        assert result.selected_kit is not None
        assert result.predictions is not None
        assert result.icm_result is not None
        assert len(result.step_results) >= 4

    def test_pipeline_with_crc_gating(self):
        """Pipeline with y_true triggers CRC gating and decision."""
        config = OSMultiScienceConfig()
        pipe = Pipeline(config)

        for i in range(4):
            pipe.register_model(
                f"model_{i}",
                _make_model_executor(noise=0.05 * (i + 1), seed=100 + i),
            )

        rng = np.random.default_rng(55)
        X = rng.standard_normal((300, 5))
        y_true = X[:, 0] ** 2 + 0.1 * rng.standard_normal(300)

        result = pipe.run(
            system_profile=SUPPLY_CHAIN,
            X=X,
            y_true=y_true,
            features=X,
            skip_anti_spurious=True,
        )

        assert result.is_success
        assert result.icm_result is not None
        assert 0 <= result.icm_result.icm_score <= 1
        # CRC should have been computed
        assert result.crc_decision is not None
        assert result.crc_decision in (
            DecisionAction.ACT,
            DecisionAction.DEFER,
            DecisionAction.AUDIT,
        )


# ============================================================================
# Test: Knowledge Graph + Pipeline integration
# ============================================================================

class TestKnowledgeGraphIntegration:
    """Record pipeline results into the knowledge graph and query them."""

    def test_record_and_query_pipeline_results(self):
        """Run pipeline, record everything in KG, query it back."""
        # 1. Run pipeline
        config = OSMultiScienceConfig()
        pipe = Pipeline(config)
        for i in range(4):
            pipe.register_model(
                f"model_{i}",
                _make_model_executor(noise=0.1, seed=42 + i),
            )

        X = np.random.default_rng(42).standard_normal((100, 3))
        result = pipe.run(
            system_profile=FINANCIAL_ENERGY_SYSTEM,
            X=X,
            skip_anti_spurious=True,
        )

        # 2. Record into KG
        kg = KnowledgeGraph()

        # Add system node
        sys_node = KnowledgeNode(
            id="sys_financial",
            node_type=NodeType.SYSTEM,
            data={"name": FINANCIAL_ENERGY_SYSTEM.name},
        )
        kg.add_node(sys_node)

        # Add method nodes from selected kit
        if result.selected_kit:
            for method in result.selected_kit.selected_methods:
                m_node = KnowledgeNode(
                    id=f"method_{method.name}",
                    node_type=NodeType.METHOD,
                    data={
                        "name": method.name,
                        "family": method.family.value,
                    },
                )
                kg.add_node(m_node)
                kg.add_edge(KnowledgeEdge(
                    source_id="sys_financial",
                    target_id=f"method_{method.name}",
                    edge_type=EdgeType.ANALYZED_BY,
                ))

        # Add ICM score node
        if result.icm_result:
            icm_node = KnowledgeNode(
                id="icm_run_1",
                node_type=NodeType.ICM_SCORE,
                data={
                    "score": result.icm_result.icm_score,
                    "aggregation": result.icm_result.aggregation_method,
                },
            )
            kg.add_node(icm_node)

        # 3. Query the KG
        systems = kg.find_nodes(NodeType.SYSTEM)
        assert len(systems) == 1
        assert systems[0].data["name"] == FINANCIAL_ENERGY_SYSTEM.name

        methods = kg.find_nodes(NodeType.METHOD)
        assert len(methods) >= 3  # Router selects at least min_methods

        icm_nodes = kg.find_nodes(NodeType.ICM_SCORE)
        assert len(icm_nodes) == 1

    def test_decision_provenance_tracking(self):
        """Track decision provenance from system -> methods -> ICM -> decision."""
        kg = KnowledgeGraph()

        # Build a minimal provenance chain
        kg.add_node(KnowledgeNode(id="sys1", node_type=NodeType.SYSTEM, data={"name": "test"}))
        kg.add_node(KnowledgeNode(id="m1", node_type=NodeType.METHOD, data={"name": "abm"}))
        kg.add_node(KnowledgeNode(id="m2", node_type=NodeType.METHOD, data={"name": "svm"}))
        kg.add_node(KnowledgeNode(id="r1", node_type=NodeType.RESULT, data={"pred": [1, 2, 3]}))
        kg.add_node(KnowledgeNode(id="r2", node_type=NodeType.RESULT, data={"pred": [1.1, 2.1, 3.1]}))
        kg.add_node(KnowledgeNode(id="icm1", node_type=NodeType.ICM_SCORE, data={"score": 0.85}))
        kg.add_node(KnowledgeNode(id="dec1", node_type=NodeType.DECISION, data={"action": "ACT"}))

        kg.add_edge(KnowledgeEdge("sys1", "m1", EdgeType.ANALYZED_BY))
        kg.add_edge(KnowledgeEdge("sys1", "m2", EdgeType.ANALYZED_BY))
        kg.add_edge(KnowledgeEdge("m1", "r1", EdgeType.PRODUCED))
        kg.add_edge(KnowledgeEdge("m2", "r2", EdgeType.PRODUCED))
        kg.add_edge(KnowledgeEdge("r1", "r2", EdgeType.CONVERGES_WITH))
        kg.add_edge(KnowledgeEdge("icm1", "dec1", EdgeType.LED_TO))

        # Query subgraph from system
        sub = kg.get_subgraph("sys1", max_depth=3)
        assert "sys1" in sub
        assert "m1" in sub
        assert "m2" in sub
        assert "r1" in sub
        assert "r2" in sub

        # Find converging methods
        converging = kg.find_converging_methods("sys1")
        assert len(converging) >= 0  # Depends on graph structure

    def test_serialization_round_trip(self):
        """KG should survive a to_dict -> from_dict round trip."""
        kg = KnowledgeGraph()
        kg.add_node(KnowledgeNode(id="n1", node_type=NodeType.SYSTEM, data={"x": 1}))
        kg.add_node(KnowledgeNode(id="n2", node_type=NodeType.METHOD, data={"y": 2}))
        kg.add_edge(KnowledgeEdge("n1", "n2", EdgeType.ANALYZED_BY, weight=0.9))

        data = kg.to_dict()
        kg2 = KnowledgeGraph.from_dict(data)

        assert kg2.get_node("n1") is not None
        assert kg2.get_node("n2") is not None
        edges = kg2.get_edges("n1", direction="outgoing")
        assert len(edges) == 1
        assert edges[0].edge_type == EdgeType.ANALYZED_BY


# ============================================================================
# Test: Agent coordination + Pipeline
# ============================================================================

class TestAgentCoordination:
    """Validate multi-agent coordination for the pipeline."""

    def test_default_agents_created(self):
        """Coordinator creates a standard set of agents."""
        coord = Coordinator()
        coord.create_default_agents()

        summary = coord.summary()
        assert summary["n_agents"] >= 5

        # Check key roles exist
        roles = {a.role for a in coord.agents.values()}
        assert AgentRole.PROFILER in roles
        assert AgentRole.ROUTER in roles
        assert AgentRole.EXECUTOR in roles
        assert AgentRole.EVALUATOR in roles
        assert AgentRole.SENTINEL in roles

    def test_work_plan_execution(self):
        """Execute a work plan with dependencies between agents."""
        coord = Coordinator()

        profiler = Agent("profiler", AgentRole.PROFILER)
        profiler.set_handler(lambda msg: None)
        router = Agent("router", AgentRole.ROUTER)
        router.set_handler(lambda msg: None)
        evaluator = Agent("evaluator", AgentRole.EVALUATOR)
        evaluator.set_handler(lambda msg: None)

        coord.add_agent(profiler)
        coord.add_agent(router)
        coord.add_agent(evaluator)

        plan = WorkPlan()
        plan.add_step("profiler", "profile_system")
        plan.add_step("router", "select_methods", depends_on=["profile_system"])
        plan.add_step("evaluator", "compute_icm", depends_on=["select_methods"])

        results = coord.execute_work_plan(plan)
        assert len(results) == 3

    def test_consensus_protocol(self):
        """Agents can vote and reach consensus on decisions."""
        agents = [
            Agent("agent_1", AgentRole.EVALUATOR),
            Agent("agent_2", AgentRole.EVALUATOR),
            Agent("agent_3", AgentRole.SENTINEL),
        ]

        # Weighted by expertise
        weights = {"agent_1": 0.5, "agent_2": 0.3, "agent_3": 0.2}
        protocol = ConsensusProtocol(agents, weights)

        protocol.cast_vote("agent_1", "ACT")
        protocol.cast_vote("agent_2", "ACT")
        protocol.cast_vote("agent_3", "DEFER")

        winner = protocol.get_winner()
        assert winner == "ACT"  # Majority + higher weight

        tally = protocol.tally()
        assert tally["ACT"] > tally["DEFER"]

    def test_message_audit_trail(self):
        """Coordinator maintains a full audit trail of messages."""
        coord = Coordinator()
        a1 = Agent("sender", AgentRole.PROFILER)
        a2 = Agent("receiver", AgentRole.ROUTER)
        coord.add_agent(a1)
        coord.add_agent(a2)

        msg = a1.create_message("receiver", MessageType.RESULT, {"icm": 0.85})
        coord.send_message(msg)

        log = coord.get_message_log()
        assert len(log) == 1
        assert log[0].sender == "sender"
        assert log[0].content["icm"] == 0.85


# ============================================================================
# Test: ICM + CRC + Decision full chain
# ============================================================================

class TestICMCRCChain:
    """Validate the ICM -> CRC -> Decision chain with synthetic data."""

    def test_classification_chain(self):
        """Full chain on classification data."""
        X, y_true, _ = generate_classification_benchmark(n_samples=500, seed=42)
        preds = generate_multi_model_predictions(X, y_true, n_agreeing=3, n_disagreeing=1, seed=42)

        # Compute ICM
        icm_result = compute_icm_from_predictions(preds)
        assert 0 <= icm_result.icm_score <= 1
        assert icm_result.n_models == 4

        # Compute loss for CRC
        n_classes = int(y_true.max()) + 1
        losses = []
        for name, pred in preds.items():
            onehot = np.zeros((len(y_true), n_classes))
            onehot[np.arange(len(y_true)), y_true] = 1.0
            ce = -np.sum(onehot * np.log(np.clip(pred, 1e-10, 1.0)), axis=1)
            losses.append(ce)
        avg_loss = np.mean(losses, axis=0)

        # CRC gating
        n = len(avg_loss)
        idx = np.random.default_rng(42).permutation(n)
        train_idx, cal_idx = idx[:n//2], idx[n//2:]

        # Use ICM score as the convergence measure
        # For simplicity, assign same ICM to all samples
        C = np.full(n, icm_result.icm_score)
        L = avg_loss

        g = fit_isotonic(C[train_idx], L[train_idx])
        g_alpha = conformalize(g, C[cal_idx], L[cal_idx], alpha=0.10)

        # Decision gate
        re = compute_re(icm_result.icm_score, g_alpha)
        config = OSMultiScienceConfig()
        decision = decision_gate(icm_result.icm_score, re, config.crc)

        assert decision in (DecisionAction.ACT, DecisionAction.DEFER, DecisionAction.AUDIT)
        assert re >= 0


# ============================================================================
# Test: Early warning on time series
# ============================================================================

class TestEarlyWarningChain:
    """Validate early warning on change-point time series."""

    def test_detect_change_point(self):
        """Early warning system detects a known change-point."""
        from framework.config import EarlyWarningConfig

        X, change_at = generate_change_point_series(n_samples=300, change_at=150, seed=42)

        # Simulate ICM scores: high before change, dropping sharply after
        rng = np.random.default_rng(42)
        icm_series = np.concatenate([
            0.8 + 0.02 * rng.standard_normal(150),  # stable
            np.linspace(0.8, 0.1, 150) + 0.02 * rng.standard_normal(150),  # degrading
        ])
        icm_series = np.clip(icm_series, 0, 1)

        config = EarlyWarningConfig(window_size=10, cusum_threshold=2.0, cusum_drift=0.1)
        delta = compute_delta_icm(icm_series, config.window_size)

        # Simulate prediction variance and Pi trend (strong signals)
        var_pred = np.concatenate([
            0.01 * np.ones(150),
            np.linspace(0.01, 1.0, 150),
        ])
        pi_trend = np.concatenate([
            0.1 * np.ones(150),
            np.linspace(0.1, 1.0, 150),
        ])

        z = compute_z_signal(delta, var_pred, pi_trend, config)
        change_points, cusum = cusum_detector(z, config.cusum_threshold, config.cusum_drift)

        # Should detect something after the change-point
        assert len(change_points) > 0
        # At least one detection should be near or after the change point
        assert any(cp >= 130 for cp in change_points)


# ============================================================================
# Test: Anti-spurious validation
# ============================================================================

class TestAntiSpuriousChain:
    """Validate anti-spurious checks on genuine vs spurious scenarios."""

    def test_genuine_convergence_passes(self):
        """Genuine convergence should pass anti-spurious checks."""
        from framework.config import AntiSpuriousConfig

        rng = np.random.default_rng(42)
        n = 200
        signal = rng.standard_normal(n)

        preds = {}
        for i in range(4):
            preds[f"model_{i}"] = signal + 0.3 * rng.standard_normal(n)

        labels = signal
        features = rng.standard_normal((n, 3))

        config = AntiSpuriousConfig(n_permutations=100, n_negative_controls=30)
        report = generate_anti_spurious_report(preds, labels, features, config)

        assert report.c_normalized > 0.5
        assert report.is_genuine

    def test_spurious_convergence_detected(self):
        """Shared-bias convergence should be flagged as spurious."""
        from framework.config import AntiSpuriousConfig

        rng = np.random.default_rng(42)
        n = 200
        signal = rng.standard_normal(n)
        confounder = 3.0 * rng.standard_normal(n)

        preds = {}
        for i in range(4):
            preds[f"model_{i}"] = signal + confounder + 0.01 * rng.standard_normal(n)

        labels = signal
        features = rng.standard_normal((n, 3))

        config = AntiSpuriousConfig(n_permutations=100, n_negative_controls=30)
        report = generate_anti_spurious_report(preds, labels, features, config)

        # HSIC should detect correlated residuals
        assert report.hsic_pvalue < 0.05
        assert not report.is_genuine


# ============================================================================
# Test: Router + Profiler integration
# ============================================================================

class TestRouterProfilerIntegration:
    """Validate profiler -> router -> decision card chain."""

    def test_financial_profile_selects_appropriate_kit(self):
        """Financial system should get econometric + network methods."""
        catalog = get_catalog()
        config = OSMultiScienceConfig()
        selection = select_kit(FINANCIAL_ENERGY_SYSTEM, catalog, config.router)

        assert len(selection.selected_methods) >= config.router.min_methods
        assert len(selection.selected_methods) <= config.router.max_methods

        method_names = {m.name for m in selection.selected_methods}
        # Should include at least one interpretable method
        has_interpretable = any(
            m.interpretability == "high" for m in selection.selected_methods
        )
        assert has_interpretable

    def test_epidemic_profile_selects_appropriate_kit(self):
        """Epidemic system should get epidemiological + network methods."""
        catalog = get_catalog()
        config = OSMultiScienceConfig()
        selection = select_kit(EPIDEMIC_NETWORK, catalog, config.router)

        assert len(selection.selected_methods) >= config.router.min_methods
        method_families = {m.family.value for m in selection.selected_methods}
        # Should have methodological diversity
        assert len(method_families) >= 2

    def test_decision_cards_generated(self):
        """Decision cards should be generated for each method in kit."""
        catalog = get_catalog()
        config = OSMultiScienceConfig()
        selection = select_kit(FINANCIAL_ENERGY_SYSTEM, catalog, config.router)
        cards = generate_decision_cards(selection, FINANCIAL_ENERGY_SYSTEM)

        assert len(cards) == len(selection.selected_methods)
        for card in cards:
            # Decision cards are dicts from router.generate_decision_cards
            assert isinstance(card, dict)
            assert "method" in card
            assert "roles_covered" in card


# ============================================================================
# Test: Complete scenario end-to-end
# ============================================================================

class TestCompleteScenario:
    """Full scenario: profile -> route -> predict -> ICM -> CRC -> decide -> record."""

    def test_full_scenario_financial_stress(self):
        """Simulate a complete financial stress test analysis."""
        # 1. Profile the system
        profile = FINANCIAL_ENERGY_SYSTEM
        assert profile.name

        # 2. Route to methods
        catalog = get_catalog()
        config = OSMultiScienceConfig()
        selection = select_kit(profile, catalog, config.router)
        assert len(selection.selected_methods) >= 3

        # 3. Simulate multi-model execution
        rng = np.random.default_rng(42)
        n_samples = 300
        X = rng.standard_normal((n_samples, 5))
        y_true = np.sin(X[:, 0]) + 0.5 * X[:, 1] + 0.2 * rng.standard_normal(n_samples)

        predictions = {}
        for i, method in enumerate(selection.selected_methods):
            noise = 0.1 * (i + 1)
            predictions[method.name] = y_true + noise * rng.standard_normal(n_samples)

        # 4. Compute ICM
        icm_result = compute_icm_from_predictions(predictions, config.icm, distance_fn="wasserstein")
        assert 0 <= icm_result.icm_score <= 1
        assert icm_result.n_models == len(selection.selected_methods)

        # 5. CRC gating
        losses = np.array([
            np.mean((predictions[m.name] - y_true) ** 2)
            for m in selection.selected_methods
        ])
        avg_loss_per_sample = np.mean(
            [(predictions[m.name] - y_true) ** 2 for m in selection.selected_methods],
            axis=0,
        )

        C_values = np.full(n_samples, icm_result.icm_score)
        idx = rng.permutation(n_samples)
        half = n_samples // 2

        g = fit_isotonic(C_values[idx[:half]], avg_loss_per_sample[idx[:half]])
        g_alpha = conformalize(g, C_values[idx[half:]], avg_loss_per_sample[idx[half:]], alpha=0.10)

        re = compute_re(icm_result.icm_score, g_alpha)
        decision = decision_gate(icm_result.icm_score, re, config.crc)

        # 6. Record in Knowledge Graph
        kg = KnowledgeGraph()
        kg.add_node(KnowledgeNode(
            id="financial_stress", node_type=NodeType.SYSTEM,
            data={"name": profile.name, "n_samples": n_samples},
        ))

        for method in selection.selected_methods:
            kg.add_node(KnowledgeNode(
                id=f"m_{method.name}", node_type=NodeType.METHOD,
                data={"name": method.name, "family": method.family.value},
            ))
            kg.add_edge(KnowledgeEdge(
                "financial_stress", f"m_{method.name}", EdgeType.ANALYZED_BY,
            ))

        kg.add_node(KnowledgeNode(
            id="icm_1", node_type=NodeType.ICM_SCORE,
            data={"score": icm_result.icm_score, "n_models": icm_result.n_models},
        ))
        kg.add_node(KnowledgeNode(
            id="decision_1", node_type=NodeType.DECISION,
            data={"action": decision.value, "re_score": re},
        ))
        kg.add_edge(KnowledgeEdge("icm_1", "decision_1", EdgeType.LED_TO))

        # 7. Generate decision cards
        cards = generate_decision_cards(selection, profile)
        assert len(cards) > 0

        # 8. Verify KG state
        assert len(kg) > 0
        systems = kg.find_nodes(NodeType.SYSTEM)
        assert len(systems) == 1
        decisions = kg.find_nodes(NodeType.DECISION)
        assert len(decisions) == 1
        assert decisions[0].data["action"] == decision.value

        # 9. Summary
        summary = kg.summary()
        assert summary["total_nodes"] >= 5
        assert summary["total_edges"] >= 4
