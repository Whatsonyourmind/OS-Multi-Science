"""Comprehensive tests for the Knowledge Graph module.

Tests cover:
- Node and edge CRUD operations
- Query API (find_nodes, get_neighbors, get_subgraph)
- Epistemic queries (converging methods, conflicting results, lineage, provenance)
- Serialization round-trip
- Edge cases (missing nodes, empty graph, circular references, duplicates)
"""

from __future__ import annotations

import time

import pytest

from knowledge.graph import (
    EdgeType,
    KnowledgeEdge,
    KnowledgeGraph,
    KnowledgeNode,
    NodeType,
    generate_id,
)


# ============================================================
# Fixtures
# ============================================================

def _make_node(
    node_id: str,
    node_type: NodeType,
    data: dict | None = None,
    metadata: dict | None = None,
) -> KnowledgeNode:
    """Helper to create a KnowledgeNode with sensible defaults."""
    return KnowledgeNode(
        id=node_id,
        node_type=node_type,
        data=data or {},
        timestamp=time.time(),
        metadata=metadata or {},
    )


def _make_edge(
    source_id: str,
    target_id: str,
    edge_type: EdgeType,
    weight: float = 1.0,
    metadata: dict | None = None,
) -> KnowledgeEdge:
    """Helper to create a KnowledgeEdge."""
    return KnowledgeEdge(
        source_id=source_id,
        target_id=target_id,
        edge_type=edge_type,
        weight=weight,
        metadata=metadata or {},
    )


@pytest.fixture
def empty_graph() -> KnowledgeGraph:
    """Return a fresh empty KnowledgeGraph."""
    return KnowledgeGraph()


@pytest.fixture
def simple_graph() -> KnowledgeGraph:
    """Return a graph with a basic system -> method -> result chain."""
    g = KnowledgeGraph()
    g.add_node(_make_node("sys1", NodeType.SYSTEM, {"name": "economy"}))
    g.add_node(_make_node("m1", NodeType.METHOD, {"name": "ABM"}))
    g.add_node(_make_node("m2", NodeType.METHOD, {"name": "Econometric"}))
    g.add_node(_make_node("r1", NodeType.RESULT, {"prediction": 0.75}))
    g.add_node(_make_node("r2", NodeType.RESULT, {"prediction": 0.80}))
    g.add_edge(_make_edge("sys1", "m1", EdgeType.ANALYZED_BY))
    g.add_edge(_make_edge("sys1", "m2", EdgeType.ANALYZED_BY))
    g.add_edge(_make_edge("m1", "r1", EdgeType.PRODUCED))
    g.add_edge(_make_edge("m2", "r2", EdgeType.PRODUCED))
    return g


@pytest.fixture
def epistemic_graph() -> KnowledgeGraph:
    """Return a richer graph for epistemic query testing.

    Structure:
      sys1 --ANALYZED_BY--> m1 --PRODUCED--> r1
      sys1 --ANALYZED_BY--> m2 --PRODUCED--> r2
      sys1 --ANALYZED_BY--> m3 --PRODUCED--> r3
      r1 --CONVERGES_WITH--> r2
      r1 --CONTRADICTS--> r3
      icm1 (score=0.85) --LED_TO--> dec1
      m1 linked to icm1 via edge
    """
    g = KnowledgeGraph()
    g.add_node(_make_node("sys1", NodeType.SYSTEM, {"name": "climate"}))
    g.add_node(_make_node("m1", NodeType.METHOD, {"name": "ABM", "family": "agent_based"}))
    g.add_node(_make_node("m2", NodeType.METHOD, {"name": "SysDyn", "family": "system_dynamics"}))
    g.add_node(_make_node("m3", NodeType.METHOD, {"name": "Deep Learning", "family": "ml"}))
    g.add_node(_make_node("r1", NodeType.RESULT, {"prediction": 2.5, "label": "warming"}))
    g.add_node(_make_node("r2", NodeType.RESULT, {"prediction": 2.7, "label": "warming"}))
    g.add_node(_make_node("r3", NodeType.RESULT, {"prediction": -0.5, "label": "cooling"}))
    g.add_node(_make_node("icm1", NodeType.ICM_SCORE, {"icm_score": 0.85}))
    g.add_node(_make_node("dec1", NodeType.DECISION, {"action": "ACT", "confidence": 0.9}))

    g.add_edge(_make_edge("sys1", "m1", EdgeType.ANALYZED_BY))
    g.add_edge(_make_edge("sys1", "m2", EdgeType.ANALYZED_BY))
    g.add_edge(_make_edge("sys1", "m3", EdgeType.ANALYZED_BY))
    g.add_edge(_make_edge("m1", "r1", EdgeType.PRODUCED))
    g.add_edge(_make_edge("m2", "r2", EdgeType.PRODUCED))
    g.add_edge(_make_edge("m3", "r3", EdgeType.PRODUCED))
    g.add_edge(_make_edge("r1", "r2", EdgeType.CONVERGES_WITH))
    g.add_edge(_make_edge("r1", "r3", EdgeType.CONTRADICTS))
    g.add_edge(_make_edge("icm1", "dec1", EdgeType.LED_TO))
    # Link icm1 to m1 so find_converging_methods can discover it
    g.add_edge(_make_edge("m1", "icm1", EdgeType.PRODUCED))

    return g


# ============================================================
# 1. Node CRUD
# ============================================================

class TestNodeCRUD:
    """Tests for adding, getting, and removing nodes."""

    def test_add_and_get_node(self, empty_graph: KnowledgeGraph):
        node = _make_node("n1", NodeType.SYSTEM, {"name": "test"})
        empty_graph.add_node(node)
        retrieved = empty_graph.get_node("n1")
        assert retrieved is not None
        assert retrieved.id == "n1"
        assert retrieved.node_type == NodeType.SYSTEM
        assert retrieved.data["name"] == "test"

    def test_get_missing_node_returns_none(self, empty_graph: KnowledgeGraph):
        assert empty_graph.get_node("nonexistent") is None

    def test_add_node_overwrites(self, empty_graph: KnowledgeGraph):
        node_v1 = _make_node("n1", NodeType.SYSTEM, {"version": 1})
        node_v2 = _make_node("n1", NodeType.SYSTEM, {"version": 2})
        empty_graph.add_node(node_v1)
        empty_graph.add_node(node_v2)
        assert empty_graph.get_node("n1").data["version"] == 2

    def test_remove_node(self, simple_graph: KnowledgeGraph):
        assert simple_graph.get_node("m1") is not None
        simple_graph.remove_node("m1")
        assert simple_graph.get_node("m1") is None

    def test_remove_node_cleans_edges(self, simple_graph: KnowledgeGraph):
        simple_graph.remove_node("m1")
        # Edges from sys1 to m1 should be gone
        edges = simple_graph.get_edges("sys1", direction="outgoing")
        target_ids = [e.target_id for e in edges]
        assert "m1" not in target_ids
        # Edges from m1 to r1 should also be gone
        edges_r1 = simple_graph.get_edges("r1", direction="incoming")
        source_ids = [e.source_id for e in edges_r1]
        assert "m1" not in source_ids

    def test_remove_nonexistent_node_is_noop(self, empty_graph: KnowledgeGraph):
        empty_graph.remove_node("ghost")  # Should not raise


# ============================================================
# 2. Edge CRUD
# ============================================================

class TestEdgeCRUD:
    """Tests for adding and querying edges."""

    def test_add_edge(self, empty_graph: KnowledgeGraph):
        empty_graph.add_node(_make_node("a", NodeType.SYSTEM))
        empty_graph.add_node(_make_node("b", NodeType.METHOD))
        empty_graph.add_edge(_make_edge("a", "b", EdgeType.ANALYZED_BY))
        edges = empty_graph.get_edges("a", direction="outgoing")
        assert len(edges) == 1
        assert edges[0].target_id == "b"

    def test_add_edge_missing_source_raises(self, empty_graph: KnowledgeGraph):
        empty_graph.add_node(_make_node("b", NodeType.METHOD))
        with pytest.raises(KeyError, match="Source node"):
            empty_graph.add_edge(_make_edge("ghost", "b", EdgeType.ANALYZED_BY))

    def test_add_edge_missing_target_raises(self, empty_graph: KnowledgeGraph):
        empty_graph.add_node(_make_node("a", NodeType.SYSTEM))
        with pytest.raises(KeyError, match="Target node"):
            empty_graph.add_edge(_make_edge("a", "ghost", EdgeType.ANALYZED_BY))

    def test_get_edges_outgoing(self, simple_graph: KnowledgeGraph):
        edges = simple_graph.get_edges("sys1", direction="outgoing")
        assert len(edges) == 2
        targets = {e.target_id for e in edges}
        assert targets == {"m1", "m2"}

    def test_get_edges_incoming(self, simple_graph: KnowledgeGraph):
        edges = simple_graph.get_edges("r1", direction="incoming")
        assert len(edges) == 1
        assert edges[0].source_id == "m1"

    def test_get_edges_both(self, simple_graph: KnowledgeGraph):
        edges = simple_graph.get_edges("m1", direction="both")
        # m1 has 1 incoming (from sys1) and 1 outgoing (to r1)
        assert len(edges) == 2

    def test_edge_weight_and_metadata(self, empty_graph: KnowledgeGraph):
        empty_graph.add_node(_make_node("a", NodeType.RESULT))
        empty_graph.add_node(_make_node("b", NodeType.RESULT))
        edge = _make_edge("a", "b", EdgeType.CONVERGES_WITH, weight=0.9,
                          metadata={"confidence": 0.95})
        empty_graph.add_edge(edge)
        retrieved = empty_graph.get_edges("a", direction="outgoing")[0]
        assert retrieved.weight == pytest.approx(0.9)
        assert retrieved.metadata["confidence"] == pytest.approx(0.95)


# ============================================================
# 3. Query API
# ============================================================

class TestQueryAPI:
    """Tests for find_nodes, get_neighbors, get_subgraph."""

    def test_find_nodes_by_type(self, simple_graph: KnowledgeGraph):
        methods = simple_graph.find_nodes(node_type=NodeType.METHOD)
        assert len(methods) == 2
        names = {m.data.get("name") for m in methods}
        assert names == {"ABM", "Econometric"}

    def test_find_nodes_with_data_filter(self, simple_graph: KnowledgeGraph):
        results = simple_graph.find_nodes(
            node_type=NodeType.RESULT, prediction=0.75
        )
        assert len(results) == 1
        assert results[0].id == "r1"

    def test_find_nodes_with_metadata_filter(self, empty_graph: KnowledgeGraph):
        node = _make_node("n1", NodeType.SYSTEM, metadata={"source": "paper_A"})
        empty_graph.add_node(node)
        found = empty_graph.find_nodes(source="paper_A")
        assert len(found) == 1
        assert found[0].id == "n1"

    def test_find_nodes_no_match(self, simple_graph: KnowledgeGraph):
        found = simple_graph.find_nodes(node_type=NodeType.DECISION)
        assert len(found) == 0

    def test_find_nodes_all(self, simple_graph: KnowledgeGraph):
        all_nodes = simple_graph.find_nodes()
        assert len(all_nodes) == 5  # sys1, m1, m2, r1, r2

    def test_get_neighbors(self, simple_graph: KnowledgeGraph):
        neighbors = simple_graph.get_neighbors("sys1")
        neighbor_ids = {n.id for n in neighbors}
        assert neighbor_ids == {"m1", "m2"}

    def test_get_neighbors_with_edge_type_filter(
        self, epistemic_graph: KnowledgeGraph
    ):
        # r1 has CONVERGES_WITH -> r2 and CONTRADICTS -> r3
        converging = epistemic_graph.get_neighbors("r1", edge_type=EdgeType.CONVERGES_WITH)
        assert len(converging) == 1
        assert converging[0].id == "r2"

    def test_get_subgraph_depth_0(self, simple_graph: KnowledgeGraph):
        sub = simple_graph.get_subgraph("sys1", max_depth=0)
        assert len(sub) == 1
        assert "sys1" in sub

    def test_get_subgraph_depth_1(self, simple_graph: KnowledgeGraph):
        sub = simple_graph.get_subgraph("sys1", max_depth=1)
        assert "sys1" in sub
        assert "m1" in sub
        assert "m2" in sub
        # Results are 2 hops away, should NOT be included at depth 1
        assert "r1" not in sub

    def test_get_subgraph_depth_2(self, simple_graph: KnowledgeGraph):
        sub = simple_graph.get_subgraph("sys1", max_depth=2)
        assert len(sub) == 5  # all nodes reachable within 2 hops
        assert "r1" in sub
        assert "r2" in sub

    def test_get_subgraph_missing_root_raises(self, empty_graph: KnowledgeGraph):
        with pytest.raises(KeyError, match="Root node"):
            empty_graph.get_subgraph("ghost")

    def test_get_subgraph_preserves_edges(self, simple_graph: KnowledgeGraph):
        sub = simple_graph.get_subgraph("sys1", max_depth=2)
        edges = sub.get_edges("m1", direction="outgoing")
        assert any(e.target_id == "r1" for e in edges)


# ============================================================
# 4. Epistemic Queries
# ============================================================

class TestEpistemicQueries:
    """Tests for epistemic-specific query methods."""

    def test_find_converging_methods_via_converges_edge(
        self, epistemic_graph: KnowledgeGraph
    ):
        # m1 produced r1, which CONVERGES_WITH r2; m2 produced r2 (target of CONVERGES_WITH)
        converging = epistemic_graph.find_converging_methods("sys1", min_icm=0.5)
        # m1 should converge (r1 has CONVERGES_WITH), m2 should also (r2 is target of CONVERGES_WITH)
        assert "m1" in converging

    def test_find_converging_methods_via_icm_score(
        self, epistemic_graph: KnowledgeGraph
    ):
        # m1 is connected to icm1 (score 0.85 >= 0.5)
        converging = epistemic_graph.find_converging_methods("sys1", min_icm=0.5)
        assert "m1" in converging

    def test_find_converging_methods_high_threshold(
        self, epistemic_graph: KnowledgeGraph
    ):
        # With very high threshold, ICM-based convergence might not qualify
        # but CONVERGES_WITH edge still qualifies m1
        converging = epistemic_graph.find_converging_methods("sys1", min_icm=0.99)
        # m1 still has CONVERGES_WITH edge on r1
        assert "m1" in converging

    def test_find_converging_methods_missing_system(
        self, epistemic_graph: KnowledgeGraph
    ):
        assert epistemic_graph.find_converging_methods("nonexistent") == []

    def test_find_conflicting_results(self, epistemic_graph: KnowledgeGraph):
        conflicts = epistemic_graph.find_conflicting_results("sys1")
        assert len(conflicts) == 1
        pair = conflicts[0]
        assert set(pair) == {"r1", "r3"}

    def test_find_conflicting_results_no_conflicts(
        self, simple_graph: KnowledgeGraph
    ):
        conflicts = simple_graph.find_conflicting_results("sys1")
        assert len(conflicts) == 0

    def test_find_conflicting_results_missing_system(
        self, epistemic_graph: KnowledgeGraph
    ):
        assert epistemic_graph.find_conflicting_results("nonexistent") == []

    def test_get_method_lineage(self, epistemic_graph: KnowledgeGraph):
        lineage = epistemic_graph.get_method_lineage("r1")
        lineage_ids = {n.id for n in lineage}
        # r1 was produced by m1, m1 was analyzed_by from sys1
        assert "m1" in lineage_ids
        assert "sys1" in lineage_ids

    def test_get_method_lineage_missing_result(
        self, epistemic_graph: KnowledgeGraph
    ):
        assert epistemic_graph.get_method_lineage("nonexistent") == []

    def test_get_decision_provenance(self, epistemic_graph: KnowledgeGraph):
        prov = epistemic_graph.get_decision_provenance("dec1")
        assert prov["decision"]["id"] == "dec1"
        assert prov["decision"]["data"]["action"] == "ACT"
        icm_ids = [s["id"] for s in prov["icm_scores"]]
        assert "icm1" in icm_ids

    def test_get_decision_provenance_missing_decision(
        self, epistemic_graph: KnowledgeGraph
    ):
        assert epistemic_graph.get_decision_provenance("nonexistent") == {}


# ============================================================
# 5. Serialization
# ============================================================

class TestSerialization:
    """Tests for to_dict / from_dict round-trip."""

    def test_round_trip_empty(self, empty_graph: KnowledgeGraph):
        data = empty_graph.to_dict()
        restored = KnowledgeGraph.from_dict(data)
        assert len(restored) == 0
        assert restored.summary()["total_nodes"] == 0
        assert restored.summary()["total_edges"] == 0

    def test_round_trip_preserves_nodes(self, simple_graph: KnowledgeGraph):
        data = simple_graph.to_dict()
        restored = KnowledgeGraph.from_dict(data)
        assert len(restored) == len(simple_graph)
        for nid in ["sys1", "m1", "m2", "r1", "r2"]:
            orig = simple_graph.get_node(nid)
            rest = restored.get_node(nid)
            assert rest is not None
            assert rest.node_type == orig.node_type
            assert rest.data == orig.data

    def test_round_trip_preserves_edges(self, simple_graph: KnowledgeGraph):
        data = simple_graph.to_dict()
        restored = KnowledgeGraph.from_dict(data)
        orig_summary = simple_graph.summary()
        rest_summary = restored.summary()
        assert orig_summary["total_edges"] == rest_summary["total_edges"]

    def test_round_trip_preserves_edge_metadata(self, empty_graph: KnowledgeGraph):
        empty_graph.add_node(_make_node("a", NodeType.RESULT))
        empty_graph.add_node(_make_node("b", NodeType.RESULT))
        empty_graph.add_edge(_make_edge(
            "a", "b", EdgeType.CONVERGES_WITH,
            weight=0.77, metadata={"reason": "overlap"},
        ))
        data = empty_graph.to_dict()
        restored = KnowledgeGraph.from_dict(data)
        edges = restored.get_edges("a", direction="outgoing")
        assert len(edges) == 1
        assert edges[0].weight == pytest.approx(0.77)
        assert edges[0].metadata["reason"] == "overlap"

    def test_round_trip_complex(self, epistemic_graph: KnowledgeGraph):
        data = epistemic_graph.to_dict()
        restored = KnowledgeGraph.from_dict(data)
        # Verify same structure
        orig = epistemic_graph.summary()
        rest = restored.summary()
        assert orig["total_nodes"] == rest["total_nodes"]
        assert orig["total_edges"] == rest["total_edges"]
        # Verify epistemic queries still work
        conflicts = restored.find_conflicting_results("sys1")
        assert len(conflicts) == 1


# ============================================================
# 6. Edge Cases
# ============================================================

class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_empty_graph_summary(self, empty_graph: KnowledgeGraph):
        s = empty_graph.summary()
        assert s["total_nodes"] == 0
        assert s["total_edges"] == 0

    def test_empty_graph_find_nodes(self, empty_graph: KnowledgeGraph):
        assert empty_graph.find_nodes() == []

    def test_circular_reference(self, empty_graph: KnowledgeGraph):
        """Circular edges (A->B->A) should not cause infinite loops."""
        empty_graph.add_node(_make_node("a", NodeType.METHOD))
        empty_graph.add_node(_make_node("b", NodeType.METHOD))
        empty_graph.add_edge(_make_edge("a", "b", EdgeType.DEPENDS_ON))
        empty_graph.add_edge(_make_edge("b", "a", EdgeType.DEPENDS_ON))
        # get_subgraph should terminate
        sub = empty_graph.get_subgraph("a", max_depth=10)
        assert len(sub) == 2
        # get_neighbors should not loop
        neighbors = empty_graph.get_neighbors("a")
        assert len(neighbors) == 1
        assert neighbors[0].id == "b"

    def test_self_loop(self, empty_graph: KnowledgeGraph):
        """Self-loop edges should be handled gracefully."""
        empty_graph.add_node(_make_node("a", NodeType.METHOD))
        empty_graph.add_edge(_make_edge("a", "a", EdgeType.DEPENDS_ON))
        edges = empty_graph.get_edges("a")
        # Self-loop appears in both outgoing and incoming
        assert len(edges) == 2
        sub = empty_graph.get_subgraph("a", max_depth=5)
        assert len(sub) == 1

    def test_generate_id_uniqueness(self):
        ids = {generate_id("test") for _ in range(1000)}
        assert len(ids) == 1000

    def test_generate_id_prefix(self):
        nid = generate_id("system")
        assert nid.startswith("system_")

    def test_generate_id_no_prefix(self):
        nid = generate_id()
        assert "_" not in nid or len(nid) == 12

    def test_contains_and_len(self, simple_graph: KnowledgeGraph):
        assert "sys1" in simple_graph
        assert "ghost" not in simple_graph
        assert len(simple_graph) == 5

    def test_repr(self, simple_graph: KnowledgeGraph):
        r = repr(simple_graph)
        assert "KnowledgeGraph" in r
        assert "nodes=5" in r

    def test_summary_per_type(self, epistemic_graph: KnowledgeGraph):
        s = epistemic_graph.summary()
        assert s["nodes_system"] == 1
        assert s["nodes_method"] == 3
        assert s["nodes_result"] == 3
        assert s["nodes_icm_score"] == 1
        assert s["nodes_decision"] == 1
        assert s["total_nodes"] == 9
        # Edges: 3 ANALYZED_BY + 4 PRODUCED (m1->r1,m2->r2,m3->r3,m1->icm1) +
        # 1 CONVERGES_WITH + 1 CONTRADICTS + 1 LED_TO = 10
        assert s["total_edges"] == 10

    def test_get_edges_empty_node(self, empty_graph: KnowledgeGraph):
        """get_edges on a node not in the graph returns empty list."""
        assert empty_graph.get_edges("ghost") == []

    def test_multiple_edges_between_same_nodes(self, empty_graph: KnowledgeGraph):
        """Multiple edges between the same pair should all be stored."""
        empty_graph.add_node(_make_node("a", NodeType.RESULT))
        empty_graph.add_node(_make_node("b", NodeType.RESULT))
        empty_graph.add_edge(_make_edge("a", "b", EdgeType.CONVERGES_WITH, weight=0.8))
        empty_graph.add_edge(_make_edge("a", "b", EdgeType.CONTRADICTS, weight=0.2))
        edges = empty_graph.get_edges("a", direction="outgoing")
        assert len(edges) == 2
        types = {e.edge_type for e in edges}
        assert types == {EdgeType.CONVERGES_WITH, EdgeType.CONTRADICTS}
