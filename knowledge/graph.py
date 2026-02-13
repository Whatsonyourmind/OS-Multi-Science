"""Lightweight in-memory knowledge graph for epistemic provenance tracking.

Tracks relationships between systems, methods, results, decisions, and ICM
scores without external graph database dependencies. All storage is dict-based
and supports bidirectional edge traversal, serialization round-trips, and
epistemic queries (converging methods, conflicting results, decision lineage).
"""

from __future__ import annotations

import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


# ============================================================
# Enums
# ============================================================

class NodeType(Enum):
    """Types of nodes in the knowledge graph."""
    SYSTEM = "system"
    METHOD = "method"
    RESULT = "result"
    DECISION = "decision"
    ICM_SCORE = "icm_score"


class EdgeType(Enum):
    """Types of edges in the knowledge graph."""
    ANALYZED_BY = "analyzed_by"          # system -> method
    PRODUCED = "produced"                # method -> result
    CONVERGES_WITH = "converges_with"    # result <-> result
    DEPENDS_ON = "depends_on"            # method -> method (shared data)
    LED_TO = "led_to"                    # icm_score -> decision
    CONTRADICTS = "contradicts"          # result <-> result


# ============================================================
# Data classes
# ============================================================

@dataclass
class KnowledgeNode:
    """A node in the knowledge graph.

    Attributes
    ----------
    id : str
        Unique identifier (descriptive or UUID-based).
    node_type : NodeType
        Semantic category of the node.
    data : dict[str, Any]
        Flexible payload (e.g. ICM score value, method name, etc.).
    timestamp : float
        Creation time (time.time()).
    metadata : dict[str, Any]
        Tags, source info, and other annotations.
    """
    id: str
    node_type: NodeType
    data: dict[str, Any]
    timestamp: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class KnowledgeEdge:
    """A directed edge in the knowledge graph.

    Attributes
    ----------
    source_id : str
        ID of the source node.
    target_id : str
        ID of the target node.
    edge_type : EdgeType
        Semantic type of the relationship.
    weight : float
        Edge weight (default 1.0).
    metadata : dict[str, Any]
        Additional edge annotations.
    """
    source_id: str
    target_id: str
    edge_type: EdgeType
    weight: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)


# ============================================================
# Helpers
# ============================================================

def generate_id(prefix: str = "") -> str:
    """Generate a unique node ID with an optional descriptive prefix."""
    uid = uuid.uuid4().hex[:12]
    if prefix:
        return f"{prefix}_{uid}"
    return uid


# ============================================================
# Knowledge Graph
# ============================================================

class KnowledgeGraph:
    """Lightweight in-memory knowledge graph with epistemic provenance queries.

    Storage is entirely dict-based (no external graph DB). Edges are indexed
    by both source and target to support efficient bidirectional traversal.
    """

    def __init__(self) -> None:
        # Node storage: id -> KnowledgeNode
        self._nodes: dict[str, KnowledgeNode] = {}
        # Edge storage indexed by source and target for O(1) lookup
        self._edges_by_source: dict[str, list[KnowledgeEdge]] = {}
        self._edges_by_target: dict[str, list[KnowledgeEdge]] = {}

    # ----------------------------------------------------------------
    # Core CRUD
    # ----------------------------------------------------------------

    def add_node(self, node: KnowledgeNode) -> None:
        """Add a node to the graph. Overwrites if node ID already exists."""
        self._nodes[node.id] = node
        # Ensure edge indices exist even if no edges yet
        if node.id not in self._edges_by_source:
            self._edges_by_source[node.id] = []
        if node.id not in self._edges_by_target:
            self._edges_by_target[node.id] = []

    def add_edge(self, edge: KnowledgeEdge) -> None:
        """Add a directed edge. Both source and target must already exist.

        Raises
        ------
        KeyError
            If either source_id or target_id is not in the graph.
        """
        if edge.source_id not in self._nodes:
            raise KeyError(
                f"Source node '{edge.source_id}' not found in graph."
            )
        if edge.target_id not in self._nodes:
            raise KeyError(
                f"Target node '{edge.target_id}' not found in graph."
            )
        self._edges_by_source.setdefault(edge.source_id, []).append(edge)
        self._edges_by_target.setdefault(edge.target_id, []).append(edge)

    def get_node(self, node_id: str) -> KnowledgeNode | None:
        """Return a node by ID, or None if not found."""
        return self._nodes.get(node_id)

    def get_edges(
        self,
        node_id: str,
        direction: str = "both",
    ) -> list[KnowledgeEdge]:
        """Return edges connected to *node_id*.

        Parameters
        ----------
        node_id : str
            The node whose edges to retrieve.
        direction : str
            One of ``"outgoing"``, ``"incoming"``, or ``"both"`` (default).

        Returns
        -------
        list[KnowledgeEdge]
        """
        edges: list[KnowledgeEdge] = []
        if direction in ("outgoing", "both"):
            edges.extend(self._edges_by_source.get(node_id, []))
        if direction in ("incoming", "both"):
            edges.extend(self._edges_by_target.get(node_id, []))
        return edges

    def remove_node(self, node_id: str) -> None:
        """Remove a node and all its incident edges.

        Does nothing if the node does not exist.
        """
        if node_id not in self._nodes:
            return

        # Remove outgoing edges (also remove them from target indices)
        for edge in list(self._edges_by_source.get(node_id, [])):
            target_edges = self._edges_by_target.get(edge.target_id, [])
            self._edges_by_target[edge.target_id] = [
                e for e in target_edges if e is not edge
            ]
        self._edges_by_source.pop(node_id, None)

        # Remove incoming edges (also remove them from source indices)
        for edge in list(self._edges_by_target.get(node_id, [])):
            source_edges = self._edges_by_source.get(edge.source_id, [])
            self._edges_by_source[edge.source_id] = [
                e for e in source_edges if e is not edge
            ]
        self._edges_by_target.pop(node_id, None)

        del self._nodes[node_id]

    # ----------------------------------------------------------------
    # Query API
    # ----------------------------------------------------------------

    def find_nodes(
        self,
        node_type: NodeType | None = None,
        **filters: Any,
    ) -> list[KnowledgeNode]:
        """Find nodes matching optional type and data/metadata filters.

        Parameters
        ----------
        node_type : NodeType | None
            If given, restrict to nodes of this type.
        **filters
            Key-value pairs matched against node.data and node.metadata.
            A node matches if every key exists in either ``data`` or
            ``metadata`` and has an equal value.

        Returns
        -------
        list[KnowledgeNode]
        """
        results: list[KnowledgeNode] = []
        for node in self._nodes.values():
            if node_type is not None and node.node_type != node_type:
                continue
            if filters:
                match = True
                for key, value in filters.items():
                    node_val = node.data.get(key, node.metadata.get(key, _SENTINEL))
                    if node_val is _SENTINEL or node_val != value:
                        match = False
                        break
                if not match:
                    continue
            results.append(node)
        return results

    def get_neighbors(
        self,
        node_id: str,
        edge_type: EdgeType | None = None,
    ) -> list[KnowledgeNode]:
        """Return neighbor nodes connected to *node_id*.

        For outgoing edges, the neighbor is the target; for incoming edges
        it is the source. An optional *edge_type* filter is applied.

        Parameters
        ----------
        node_id : str
            The reference node.
        edge_type : EdgeType | None
            If given, only follow edges of this type.

        Returns
        -------
        list[KnowledgeNode]
            De-duplicated list of neighbor nodes.
        """
        neighbor_ids: set[str] = set()
        for edge in self._edges_by_source.get(node_id, []):
            if edge_type is None or edge.edge_type == edge_type:
                neighbor_ids.add(edge.target_id)
        for edge in self._edges_by_target.get(node_id, []):
            if edge_type is None or edge.edge_type == edge_type:
                neighbor_ids.add(edge.source_id)
        return [self._nodes[nid] for nid in neighbor_ids if nid in self._nodes]

    def get_subgraph(
        self,
        root_id: str,
        max_depth: int = 3,
    ) -> "KnowledgeGraph":
        """Extract the subgraph reachable from *root_id* within *max_depth* hops.

        Traversal is bidirectional (follows both incoming and outgoing edges).

        Parameters
        ----------
        root_id : str
            Starting node.
        max_depth : int
            Maximum BFS depth (default 3).

        Returns
        -------
        KnowledgeGraph
            A new graph containing only reachable nodes and their mutual edges.

        Raises
        ------
        KeyError
            If *root_id* is not in the graph.
        """
        if root_id not in self._nodes:
            raise KeyError(f"Root node '{root_id}' not found in graph.")

        visited: set[str] = set()
        queue: deque[tuple[str, int]] = deque([(root_id, 0)])

        while queue:
            current_id, depth = queue.popleft()
            if current_id in visited:
                continue
            visited.add(current_id)
            if depth >= max_depth:
                continue
            # Explore neighbors in both directions
            for edge in self._edges_by_source.get(current_id, []):
                if edge.target_id not in visited:
                    queue.append((edge.target_id, depth + 1))
            for edge in self._edges_by_target.get(current_id, []):
                if edge.source_id not in visited:
                    queue.append((edge.source_id, depth + 1))

        # Build new graph
        sub = KnowledgeGraph()
        for nid in visited:
            sub.add_node(self._nodes[nid])
        # Add edges where both endpoints are in the subgraph
        for nid in visited:
            for edge in self._edges_by_source.get(nid, []):
                if edge.target_id in visited:
                    sub.add_edge(edge)
        return sub

    # ----------------------------------------------------------------
    # Epistemic queries
    # ----------------------------------------------------------------

    def find_converging_methods(
        self,
        system_id: str,
        min_icm: float = 0.5,
    ) -> list[str]:
        """Find methods that analyzed *system_id* and have converging results.

        A method is "converging" if there exists an ICM_SCORE node connected
        (via LED_TO or as neighbor) with ``icm_score >= min_icm`` that
        relates to this system's analysis chain.

        More precisely the algorithm follows:
        ``system --ANALYZED_BY--> method --PRODUCED--> result``
        and collects methods whose results have a CONVERGES_WITH edge to
        another result, **or** whose analysis chain includes an ICM_SCORE
        node with score >= min_icm.

        Parameters
        ----------
        system_id : str
            The system node ID.
        min_icm : float
            Minimum ICM score threshold.

        Returns
        -------
        list[str]
            Method node IDs that meet the convergence criterion.
        """
        if system_id not in self._nodes:
            return []

        # Step 1: find all methods that analyzed this system
        method_ids: list[str] = []
        for edge in self._edges_by_source.get(system_id, []):
            if edge.edge_type == EdgeType.ANALYZED_BY:
                method_ids.append(edge.target_id)

        converging: list[str] = []
        for mid in method_ids:
            # Check if method produced results that converge
            has_convergence = False

            # Gather results produced by this method
            result_ids: list[str] = []
            for edge in self._edges_by_source.get(mid, []):
                if edge.edge_type == EdgeType.PRODUCED:
                    result_ids.append(edge.target_id)

            # Check results for CONVERGES_WITH edges
            for rid in result_ids:
                for edge in self._edges_by_source.get(rid, []):
                    if edge.edge_type == EdgeType.CONVERGES_WITH:
                        has_convergence = True
                        break
                if not has_convergence:
                    for edge in self._edges_by_target.get(rid, []):
                        if edge.edge_type == EdgeType.CONVERGES_WITH:
                            has_convergence = True
                            break
                if has_convergence:
                    break

            # Also check for ICM_SCORE nodes with high score in the neighborhood
            if not has_convergence:
                # Look at ICM_SCORE nodes connected to this method's results
                for rid in result_ids:
                    neighbors = self.get_neighbors(rid)
                    for neighbor in neighbors:
                        if neighbor.node_type == NodeType.ICM_SCORE:
                            score = neighbor.data.get("icm_score", 0.0)
                            if score >= min_icm:
                                has_convergence = True
                                break
                    if has_convergence:
                        break

            # Also check ICM_SCORE nodes directly connected to the method
            if not has_convergence:
                neighbors = self.get_neighbors(mid)
                for neighbor in neighbors:
                    if neighbor.node_type == NodeType.ICM_SCORE:
                        score = neighbor.data.get("icm_score", 0.0)
                        if score >= min_icm:
                            has_convergence = True
                            break

            if has_convergence:
                converging.append(mid)

        return converging

    def find_conflicting_results(
        self,
        system_id: str,
    ) -> list[tuple[str, str]]:
        """Find pairs of results that contradict each other for a system.

        Follows ``system --ANALYZED_BY--> method --PRODUCED--> result``
        then collects pairs of results connected by CONTRADICTS edges.

        Parameters
        ----------
        system_id : str
            The system node ID.

        Returns
        -------
        list[tuple[str, str]]
            Pairs of contradicting result node IDs.
        """
        if system_id not in self._nodes:
            return []

        # Gather all result IDs produced by methods that analyzed this system
        result_ids: set[str] = set()
        for edge in self._edges_by_source.get(system_id, []):
            if edge.edge_type == EdgeType.ANALYZED_BY:
                method_id = edge.target_id
                for medge in self._edges_by_source.get(method_id, []):
                    if medge.edge_type == EdgeType.PRODUCED:
                        result_ids.add(medge.target_id)

        # Find CONTRADICTS edges between these results
        conflicts: list[tuple[str, str]] = []
        seen: set[tuple[str, str]] = set()
        for rid in result_ids:
            for edge in self._edges_by_source.get(rid, []):
                if (
                    edge.edge_type == EdgeType.CONTRADICTS
                    and edge.target_id in result_ids
                ):
                    pair = tuple(sorted((edge.source_id, edge.target_id)))
                    if pair not in seen:
                        seen.add(pair)
                        conflicts.append((edge.source_id, edge.target_id))
            for edge in self._edges_by_target.get(rid, []):
                if (
                    edge.edge_type == EdgeType.CONTRADICTS
                    and edge.source_id in result_ids
                ):
                    pair = tuple(sorted((edge.source_id, edge.target_id)))
                    if pair not in seen:
                        seen.add(pair)
                        conflicts.append((edge.source_id, edge.target_id))

        return conflicts

    def get_method_lineage(self, result_id: str) -> list[KnowledgeNode]:
        """Trace the lineage from a result back to all contributing methods.

        Walks backwards along PRODUCED and DEPENDS_ON edges to collect all
        method (and system) nodes that contributed to the result.

        Parameters
        ----------
        result_id : str
            The result node to trace back from.

        Returns
        -------
        list[KnowledgeNode]
            Ordered list of ancestor nodes (methods, systems) found via
            backward traversal. The result node itself is excluded.
        """
        if result_id not in self._nodes:
            return []

        visited: set[str] = set()
        lineage: list[KnowledgeNode] = []
        queue: deque[str] = deque([result_id])

        while queue:
            current_id = queue.popleft()
            if current_id in visited:
                continue
            visited.add(current_id)

            # Walk backward: who produced this? (incoming PRODUCED edges)
            for edge in self._edges_by_target.get(current_id, []):
                if edge.edge_type in (EdgeType.PRODUCED, EdgeType.DEPENDS_ON):
                    src = edge.source_id
                    if src not in visited and src in self._nodes:
                        lineage.append(self._nodes[src])
                        queue.append(src)

            # If current is a method, also check for ANALYZED_BY going into it
            current_node = self._nodes.get(current_id)
            if current_node and current_node.node_type == NodeType.METHOD:
                for edge in self._edges_by_target.get(current_id, []):
                    if edge.edge_type == EdgeType.ANALYZED_BY:
                        src = edge.source_id
                        if src not in visited and src in self._nodes:
                            lineage.append(self._nodes[src])
                            queue.append(src)

        return lineage

    def get_decision_provenance(self, decision_id: str) -> dict[str, Any]:
        """Trace the full provenance chain for a decision node.

        Returns a dict with:
        - ``decision``: the decision node data
        - ``icm_scores``: ICM score nodes that led to the decision
        - ``methods``: methods involved in producing ICM inputs
        - ``systems``: systems that were analyzed
        - ``results``: results that fed into the decision chain

        Parameters
        ----------
        decision_id : str
            The decision node ID.

        Returns
        -------
        dict[str, Any]
            Provenance record with categorized contributing nodes.
        """
        decision_node = self._nodes.get(decision_id)
        if decision_node is None:
            return {}

        provenance: dict[str, Any] = {
            "decision": {
                "id": decision_node.id,
                "data": decision_node.data,
                "metadata": decision_node.metadata,
            },
            "icm_scores": [],
            "methods": [],
            "systems": [],
            "results": [],
        }

        # BFS backward from the decision
        visited: set[str] = {decision_id}
        queue: deque[str] = deque()

        # Find all incoming edges to decision
        for edge in self._edges_by_target.get(decision_id, []):
            if edge.source_id not in visited:
                queue.append(edge.source_id)

        while queue:
            current_id = queue.popleft()
            if current_id in visited:
                continue
            visited.add(current_id)

            node = self._nodes.get(current_id)
            if node is None:
                continue

            entry = {"id": node.id, "data": node.data, "metadata": node.metadata}
            if node.node_type == NodeType.ICM_SCORE:
                provenance["icm_scores"].append(entry)
            elif node.node_type == NodeType.METHOD:
                provenance["methods"].append(entry)
            elif node.node_type == NodeType.SYSTEM:
                provenance["systems"].append(entry)
            elif node.node_type == NodeType.RESULT:
                provenance["results"].append(entry)

            # Continue backward traversal
            for edge in self._edges_by_target.get(current_id, []):
                if edge.source_id not in visited:
                    queue.append(edge.source_id)

        return provenance

    # ----------------------------------------------------------------
    # Serialization
    # ----------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialize the graph to a plain dict (JSON-compatible).

        Returns
        -------
        dict
            ``{"nodes": [...], "edges": [...]}`` where each node/edge is
            represented as a dict with string-typed enum values.
        """
        nodes_list: list[dict[str, Any]] = []
        for node in self._nodes.values():
            nodes_list.append({
                "id": node.id,
                "node_type": node.node_type.value,
                "data": node.data,
                "timestamp": node.timestamp,
                "metadata": node.metadata,
            })

        edges_list: list[dict[str, Any]] = []
        seen_edges: set[int] = set()
        for edge_list in self._edges_by_source.values():
            for edge in edge_list:
                eid = id(edge)
                if eid in seen_edges:
                    continue
                seen_edges.add(eid)
                edges_list.append({
                    "source_id": edge.source_id,
                    "target_id": edge.target_id,
                    "edge_type": edge.edge_type.value,
                    "weight": edge.weight,
                    "metadata": edge.metadata,
                })

        return {"nodes": nodes_list, "edges": edges_list}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "KnowledgeGraph":
        """Deserialize a graph from a dict produced by ``to_dict()``.

        Parameters
        ----------
        data : dict
            Dict with ``"nodes"`` and ``"edges"`` keys.

        Returns
        -------
        KnowledgeGraph
        """
        graph = cls()
        for nd in data.get("nodes", []):
            node = KnowledgeNode(
                id=nd["id"],
                node_type=NodeType(nd["node_type"]),
                data=nd.get("data", {}),
                timestamp=nd.get("timestamp", 0.0),
                metadata=nd.get("metadata", {}),
            )
            graph.add_node(node)

        for ed in data.get("edges", []):
            edge = KnowledgeEdge(
                source_id=ed["source_id"],
                target_id=ed["target_id"],
                edge_type=EdgeType(ed["edge_type"]),
                weight=ed.get("weight", 1.0),
                metadata=ed.get("metadata", {}),
            )
            graph.add_edge(edge)

        return graph

    # ----------------------------------------------------------------
    # Stats
    # ----------------------------------------------------------------

    def summary(self) -> dict[str, int]:
        """Return summary statistics about the graph.

        Returns
        -------
        dict[str, int]
            Counts for total nodes, total edges, and per-type breakdowns.
        """
        total_edges = sum(len(edges) for edges in self._edges_by_source.values())

        type_counts: dict[str, int] = {}
        for node in self._nodes.values():
            key = f"nodes_{node.node_type.value}"
            type_counts[key] = type_counts.get(key, 0) + 1

        edge_type_counts: dict[str, int] = {}
        for edge_list in self._edges_by_source.values():
            for edge in edge_list:
                key = f"edges_{edge.edge_type.value}"
                edge_type_counts[key] = edge_type_counts.get(key, 0) + 1

        return {
            "total_nodes": len(self._nodes),
            "total_edges": total_edges,
            **type_counts,
            **edge_type_counts,
        }

    def __len__(self) -> int:
        """Return the number of nodes in the graph."""
        return len(self._nodes)

    def __contains__(self, node_id: str) -> bool:
        """Check if a node ID exists in the graph."""
        return node_id in self._nodes

    def __repr__(self) -> str:
        total_edges = sum(len(e) for e in self._edges_by_source.values())
        return (
            f"KnowledgeGraph(nodes={len(self._nodes)}, edges={total_edges})"
        )


# Sentinel for missing dict keys
_SENTINEL = object()
