"""
Causal Trace Graph implementation.

The graph maintains all nodes and edges, supporting:
- Forward tracing: What did this decision affect?
- Backward tracing: How did this result come about?
- Counterfactual analysis: What if we had chosen differently?
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Iterator, Callable
from collections import defaultdict

import networkx as nx

from agenttrace.core.node import Node, NodeType
from agenttrace.core.edge import Edge, EdgeType


class CausalGraph:
    """
    A directed graph representing causal relationships in a multi-agent system.

    Uses NetworkX as the underlying graph implementation, with extensions
    for temporal queries and causal analysis.
    """

    def __init__(self, run_id: str | None = None):
        """
        Initialize a new causal graph.

        Args:
            run_id: Optional identifier for this execution run
        """
        self._graph = nx.DiGraph()
        self._nodes: dict[str, Node] = {}
        self._edges: dict[str, Edge] = {}
        self._run_id = run_id or self._generate_run_id()

        # Indexes for efficient queries
        self._nodes_by_agent: dict[str, list[str]] = defaultdict(list)
        self._nodes_by_type: dict[NodeType, list[str]] = defaultdict(list)
        self._nodes_by_time: list[tuple[datetime, str]] = []

    @staticmethod
    def _generate_run_id() -> str:
        """Generate a unique run ID."""
        import uuid

        return f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"

    @property
    def run_id(self) -> str:
        return self._run_id

    def add_node(self, node: Node) -> Node:
        """
        Add a node to the graph.

        Args:
            node: The node to add

        Returns:
            The added node (with run_id set)
        """
        node.run_id = self._run_id
        self._nodes[node.id] = node
        self._graph.add_node(node.id, data=node)

        # Update indexes
        self._nodes_by_agent[node.agent_id].append(node.id)
        self._nodes_by_type[node.type].append(node.id)
        self._nodes_by_time.append((node.timestamp, node.id))
        self._nodes_by_time.sort(key=lambda x: x[0])

        # Auto-create edges from parent_ids
        for parent_id in node.parent_ids:
            if parent_id in self._nodes:
                self.add_edge(
                    Edge(
                        source_id=parent_id,
                        target_id=node.id,
                        type=EdgeType.DATA_FLOW,
                    )
                )

        return node

    def add_edge(self, edge: Edge) -> Edge:
        """
        Add an edge to the graph.

        Args:
            edge: The edge to add

        Returns:
            The added edge
        """
        if edge.source_id not in self._nodes:
            raise ValueError(f"Source node {edge.source_id} not found")
        if edge.target_id not in self._nodes:
            raise ValueError(f"Target node {edge.target_id} not found")

        self._edges[edge.id] = edge
        self._graph.add_edge(
            edge.source_id,
            edge.target_id,
            id=edge.id,
            data=edge,
        )
        return edge

    def get_node(self, node_id: str) -> Node | None:
        """Get a node by ID."""
        return self._nodes.get(node_id)

    def get_edge(self, edge_id: str) -> Edge | None:
        """Get an edge by ID."""
        return self._edges.get(edge_id)

    def get_all_edges(self) -> list[Edge]:
        """Get all edges in the graph."""
        return list(self._edges.values())

    def get_nodes_by_agent(self, agent_id: str) -> list[Node]:
        """Get all nodes created by a specific agent."""
        return [self._nodes[nid] for nid in self._nodes_by_agent.get(agent_id, [])]

    def get_nodes_by_type(self, node_type: NodeType) -> list[Node]:
        """Get all nodes of a specific type."""
        return [self._nodes[nid] for nid in self._nodes_by_type.get(node_type, [])]

    def get_nodes_in_time_range(self, start: datetime, end: datetime) -> list[Node]:
        """Get all nodes within a time range."""
        result = []
        for ts, node_id in self._nodes_by_time:
            if start <= ts <= end:
                result.append(self._nodes[node_id])
            elif ts > end:
                break
        return result

    # === Query Operations ===

    def trace_forward(
        self, node_id: str, max_depth: int = -1, edge_filter: Callable[[Edge], bool] | None = None
    ) -> list[Node]:
        """
        Forward tracing: Find all nodes affected by this node.

        Args:
            node_id: Starting node ID
            max_depth: Maximum traversal depth (-1 for unlimited)
            edge_filter: Optional filter for edges to follow

        Returns:
            List of affected nodes in causal order
        """
        if node_id not in self._nodes:
            return []

        visited = set()
        result = []
        queue = [(node_id, 0)]

        while queue:
            current_id, depth = queue.pop(0)
            if current_id in visited:
                continue
            if max_depth >= 0 and depth > max_depth:
                continue

            visited.add(current_id)
            if current_id != node_id:
                result.append(self._nodes[current_id])

            # Get successors
            for successor_id in self._graph.successors(current_id):
                edge_data = self._graph.edges[current_id, successor_id]
                edge = edge_data.get("data")

                if edge_filter is None or (edge and edge_filter(edge)):
                    queue.append((successor_id, depth + 1))

        return result

    def trace_backward(
        self, node_id: str, max_depth: int = -1, edge_filter: Callable[[Edge], bool] | None = None
    ) -> list[Node]:
        """
        Backward tracing: Find all nodes that caused this node.

        Args:
            node_id: Starting node ID
            max_depth: Maximum traversal depth (-1 for unlimited)
            edge_filter: Optional filter for edges to follow

        Returns:
            List of causal nodes in reverse causal order
        """
        if node_id not in self._nodes:
            return []

        visited = set()
        result = []
        queue = [(node_id, 0)]

        while queue:
            current_id, depth = queue.pop(0)
            if current_id in visited:
                continue
            if max_depth >= 0 and depth > max_depth:
                continue

            visited.add(current_id)
            if current_id != node_id:
                result.append(self._nodes[current_id])

            # Get predecessors
            for predecessor_id in self._graph.predecessors(current_id):
                edge_data = self._graph.edges[predecessor_id, current_id]
                edge = edge_data.get("data")

                if edge_filter is None or (edge and edge_filter(edge)):
                    queue.append((predecessor_id, depth + 1))

        return result

    def find_root_causes(self, node_id: str) -> list[Node]:
        """
        Find the root causes of a node (nodes with no predecessors in the trace).

        Args:
            node_id: The node to analyze

        Returns:
            List of root cause nodes
        """
        causes = self.trace_backward(node_id)
        root_causes = []

        for node in causes:
            predecessors = list(self._graph.predecessors(node.id))
            if not predecessors:
                root_causes.append(node)

        return root_causes

    def find_path(self, source_id: str, target_id: str) -> list[Node] | None:
        """
        Find the causal path between two nodes.

        Args:
            source_id: Starting node ID
            target_id: Target node ID

        Returns:
            List of nodes in the path, or None if no path exists
        """
        try:
            path_ids = nx.shortest_path(self._graph, source_id, target_id)
            return [self._nodes[nid] for nid in path_ids]
        except nx.NetworkXNoPath:
            return None

    def get_causal_chain(self, node_id: str) -> list[list[Node]]:
        """
        Get all causal chains leading to a node.

        Returns a list of paths, where each path is a list of nodes
        from a root cause to the target node.
        """
        if node_id not in self._nodes:
            return []

        # Find all root causes first
        root_causes = self.find_root_causes(node_id)
        if not root_causes:
            # The node itself might be a root
            return [[self._nodes[node_id]]]

        chains = []
        for root in root_causes:
            try:
                for path in nx.all_simple_paths(self._graph, root.id, node_id):
                    chains.append([self._nodes[nid] for nid in path])
            except nx.NetworkXNoPath:
                continue

        return chains

    # === Counterfactual Analysis ===

    def create_counterfactual(self, node_id: str, alternative_data: Any) -> "CausalGraph":
        """
        Create a counterfactual graph where a node has different data.

        This creates a copy of the graph with the specified node modified,
        useful for "what-if" analysis.

        Args:
            node_id: The node to modify
            alternative_data: The alternative data for the node

        Returns:
            A new CausalGraph with the counterfactual modification
        """
        if node_id not in self._nodes:
            raise ValueError(f"Node {node_id} not found")

        # Create a new graph
        cf_graph = CausalGraph(run_id=f"{self._run_id}_cf_{node_id[:8]}")

        # Copy all nodes, modifying the target
        for nid, node in self._nodes.items():
            if nid == node_id:
                cf_node = Node(
                    id=node.id,
                    type=node.type,
                    agent_id=node.agent_id,
                    timestamp=node.timestamp,
                    data=alternative_data,
                    metadata={**node.metadata, "counterfactual": True, "original_data": node.data},
                    parent_ids=node.parent_ids.copy(),
                )
            else:
                cf_node = Node(
                    id=node.id,
                    type=node.type,
                    agent_id=node.agent_id,
                    timestamp=node.timestamp,
                    data=node.data,
                    metadata=node.metadata.copy(),
                    parent_ids=node.parent_ids.copy(),
                )
            cf_graph._nodes[cf_node.id] = cf_node
            cf_graph._graph.add_node(cf_node.id, data=cf_node)
            cf_graph._nodes_by_agent[cf_node.agent_id].append(cf_node.id)
            cf_graph._nodes_by_type[cf_node.type].append(cf_node.id)

        # Copy all edges
        for eid, edge in self._edges.items():
            cf_edge = Edge(
                id=edge.id,
                source_id=edge.source_id,
                target_id=edge.target_id,
                type=edge.type,
                confidence=edge.confidence,
                metadata=edge.metadata.copy(),
            )
            cf_graph._edges[cf_edge.id] = cf_edge
            cf_graph._graph.add_edge(
                cf_edge.source_id,
                cf_edge.target_id,
                id=cf_edge.id,
                data=cf_edge,
            )

        return cf_graph

    def compare_with(self, other: "CausalGraph") -> dict[str, Any]:
        """
        Compare this graph with another (e.g., a counterfactual).

        Returns:
            Dictionary with comparison results
        """
        self_nodes = set(self._nodes.keys())
        other_nodes = set(other._nodes.keys())

        common_nodes = self_nodes & other_nodes
        different_data = []

        for node_id in common_nodes:
            self_node = self._nodes[node_id]
            other_node = other._nodes[node_id]
            if self_node.data != other_node.data:
                different_data.append(
                    {
                        "node_id": node_id,
                        "self_data": self_node.data,
                        "other_data": other_node.data,
                    }
                )

        return {
            "common_nodes": len(common_nodes),
            "self_only_nodes": list(self_nodes - other_nodes),
            "other_only_nodes": list(other_nodes - self_nodes),
            "different_data": different_data,
        }

    # === Serialization ===

    def to_dict(self) -> dict[str, Any]:
        """Convert graph to dictionary for serialization."""
        return {
            "run_id": self._run_id,
            "nodes": [n.to_dict() for n in self._nodes.values()],
            "edges": [e.to_dict() for e in self._edges.values()],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "CausalGraph":
        """Create graph from dictionary."""
        graph = cls(run_id=d["run_id"])

        # Add nodes first
        for node_dict in d["nodes"]:
            node = Node.from_dict(node_dict)
            graph._nodes[node.id] = node
            graph._graph.add_node(node.id, data=node)
            graph._nodes_by_agent[node.agent_id].append(node.id)
            graph._nodes_by_type[node.type].append(node.id)
            graph._nodes_by_time.append((node.timestamp, node.id))

        graph._nodes_by_time.sort(key=lambda x: x[0])

        # Then add edges
        for edge_dict in d["edges"]:
            edge = Edge.from_dict(edge_dict)
            graph._edges[edge.id] = edge
            graph._graph.add_edge(
                edge.source_id,
                edge.target_id,
                id=edge.id,
                data=edge,
            )

        return graph

    def to_json(self, indent: int = 2) -> str:
        """Convert graph to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)

    @classmethod
    def from_json(cls, json_str: str) -> "CausalGraph":
        """Create graph from JSON string."""
        return cls.from_dict(json.loads(json_str))

    # === Statistics ===

    @property
    def node_count(self) -> int:
        return len(self._nodes)

    @property
    def edge_count(self) -> int:
        return len(self._edges)

    def get_statistics(self) -> dict[str, Any]:
        """Get graph statistics."""
        return {
            "run_id": self._run_id,
            "node_count": self.node_count,
            "edge_count": self.edge_count,
            "agents": list(self._nodes_by_agent.keys()),
            "node_types": {t.value: len(ids) for t, ids in self._nodes_by_type.items()},
            "is_dag": nx.is_directed_acyclic_graph(self._graph),
        }

    def __len__(self) -> int:
        return self.node_count

    def __contains__(self, node_id: str) -> bool:
        return node_id in self._nodes

    def __iter__(self) -> Iterator[Node]:
        return iter(self._nodes.values())
