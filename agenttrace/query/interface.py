"""
High-level query interface for causal analysis.

Provides a simple API for common debugging queries.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Callable

from agenttrace.core.graph import CausalGraph
from agenttrace.core.node import Node, NodeType
from agenttrace.core.edge import Edge, EdgeType
from agenttrace.inference.engine import InferenceEngine


class QueryInterface:
    """
    High-level query interface for causal trace analysis.

    Provides simple methods for common debugging operations.
    """

    def __init__(self, graph: CausalGraph):
        """
        Initialize the query interface.

        Args:
            graph: The causal graph to query
        """
        self.graph = graph
        self._inference_engine = InferenceEngine()

    # === Forward Tracing ===

    def what_happened_after(
        self,
        node_id: str,
        max_depth: int = -1,
    ) -> list[Node]:
        """
        Find everything that happened as a result of a node.

        Args:
            node_id: The starting node
            max_depth: Maximum steps to follow

        Returns:
            List of affected nodes
        """
        return self.graph.trace_forward(node_id, max_depth)

    def what_agents_were_affected(
        self,
        node_id: str,
    ) -> dict[str, list[Node]]:
        """
        Find which agents were affected by a node.

        Args:
            node_id: The starting node

        Returns:
            Dictionary mapping agent_id to list of affected nodes
        """
        affected = self.graph.trace_forward(node_id)
        by_agent: dict[str, list[Node]] = {}

        for node in affected:
            if node.agent_id not in by_agent:
                by_agent[node.agent_id] = []
            by_agent[node.agent_id].append(node)

        return by_agent

    def what_tools_were_called(
        self,
        node_id: str,
    ) -> list[Node]:
        """
        Find all tool calls that resulted from a node.

        Args:
            node_id: The starting node

        Returns:
            List of tool call nodes
        """
        affected = self.graph.trace_forward(node_id)
        return [n for n in affected if n.type == NodeType.TOOL_CALL]

    # === Backward Tracing ===

    def why_did_this_happen(
        self,
        node_id: str,
        max_depth: int = -1,
    ) -> list[Node]:
        """
        Find all causes of a node.

        Args:
            node_id: The node to explain
            max_depth: Maximum steps to trace back

        Returns:
            List of causal nodes
        """
        return self.graph.trace_backward(node_id, max_depth)

    def what_was_the_root_cause(
        self,
        node_id: str,
    ) -> list[Node]:
        """
        Find the root causes (initial triggers) for a node.

        Args:
            node_id: The node to analyze

        Returns:
            List of root cause nodes
        """
        return self.graph.find_root_causes(node_id)

    def how_did_we_get_here(
        self,
        node_id: str,
    ) -> list[list[Node]]:
        """
        Get all paths that led to a node.

        Args:
            node_id: The destination node

        Returns:
            List of paths (each path is a list of nodes)
        """
        return self.graph.get_causal_chain(node_id)

    def what_data_influenced_this(
        self,
        node_id: str,
    ) -> list[tuple[Node, float]]:
        """
        Find nodes whose data influenced this node.

        Args:
            node_id: The node to analyze

        Returns:
            List of (node, confidence) tuples
        """
        return self._inference_engine.dataflow_analyzer.find_data_sources(
            self.graph, node_id
        )

    # === Error Analysis ===

    def explain_error(
        self,
        error_node_id: str,
    ) -> dict[str, Any]:
        """
        Get a comprehensive explanation of an error.

        Args:
            error_node_id: ID of an error node

        Returns:
            Explanation dictionary with causes and suggestions
        """
        node = self.graph.get_node(error_node_id)
        if node is None:
            return {"error": "Node not found"}

        if node.type != NodeType.ERROR:
            return {"error": "Not an error node"}

        root_causes = self.what_was_the_root_cause(error_node_id)
        chains = self.how_did_we_get_here(error_node_id)
        suggestions = self._inference_engine.suggest_likely_causes(
            self.graph, error_node_id
        )

        return {
            "error_info": {
                "type": node.data.get("error_type", "Unknown"),
                "message": node.data.get("error_message", ""),
                "agent": node.agent_id,
                "timestamp": node.timestamp.isoformat(),
            },
            "root_causes": [
                {
                    "node_id": n.id,
                    "agent": n.agent_id,
                    "type": n.type.value,
                    "summary": n.summary,
                }
                for n in root_causes
            ],
            "causal_depth": max(len(c) for c in chains) if chains else 0,
            "likely_causes": suggestions,
        }

    def find_all_errors(self) -> list[Node]:
        """Find all error nodes in the graph."""
        return self.graph.get_nodes_by_type(NodeType.ERROR)

    # === Counterfactual Analysis ===

    def what_if(
        self,
        node_id: str,
        alternative_data: Any,
    ) -> dict[str, Any]:
        """
        Analyze what would have happened with different data.

        Args:
            node_id: The node to modify
            alternative_data: The alternative value

        Returns:
            Comparison between original and counterfactual
        """
        cf_graph = self.graph.create_counterfactual(node_id, alternative_data)
        return self.graph.compare_with(cf_graph)

    def compare_decisions(
        self,
        node_id1: str,
        node_id2: str,
    ) -> dict[str, Any]:
        """
        Compare the effects of two different nodes.

        Args:
            node_id1: First node
            node_id2: Second node

        Returns:
            Comparison of their effects
        """
        effects1 = set(n.id for n in self.what_happened_after(node_id1))
        effects2 = set(n.id for n in self.what_happened_after(node_id2))

        return {
            "node1_only_effects": list(effects1 - effects2),
            "node2_only_effects": list(effects2 - effects1),
            "common_effects": list(effects1 & effects2),
            "node1_effect_count": len(effects1),
            "node2_effect_count": len(effects2),
        }

    # === Filtering and Searching ===

    def find_nodes(
        self,
        predicate: Callable[[Node], bool],
    ) -> list[Node]:
        """
        Find nodes matching a predicate.

        Args:
            predicate: Function that returns True for matching nodes

        Returns:
            List of matching nodes
        """
        return [n for n in self.graph if predicate(n)]

    def find_by_agent(self, agent_id: str) -> list[Node]:
        """Find all nodes from a specific agent."""
        return self.graph.get_nodes_by_agent(agent_id)

    def find_by_type(self, node_type: NodeType) -> list[Node]:
        """Find all nodes of a specific type."""
        return self.graph.get_nodes_by_type(node_type)

    def find_in_time_range(
        self,
        start: datetime,
        end: datetime,
    ) -> list[Node]:
        """Find nodes within a time range."""
        return self.graph.get_nodes_in_time_range(start, end)

    def search_data(
        self,
        keyword: str,
        case_sensitive: bool = False,
    ) -> list[Node]:
        """
        Search for nodes containing a keyword in their data.

        Args:
            keyword: The keyword to search for
            case_sensitive: Whether to match case

        Returns:
            List of matching nodes
        """
        if not case_sensitive:
            keyword = keyword.lower()

        def matches(node: Node) -> bool:
            data_str = str(node.data)
            if not case_sensitive:
                data_str = data_str.lower()
            return keyword in data_str

        return self.find_nodes(matches)

    # === Path Analysis ===

    def find_path_between(
        self,
        source_id: str,
        target_id: str,
    ) -> list[Node] | None:
        """
        Find the shortest causal path between two nodes.

        Args:
            source_id: Starting node
            target_id: Ending node

        Returns:
            Path as list of nodes, or None if no path exists
        """
        return self.graph.find_path(source_id, target_id)

    def are_causally_related(
        self,
        node_id1: str,
        node_id2: str,
    ) -> bool:
        """
        Check if two nodes are causally related.

        Args:
            node_id1: First node
            node_id2: Second node

        Returns:
            True if there's a path between them (in either direction)
        """
        path1 = self.graph.find_path(node_id1, node_id2)
        path2 = self.graph.find_path(node_id2, node_id1)
        return path1 is not None or path2 is not None

    # === Statistics ===

    def get_agent_summary(self) -> dict[str, dict[str, Any]]:
        """Get a summary of activity by agent."""
        summary: dict[str, dict[str, Any]] = {}

        for node in self.graph:
            if node.agent_id not in summary:
                summary[node.agent_id] = {
                    "node_count": 0,
                    "error_count": 0,
                    "types": {},
                }

            summary[node.agent_id]["node_count"] += 1

            if node.type == NodeType.ERROR:
                summary[node.agent_id]["error_count"] += 1

            type_name = node.type.value
            if type_name not in summary[node.agent_id]["types"]:
                summary[node.agent_id]["types"][type_name] = 0
            summary[node.agent_id]["types"][type_name] += 1

        return summary

    def get_timeline(self) -> list[dict[str, Any]]:
        """Get a chronological timeline of events."""
        nodes = list(self.graph)
        nodes.sort(key=lambda n: n.timestamp)

        return [
            {
                "timestamp": n.timestamp.isoformat(),
                "node_id": n.id,
                "agent_id": n.agent_id,
                "type": n.type.value,
                "summary": n.summary,
            }
            for n in nodes
        ]
