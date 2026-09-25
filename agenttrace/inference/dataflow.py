"""
Data flow analysis for causal relationship inference.

Analyzes how data flows between nodes to infer causal relationships.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any
from difflib import SequenceMatcher

from agenttrace.core.graph import CausalGraph
from agenttrace.core.node import Node, NodeType
from agenttrace.core.edge import Edge, EdgeType


class DataFlowAnalyzer:
    """
    Analyzes data flow patterns to infer causal relationships.

    This analyzer examines the data in nodes to determine if
    output from one node was used as input to another.
    """

    def __init__(
        self,
        similarity_threshold: float = 0.7,
        exact_match_bonus: float = 0.2,
    ):
        """
        Initialize the data flow analyzer.

        Args:
            similarity_threshold: Minimum similarity score to consider a match
            exact_match_bonus: Bonus confidence for exact matches
        """
        self.similarity_threshold = similarity_threshold
        self.exact_match_bonus = exact_match_bonus

    def analyze(self, graph: CausalGraph) -> list[Edge]:
        """
        Analyze the graph and infer data flow relationships.

        Args:
            graph: The causal graph to analyze

        Returns:
            List of inferred edges
        """
        inferred_edges: list[Edge] = []
        nodes = list(graph)

        # Sort by timestamp for proper ordering
        nodes.sort(key=lambda n: n.timestamp)

        # Compare each pair of nodes
        for i, source in enumerate(nodes):
            if source.type not in (NodeType.AGENT_OUTPUT, NodeType.TOOL_RESULT):
                continue

            for target in nodes[i + 1:]:
                if target.type not in (NodeType.AGENT_INPUT, NodeType.TOOL_CALL):
                    continue

                # Skip if already connected
                if target.id in source.parent_ids or self._are_connected(graph, source.id, target.id):
                    continue

                # Check for data flow
                confidence = self._compute_data_flow_confidence(source.data, target.data)

                if confidence >= self.similarity_threshold:
                    edge = Edge(
                        source_id=source.id,
                        target_id=target.id,
                        type=EdgeType.DATA_FLOW,
                        confidence=min(confidence, 1.0),
                        metadata={
                            "inference_method": "data_flow_analysis",
                            "similarity_score": confidence,
                        },
                    )
                    inferred_edges.append(edge)

        return inferred_edges

    def _are_connected(self, graph: CausalGraph, source_id: str, target_id: str) -> bool:
        """Check if two nodes are already connected."""
        path = graph.find_path(source_id, target_id)
        return path is not None

    def _compute_data_flow_confidence(self, source_data: Any, target_data: Any) -> float:
        """
        Compute confidence that data flowed from source to target.

        Args:
            source_data: Data from source node
            target_data: Data from target node

        Returns:
            Confidence score between 0 and 1
        """
        # Handle direct equality
        if source_data == target_data:
            return 1.0

        # Convert to string for comparison
        source_str = self._data_to_string(source_data)
        target_str = self._data_to_string(target_data)

        # Check for substring containment
        if source_str in target_str or target_str in source_str:
            containment_ratio = min(len(source_str), len(target_str)) / max(len(source_str), len(target_str), 1)
            return 0.8 + (0.2 * containment_ratio)

        # Check for structural similarity (dict/list patterns)
        if isinstance(source_data, dict) and isinstance(target_data, dict):
            return self._dict_similarity(source_data, target_data)

        if isinstance(source_data, (list, tuple)) and isinstance(target_data, (list, tuple)):
            return self._list_similarity(list(source_data), list(target_data))

        # Fall back to string similarity
        return self._string_similarity(source_str, target_str)

    def _data_to_string(self, data: Any) -> str:
        """Convert data to a normalized string representation."""
        if isinstance(data, str):
            return data
        try:
            return json.dumps(data, sort_keys=True, default=str)
        except (TypeError, ValueError):
            return str(data)

    def _string_similarity(self, s1: str, s2: str) -> float:
        """Compute string similarity using SequenceMatcher."""
        if not s1 or not s2:
            return 0.0
        return SequenceMatcher(None, s1, s2).ratio()

    def _dict_similarity(self, d1: dict, d2: dict) -> float:
        """Compute similarity between two dictionaries."""
        if not d1 or not d2:
            return 0.0

        keys1 = set(d1.keys())
        keys2 = set(d2.keys())

        common_keys = keys1 & keys2
        all_keys = keys1 | keys2

        if not all_keys:
            return 0.0

        # Key overlap score
        key_score = len(common_keys) / len(all_keys)

        # Value similarity for common keys
        value_scores = []
        for key in common_keys:
            value_scores.append(
                self._compute_data_flow_confidence(d1[key], d2[key])
            )

        if value_scores:
            value_score = sum(value_scores) / len(value_scores)
        else:
            value_score = 0.0

        return (key_score * 0.4) + (value_score * 0.6)

    def _list_similarity(self, l1: list, l2: list) -> float:
        """Compute similarity between two lists."""
        if not l1 or not l2:
            return 0.0

        # Check for element overlap
        scores = []
        for item1 in l1[:10]:  # Limit comparison size
            best_score = 0.0
            for item2 in l2[:10]:
                score = self._compute_data_flow_confidence(item1, item2)
                best_score = max(best_score, score)
            scores.append(best_score)

        if scores:
            return sum(scores) / len(scores)
        return 0.0

    def find_data_sources(
        self,
        graph: CausalGraph,
        target_node_id: str,
        data_key: str | None = None,
    ) -> list[tuple[Node, float]]:
        """
        Find potential source nodes for data in a target node.

        Args:
            graph: The causal graph
            target_node_id: ID of the target node
            data_key: Optional specific key to search for in dict data

        Returns:
            List of (node, confidence) tuples for potential sources
        """
        target = graph.get_node(target_node_id)
        if target is None:
            return []

        # Get the data to search for
        if data_key and isinstance(target.data, dict):
            search_data = target.data.get(data_key)
        else:
            search_data = target.data

        results: list[tuple[Node, float]] = []

        # Check all nodes before this one
        for node in graph:
            if node.timestamp >= target.timestamp:
                continue

            if node.type not in (NodeType.AGENT_OUTPUT, NodeType.TOOL_RESULT, NodeType.STATE_READ):
                continue

            # Check if this node's data matches
            node_data = node.data
            if data_key and isinstance(node_data, dict):
                node_data = node_data.get(data_key, node_data)

            confidence = self._compute_data_flow_confidence(node_data, search_data)
            if confidence >= self.similarity_threshold:
                results.append((node, confidence))

        # Sort by confidence
        results.sort(key=lambda x: x[1], reverse=True)
        return results


def compute_data_fingerprint(data: Any) -> str:
    """
    Compute a fingerprint for data to enable fast comparison.

    Args:
        data: The data to fingerprint

    Returns:
        A hex string fingerprint
    """
    if isinstance(data, str):
        content = data
    else:
        try:
            content = json.dumps(data, sort_keys=True, default=str)
        except (TypeError, ValueError):
            content = str(data)

    return hashlib.md5(content.encode()).hexdigest()
