"""
Temporal analysis for causal relationship inference.

Analyzes timing relationships between nodes to infer causality
based on temporal proximity and ordering.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

from agenttrace.core.graph import CausalGraph
from agenttrace.core.node import Node, NodeType
from agenttrace.core.edge import Edge, EdgeType


class TemporalAnalyzer:
    """
    Analyzes temporal patterns to infer causal relationships.

    Uses the principle that causes must precede effects, and
    closer temporal proximity suggests stronger causation.
    """

    def __init__(
        self,
        max_time_delta: timedelta = timedelta(seconds=10),
        decay_factor: float = 0.1,
    ):
        """
        Initialize the temporal analyzer.

        Args:
            max_time_delta: Maximum time gap to consider for causation
            decay_factor: How quickly confidence decays with time
        """
        self.max_time_delta = max_time_delta
        self.decay_factor = decay_factor

    def analyze(self, graph: CausalGraph) -> list[Edge]:
        """
        Analyze temporal patterns and infer causal relationships.

        Args:
            graph: The causal graph to analyze

        Returns:
            List of inferred temporal edges
        """
        inferred_edges: list[Edge] = []
        nodes = list(graph)

        # Sort by timestamp
        nodes.sort(key=lambda n: n.timestamp)

        # Group nodes by agent for intra-agent temporal analysis
        agent_nodes: dict[str, list[Node]] = {}
        for node in nodes:
            if node.agent_id not in agent_nodes:
                agent_nodes[node.agent_id] = []
            agent_nodes[node.agent_id].append(node)

        # Analyze intra-agent temporal patterns
        for agent_id, agent_node_list in agent_nodes.items():
            edges = self._analyze_agent_sequence(agent_node_list, graph)
            inferred_edges.extend(edges)

        # Analyze inter-agent temporal patterns (trigger-response)
        inter_edges = self._analyze_inter_agent_patterns(nodes, graph)
        inferred_edges.extend(inter_edges)

        return inferred_edges

    def _analyze_agent_sequence(
        self,
        nodes: list[Node],
        graph: CausalGraph,
    ) -> list[Edge]:
        """Analyze temporal sequence within an agent."""
        edges: list[Edge] = []

        for i, node in enumerate(nodes[:-1]):
            next_node = nodes[i + 1]

            # Check if already connected
            if self._are_connected(graph, node.id, next_node.id):
                continue

            # Check time delta
            delta = next_node.timestamp - node.timestamp
            if delta > self.max_time_delta:
                continue

            # Compute confidence based on temporal proximity
            confidence = self._compute_temporal_confidence(delta)

            # Only create edge for meaningful transitions
            if self._is_meaningful_transition(node.type, next_node.type):
                edge = Edge(
                    source_id=node.id,
                    target_id=next_node.id,
                    type=EdgeType.TEMPORAL,
                    confidence=confidence,
                    metadata={
                        "inference_method": "temporal_analysis",
                        "time_delta_ms": delta.total_seconds() * 1000,
                    },
                )
                edges.append(edge)

        return edges

    def _analyze_inter_agent_patterns(
        self,
        nodes: list[Node],
        graph: CausalGraph,
    ) -> list[Edge]:
        """Analyze trigger-response patterns between agents."""
        edges: list[Edge] = []

        # Find output nodes that might trigger responses
        outputs = [n for n in nodes if n.type in (
            NodeType.AGENT_OUTPUT, NodeType.TOOL_RESULT
        )]

        for output in outputs:
            # Find inputs from other agents that occurred shortly after
            for node in nodes:
                if node.timestamp <= output.timestamp:
                    continue

                if node.agent_id == output.agent_id:
                    continue

                if node.type != NodeType.AGENT_INPUT:
                    continue

                # Check if already connected
                if self._are_connected(graph, output.id, node.id):
                    continue

                delta = node.timestamp - output.timestamp
                if delta > self.max_time_delta:
                    break  # Nodes are sorted, no point checking further

                confidence = self._compute_temporal_confidence(delta) * 0.8  # Lower base confidence for inter-agent

                edge = Edge(
                    source_id=output.id,
                    target_id=node.id,
                    type=EdgeType.TRIGGER_RESPONSE,
                    confidence=confidence,
                    metadata={
                        "inference_method": "temporal_analysis",
                        "pattern": "inter_agent_trigger",
                        "time_delta_ms": delta.total_seconds() * 1000,
                    },
                )
                edges.append(edge)

        return edges

    def _are_connected(self, graph: CausalGraph, source_id: str, target_id: str) -> bool:
        """Check if two nodes are already connected."""
        path = graph.find_path(source_id, target_id)
        return path is not None

    def _compute_temporal_confidence(self, delta: timedelta) -> float:
        """
        Compute confidence based on temporal proximity.

        Uses exponential decay - closer events have higher confidence.
        """
        seconds = delta.total_seconds()
        max_seconds = self.max_time_delta.total_seconds()

        if seconds <= 0:
            return 0.0

        # Exponential decay
        confidence = 1.0 * (1 - (seconds / max_seconds) ** self.decay_factor)
        return max(0.0, min(1.0, confidence))

    def _is_meaningful_transition(
        self,
        from_type: NodeType,
        to_type: NodeType,
    ) -> bool:
        """Check if a transition between node types is meaningful."""
        meaningful_pairs = {
            (NodeType.AGENT_INPUT, NodeType.AGENT_OUTPUT),
            (NodeType.AGENT_INPUT, NodeType.TOOL_CALL),
            (NodeType.TOOL_CALL, NodeType.TOOL_RESULT),
            (NodeType.TOOL_RESULT, NodeType.AGENT_OUTPUT),
            (NodeType.STATE_READ, NodeType.AGENT_OUTPUT),
            (NodeType.AGENT_OUTPUT, NodeType.STATE_WRITE),
            (NodeType.DECISION, NodeType.AGENT_OUTPUT),
            (NodeType.DECISION, NodeType.TOOL_CALL),
        }
        return (from_type, to_type) in meaningful_pairs

    def detect_anomalies(
        self,
        graph: CausalGraph,
        expected_duration: dict[str, timedelta] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Detect temporal anomalies in the execution.

        Args:
            graph: The causal graph to analyze
            expected_duration: Expected duration for operations by name

        Returns:
            List of anomaly descriptions
        """
        anomalies: list[dict[str, Any]] = []
        nodes = list(graph)
        nodes.sort(key=lambda n: n.timestamp)

        # Detect unusually long gaps
        for i, node in enumerate(nodes[:-1]):
            next_node = nodes[i + 1]
            delta = next_node.timestamp - node.timestamp

            # Flag gaps longer than max_time_delta
            if delta > self.max_time_delta:
                anomalies.append({
                    "type": "long_gap",
                    "from_node": node.id,
                    "to_node": next_node.id,
                    "duration_ms": delta.total_seconds() * 1000,
                    "description": f"Unusual gap of {delta.total_seconds():.2f}s between operations",
                })

        # Detect operations that took longer than expected
        if expected_duration:
            # Find input-output pairs
            for node in nodes:
                if node.type != NodeType.AGENT_INPUT:
                    continue

                operation = node.metadata.get("operation", "")
                if operation not in expected_duration:
                    continue

                # Find the corresponding output
                for other in nodes:
                    if (other.type == NodeType.AGENT_OUTPUT and
                        other.agent_id == node.agent_id and
                        other.timestamp > node.timestamp):

                        delta = other.timestamp - node.timestamp
                        expected = expected_duration[operation]

                        if delta > expected * 2:  # More than 2x expected
                            anomalies.append({
                                "type": "slow_operation",
                                "operation": operation,
                                "input_node": node.id,
                                "output_node": other.id,
                                "duration_ms": delta.total_seconds() * 1000,
                                "expected_ms": expected.total_seconds() * 1000,
                                "description": f"Operation {operation} took {delta.total_seconds():.2f}s (expected {expected.total_seconds():.2f}s)",
                            })
                        break

        return anomalies

    def compute_latency_stats(
        self,
        graph: CausalGraph,
    ) -> dict[str, dict[str, float]]:
        """
        Compute latency statistics for operations.

        Returns:
            Dictionary mapping operation names to latency stats
        """
        stats: dict[str, list[float]] = {}
        nodes = list(graph)

        # Find input-output pairs for each operation
        for node in nodes:
            if node.type != NodeType.AGENT_INPUT:
                continue

            operation = node.metadata.get("operation", "unknown")

            # Find corresponding output
            for other in nodes:
                if (other.type == NodeType.AGENT_OUTPUT and
                    other.agent_id == node.agent_id and
                    other.timestamp > node.timestamp and
                    other.metadata.get("operation") == operation):

                    delta = (other.timestamp - node.timestamp).total_seconds() * 1000

                    if operation not in stats:
                        stats[operation] = []
                    stats[operation].append(delta)
                    break

        # Compute statistics
        result: dict[str, dict[str, float]] = {}
        for operation, latencies in stats.items():
            if not latencies:
                continue

            result[operation] = {
                "count": len(latencies),
                "min_ms": min(latencies),
                "max_ms": max(latencies),
                "avg_ms": sum(latencies) / len(latencies),
                "median_ms": sorted(latencies)[len(latencies) // 2],
            }

        return result
