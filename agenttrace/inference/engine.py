"""
Main inference engine that combines multiple analysis techniques.
"""

from __future__ import annotations

from typing import Any

from agenttrace.core.graph import CausalGraph
from agenttrace.core.node import NodeType
from agenttrace.core.edge import Edge, EdgeType
from agenttrace.inference.dataflow import DataFlowAnalyzer
from agenttrace.inference.temporal import TemporalAnalyzer


class InferenceEngine:
    """
    Main inference engine for causal relationship discovery.

    Combines data flow analysis and temporal analysis to
    infer causal relationships between nodes.
    """

    def __init__(
        self,
        dataflow_weight: float = 0.6,
        temporal_weight: float = 0.4,
        min_confidence: float = 0.5,
    ):
        """
        Initialize the inference engine.

        Args:
            dataflow_weight: Weight for data flow confidence
            temporal_weight: Weight for temporal confidence
            min_confidence: Minimum combined confidence to keep an edge
        """
        self.dataflow_analyzer = DataFlowAnalyzer()
        self.temporal_analyzer = TemporalAnalyzer()
        self.dataflow_weight = dataflow_weight
        self.temporal_weight = temporal_weight
        self.min_confidence = min_confidence

    def infer_relationships(
        self,
        graph: CausalGraph,
        apply_to_graph: bool = True,
    ) -> list[Edge]:
        """
        Infer causal relationships in the graph.

        Args:
            graph: The causal graph to analyze
            apply_to_graph: If True, add inferred edges to the graph

        Returns:
            List of inferred edges
        """
        # Get inferences from both analyzers
        dataflow_edges = self.dataflow_analyzer.analyze(graph)
        temporal_edges = self.temporal_analyzer.analyze(graph)

        # Combine edges with same source-target pairs
        combined = self._combine_edges(dataflow_edges, temporal_edges)

        # Filter by minimum confidence
        filtered = [e for e in combined if e.confidence >= self.min_confidence]

        # Apply to graph if requested
        if apply_to_graph:
            for edge in filtered:
                try:
                    graph.add_edge(edge)
                except ValueError:
                    pass  # Edge already exists or invalid

        return filtered

    def _combine_edges(
        self,
        dataflow_edges: list[Edge],
        temporal_edges: list[Edge],
    ) -> list[Edge]:
        """Combine edges from different analyzers."""
        # Index edges by source-target pair
        edge_map: dict[tuple[str, str], dict[str, Any]] = {}

        for edge in dataflow_edges:
            key = (edge.source_id, edge.target_id)
            if key not in edge_map:
                edge_map[key] = {
                    "dataflow_confidence": 0.0,
                    "temporal_confidence": 0.0,
                    "dataflow_edge": None,
                    "temporal_edge": None,
                }
            edge_map[key]["dataflow_confidence"] = edge.confidence
            edge_map[key]["dataflow_edge"] = edge

        for edge in temporal_edges:
            key = (edge.source_id, edge.target_id)
            if key not in edge_map:
                edge_map[key] = {
                    "dataflow_confidence": 0.0,
                    "temporal_confidence": 0.0,
                    "dataflow_edge": None,
                    "temporal_edge": None,
                }
            edge_map[key]["temporal_confidence"] = edge.confidence
            edge_map[key]["temporal_edge"] = edge

        # Create combined edges
        combined: list[Edge] = []
        for key, data in edge_map.items():
            source_id, target_id = key

            # Compute combined confidence
            df_conf = data["dataflow_confidence"]
            t_conf = data["temporal_confidence"]

            # If both analyses agree, boost confidence
            if df_conf > 0 and t_conf > 0:
                combined_conf = (
                    df_conf * self.dataflow_weight + t_conf * self.temporal_weight + 0.1  # Agreement bonus
                )
            else:
                combined_conf = df_conf * self.dataflow_weight + t_conf * self.temporal_weight

            combined_conf = min(combined_conf, 1.0)

            # Determine edge type
            if data["dataflow_edge"]:
                edge_type = data["dataflow_edge"].type
            elif data["temporal_edge"]:
                edge_type = data["temporal_edge"].type
            else:
                edge_type = EdgeType.DATA_FLOW

            # Merge metadata
            metadata = {
                "inference_method": "combined",
                "dataflow_confidence": df_conf,
                "temporal_confidence": t_conf,
                "combined_confidence": combined_conf,
            }

            if data["dataflow_edge"]:
                metadata.update(data["dataflow_edge"].metadata)
            if data["temporal_edge"]:
                metadata.update(data["temporal_edge"].metadata)

            combined.append(
                Edge(
                    source_id=source_id,
                    target_id=target_id,
                    type=edge_type,
                    confidence=combined_conf,
                    metadata=metadata,
                )
            )

        return combined

    def analyze_causality(
        self,
        graph: CausalGraph,
        target_node_id: str,
    ) -> dict[str, Any]:
        """
        Perform comprehensive causality analysis for a target node.

        Args:
            graph: The causal graph
            target_node_id: ID of the node to analyze

        Returns:
            Comprehensive analysis results
        """
        target = graph.get_node(target_node_id)
        if target is None:
            return {"error": "Node not found"}

        # Find data sources
        data_sources = self.dataflow_analyzer.find_data_sources(graph, target_node_id)

        # Get backward trace
        causes = graph.trace_backward(target_node_id)

        # Compute latency stats
        latency_stats = self.temporal_analyzer.compute_latency_stats(graph)

        # Detect anomalies
        anomalies = self.temporal_analyzer.detect_anomalies(graph)

        return {
            "target_node": target.to_dict(),
            "direct_causes": [
                {"node": n.to_dict(), "confidence": c}
                for n, c in data_sources[:5]  # Top 5 sources
            ],
            "all_causes": [n.to_dict() for n in causes],
            "cause_count": len(causes),
            "latency_stats": latency_stats,
            "anomalies": [
                a for a in anomalies if a.get("to_node") == target_node_id or a.get("from_node") == target_node_id
            ],
        }

    def suggest_likely_causes(
        self,
        graph: CausalGraph,
        error_node_id: str,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """
        Suggest likely root causes for an error.

        Args:
            graph: The causal graph
            error_node_id: ID of the error node
            top_k: Number of suggestions to return

        Returns:
            List of likely cause suggestions with explanations
        """
        error_node = graph.get_node(error_node_id)
        if error_node is None:
            return []

        suggestions: list[dict[str, Any]] = []

        # Get all causes
        causes = graph.trace_backward(error_node_id)

        # Score each cause
        for cause in causes:
            score = 0.0
            reasons = []

            # Score based on node type
            if cause.type == NodeType.STATE_WRITE:
                score += 0.3
                reasons.append("State modification before error")
            elif cause.type == NodeType.TOOL_CALL:
                score += 0.25
                reasons.append("External tool invocation")
            elif cause.type == NodeType.DECISION:
                score += 0.35
                reasons.append("Decision point in causal chain")

            # Score based on proximity
            chain = graph.find_path(cause.id, error_node_id)
            if chain:
                proximity = 1.0 / len(chain)
                score += proximity * 0.2
                if len(chain) <= 2:
                    reasons.append(f"Direct cause ({len(chain)} step{'s' if len(chain) > 1 else ''} away)")
                else:
                    reasons.append(f"Indirect cause ({len(chain)} steps away)")

            # Score based on data similarity to error
            data_sources = self.dataflow_analyzer.find_data_sources(graph, error_node_id)
            for source_node, conf in data_sources:
                if source_node.id == cause.id:
                    score += conf * 0.3
                    reasons.append(f"Data similarity score: {conf:.2f}")
                    break

            if score > 0:
                suggestions.append(
                    {
                        "node": cause.to_dict(),
                        "score": min(score, 1.0),
                        "reasons": reasons,
                    }
                )

        # Sort by score and return top k
        suggestions.sort(key=lambda x: x["score"], reverse=True)
        return suggestions[:top_k]
