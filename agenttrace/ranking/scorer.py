"""
Causal Scorer for node ranking.

Computes various features that indicate likelihood of being root cause.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional

from agenttrace.core.graph import CausalGraph
from agenttrace.core.node import Node


@dataclass
class NodeFeatures:
    """Features extracted for a node."""

    node_id: str
    agent_id: str

    # Position features
    depth_from_error: int  # How many hops from error node
    depth_from_root: int  # How many hops from graph root
    relative_position: float  # 0=start, 1=error

    # Structural features
    in_degree: int  # Number of incoming edges
    out_degree: int  # Number of outgoing edges
    betweenness: float  # Betweenness centrality
    descendants_count: int  # How many nodes downstream

    # Content features
    has_error_keyword: bool  # Content mentions error-like words
    action_type: str  # Type of action
    data_size: int  # Complexity of data

    # Flow features
    is_divergence_point: bool  # Where data flow splits
    downstream_errors: int  # Count of error-like nodes downstream

    # Edge features
    avg_edge_confidence: float  # Average confidence of incoming edges

    # Graph-level features
    total_nodes: int = 0  # Total nodes in graph

    # Raw score (before normalization)
    raw_score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "agent_id": self.agent_id,
            "depth_from_error": self.depth_from_error,
            "depth_from_root": self.depth_from_root,
            "relative_position": self.relative_position,
            "in_degree": self.in_degree,
            "out_degree": self.out_degree,
            "betweenness": self.betweenness,
            "descendants_count": self.descendants_count,
            "has_error_keyword": self.has_error_keyword,
            "action_type": self.action_type,
            "data_size": self.data_size,
            "is_divergence_point": self.is_divergence_point,
            "downstream_errors": self.downstream_errors,
            "avg_edge_confidence": self.avg_edge_confidence,
            "total_nodes": self.total_nodes,
            "raw_score": self.raw_score,
        }


class CausalScorer:
    """
    Scores nodes based on likelihood of being root cause.

    Uses a combination of structural, semantic, and flow features.
    """

    # Error-indicating keywords
    ERROR_KEYWORDS = {
        "error",
        "fail",
        "failed",
        "failure",
        "invalid",
        "wrong",
        "incorrect",
        "missing",
        "corrupt",
        "timeout",
        "exception",
        "crash",
        "bug",
        "issue",
        "problem",
        "conflict",
        "rejected",
        "denied",
        "refused",
        "abort",
        "terminated",
        "unexpected",
    }

    # Action types that are more likely to be root causes
    HIGH_RISK_ACTIONS = {
        "submit",
        "send",
        "write",
        "update",
        "modify",
        "create",
        "process",
        "transform",
        "calculate",
        "compute",
        "execute",
    }

    def __init__(self, weights: Optional[Dict[str, float]] = None):
        """
        Initialize scorer with optional custom weights.

        Args:
            weights: Custom weights for different feature types
        """
        # Balanced weights — position no longer encodes early-node bias
        self.weights = weights or {
            "position": 0.30,  # Impact-based (reachability + proximity)
            "structure": 0.30,  # Graph-structural signals
            "content": 0.20,  # Semantic / action-type signals
            "flow": 0.10,  # Data-flow divergence
            "confidence": 0.10,  # Edge confidence
        }

    def extract_features(
        self, graph: CausalGraph, node: Node, error_node_id: str, all_depths: Dict[str, int]
    ) -> NodeFeatures:
        """
        Extract features for a single node.

        Args:
            graph: The causal graph
            node: Node to extract features for
            error_node_id: ID of the error manifestation node
            all_depths: Pre-computed depths from error for all nodes

        Returns:
            NodeFeatures object
        """
        node_id = node.id

        # Position features
        depth_from_error = all_depths.get(node_id, 999)

        # Compute depth from root (nodes with no predecessors)
        root_depth = self._compute_depth_from_root(graph, node_id)

        # Max depth in graph for normalization
        max_depth = max(all_depths.values()) if all_depths else 1
        relative_position = 1 - (depth_from_error / max(max_depth, 1))

        # Structural features
        in_degree = len(list(graph._graph.predecessors(node_id)))
        out_degree = len(list(graph._graph.successors(node_id)))

        # Betweenness centrality (expensive, could be pre-computed)
        try:
            import networkx as nx

            betweenness = nx.betweenness_centrality(graph._graph).get(node_id, 0)
        except Exception:
            betweenness = 0.0

        # Count descendants
        descendants = graph.trace_forward(node_id)
        descendants_count = len(descendants)

        # Content features
        has_error_keyword = self._check_error_keywords(node.data)
        action_type = self._extract_action_type(node.data)
        data_size = len(str(node.data))

        # Check downstream for errors
        downstream_errors = sum(1 for d in descendants if self._check_error_keywords(d.data))

        # Flow features - is this a divergence point?
        is_divergence_point = out_degree > 1

        # Edge features
        incoming_edges = [
            graph._edges.get(e_data.get("id"))
            for _, _, e_data in graph._graph.in_edges(node_id, data=True)
            if e_data.get("id")
        ]
        avg_edge_confidence = sum(e.confidence for e in incoming_edges if e) / max(len(incoming_edges), 1)

        return NodeFeatures(
            node_id=node_id,
            agent_id=node.agent_id,
            depth_from_error=depth_from_error,
            depth_from_root=root_depth,
            relative_position=relative_position,
            in_degree=in_degree,
            out_degree=out_degree,
            betweenness=betweenness,
            descendants_count=descendants_count,
            has_error_keyword=has_error_keyword,
            action_type=action_type,
            data_size=data_size,
            is_divergence_point=is_divergence_point,
            downstream_errors=downstream_errors,
            avg_edge_confidence=avg_edge_confidence,
            total_nodes=graph.node_count,
        )

    def score_node(self, features: NodeFeatures) -> float:
        """
        Compute score for a node based on its features.

        Higher score = more likely to be root cause.

        Args:
            features: Extracted features for the node

        Returns:
            Score between 0 and 1
        """
        scores = {}

        # Position score: Earlier nodes (further from error) score higher
        # But not too early (very first nodes are less likely)
        position_score = self._position_score(features)
        scores["position"] = position_score

        # Structure score: Nodes with high out-degree propagate bugs
        structure_score = self._structure_score(features)
        scores["structure"] = structure_score

        # Content score: Based on action type and downstream errors
        content_score = self._content_score(features)
        scores["content"] = content_score

        # Flow score: Divergence points are interesting
        flow_score = self._flow_score(features)
        scores["flow"] = flow_score

        # Confidence score: Lower confidence edges = more suspicious
        confidence_score = self._confidence_score(features)
        scores["confidence"] = confidence_score

        # Weighted combination
        total_score = sum(self.weights[k] * scores[k] for k in self.weights)

        return total_score

    def _position_score(self, features: NodeFeatures) -> float:
        """
        Impact-based position score using orthogonalized features.

        Optimization O5: reachability and proximity are correlated in
        sequential traces. We use proximity as primary signal and
        reachability residual (deviation from expected) as secondary.
        """
        total = max(features.total_nodes - 1, 1)
        reachability = min(1.0, features.descendants_count / total)
        proximity = 1.0 / (1 + features.depth_from_error)

        # Expected reachability given proximity (linear model):
        # early nodes (high proximity to root) → high reachability
        # Residual = actual - expected captures anomalous influence
        expected_reach = 1.0 - proximity  # rough linear approximation
        reach_residual = max(0.0, reachability - expected_reach)

        return 0.6 * proximity + 0.4 * reach_residual

    def _structure_score(self, features: NodeFeatures) -> float:
        """
        Score based on graph structure.

        High out-degree suggests bug propagation.
        High betweenness suggests critical node.
        """
        # Out-degree contribution
        out_score = min(1.0, features.out_degree / 3)

        # Betweenness contribution (already 0-1)
        betweenness_score = features.betweenness

        # Descendants count (normalized)
        descendant_score = min(1.0, features.descendants_count / 5)

        return out_score * 0.4 + betweenness_score * 0.3 + descendant_score * 0.3

    def _content_score(self, features: NodeFeatures) -> float:
        """
        Score based on content analysis.

        Actions that modify/send data are more likely sources.
        Downstream errors indicate propagation from this node.
        """
        score = 0.0

        # High-risk action type
        if features.action_type in self.HIGH_RISK_ACTIONS:
            score += 0.4

        # Node itself has error keywords (suspicious but might be effect)
        if features.has_error_keyword:
            score += 0.1

        # Downstream errors (strong signal for root cause)
        if features.downstream_errors > 0:
            score += min(0.5, features.downstream_errors * 0.15)

        return min(1.0, score)

    def _flow_score(self, features: NodeFeatures) -> float:
        """
        Score based on data flow patterns.

        Divergence points (where data splits to multiple paths) are
        more likely to be root causes as they affect multiple downstream nodes.
        """
        score = 0.0

        if features.is_divergence_point:
            score += 0.6

        # High descendant count with multiple paths
        if features.descendants_count > 2:
            score += 0.3

        return min(1.0, score)

    def _confidence_score(self, features: NodeFeatures) -> float:
        """
        Score based on edge confidence.

        Lower confidence incoming edges suggest uncertainty in causation,
        which might indicate a transformation/modification happened.
        """
        # Invert: lower confidence = higher suspicion
        if features.avg_edge_confidence > 0:
            return 1 - features.avg_edge_confidence
        return 0.5

    def _compute_depth_from_root(self, graph: CausalGraph, node_id: str) -> int:
        """Compute depth from root nodes."""
        # Find all root nodes (no predecessors)
        visited = {node_id}
        queue = [(node_id, 0)]
        max_depth = 0

        while queue:
            current_id, depth = queue.pop(0)
            predecessors = list(graph._graph.predecessors(current_id))

            if not predecessors:
                max_depth = max(max_depth, depth)
            else:
                for pred_id in predecessors:
                    if pred_id not in visited:
                        visited.add(pred_id)
                        queue.append((pred_id, depth + 1))

        return max_depth

    def _check_error_keywords(self, data: Any) -> bool:
        """Check if data contains error-indicating keywords."""
        if data is None:
            return False

        data_str = str(data).lower()
        return any(kw in data_str for kw in self.ERROR_KEYWORDS)

    def _extract_action_type(self, data: Any) -> str:
        """Extract action type from node data."""
        if isinstance(data, dict):
            action = data.get("action", "")
            if action:
                # Extract base verb from action like "submit_code" -> "submit"
                return action.split("_")[0].lower()
        return "unknown"

    @staticmethod
    def extract_feature_vector(features: NodeFeatures) -> list[float]:
        """
        Extract the 5 sub-scores that correspond to the weight keys:
        [position_raw, structure_raw, content_raw, flow_raw, confidence_raw].

        Useful for training learned weights via logistic regression or LambdaMART.
        """
        scorer = CausalScorer.__new__(CausalScorer)
        scorer.weights = {"position": 0.30, "structure": 0.30, "content": 0.20, "flow": 0.10, "confidence": 0.10}
        return [
            scorer._position_score(features),
            scorer._structure_score(features),
            scorer._content_score(features),
            scorer._flow_score(features),
            scorer._confidence_score(features),
        ]

    @classmethod
    def from_learned_weights(cls, path: str) -> "CausalScorer":
        """
        Load learned weights from a JSON file.

        Supports both flat format {"position": float, ...} and
        wrapped format {"weights": {"position": float, ...}, ...}.
        """
        import json as _json

        with open(path) as f:
            data = _json.load(f)
        weights = data.get("weights", data) if isinstance(data, dict) else data
        return cls(weights=weights)
