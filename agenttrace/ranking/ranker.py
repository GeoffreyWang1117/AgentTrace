"""
Node Ranker for root cause identification.

Ranks nodes in the backward trace by likelihood of being the root cause.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import json

from agenttrace.core.graph import CausalGraph
from agenttrace.core.node import Node
from agenttrace.ranking.scorer import CausalScorer, NodeFeatures


@dataclass
class RankingFeatures:
    """Complete features for a ranked result."""
    node: Node
    rank: int
    score: float
    features: NodeFeatures
    explanation: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            'node_id': self.node.id,
            'agent_id': self.node.agent_id,
            'rank': self.rank,
            'score': self.score,
            'features': self.features.to_dict(),
            'explanation': self.explanation
        }


@dataclass
class RankingResult:
    """Result of ranking analysis."""
    error_node_id: str
    ranked_candidates: List[RankingFeatures]
    total_candidates: int
    top_prediction: Optional[str] = None

    def get_rank_of(self, node_id: str) -> int:
        """Get the rank of a specific node (-1 if not found)."""
        for rf in self.ranked_candidates:
            if rf.node.id == node_id:
                return rf.rank
        return -1

    def to_dict(self) -> Dict[str, Any]:
        return {
            'error_node_id': self.error_node_id,
            'total_candidates': self.total_candidates,
            'top_prediction': self.top_prediction,
            'ranked_candidates': [rc.to_dict() for rc in self.ranked_candidates]
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, default=str)


class NodeRanker:
    """
    Ranks nodes in backward trace by likelihood of being root cause.

    Combines multiple scoring signals to prioritize nodes.
    """

    def __init__(
        self,
        scorer: Optional[CausalScorer] = None,
        min_score_threshold: float = 0.1
    ):
        """
        Initialize the ranker.

        Args:
            scorer: Custom scorer (uses default if None)
            min_score_threshold: Minimum score to include in results
        """
        self.scorer = scorer or CausalScorer()
        self.min_score_threshold = min_score_threshold

    def rank_candidates(
        self,
        graph: CausalGraph,
        error_node_id: str,
        max_candidates: int = -1
    ) -> RankingResult:
        """
        Rank all candidate root causes for an error.

        Args:
            graph: The causal graph
            error_node_id: ID of the error manifestation node
            max_candidates: Maximum candidates to return (-1 for all)

        Returns:
            RankingResult with ranked candidates
        """
        # Get backward trace (all potential causes)
        backward_nodes = graph.trace_backward(error_node_id)

        if not backward_nodes:
            return RankingResult(
                error_node_id=error_node_id,
                ranked_candidates=[],
                total_candidates=0
            )

        # Compute depths for all nodes
        all_depths = self._compute_all_depths(graph, error_node_id, backward_nodes)

        # Score each candidate
        scored_candidates: List[Tuple[Node, float, NodeFeatures]] = []

        for node in backward_nodes:
            features = self.scorer.extract_features(
                graph, node, error_node_id, all_depths
            )
            score = self.scorer.score_node(features)
            features.raw_score = score

            if score >= self.min_score_threshold:
                scored_candidates.append((node, score, features))

        # Sort by score (descending)
        scored_candidates.sort(key=lambda x: x[1], reverse=True)

        # Create ranking features
        ranked = []
        for rank, (node, score, features) in enumerate(scored_candidates, 1):
            explanation = self._generate_explanation(features, score)
            rf = RankingFeatures(
                node=node,
                rank=rank,
                score=score,
                features=features,
                explanation=explanation
            )
            ranked.append(rf)

            if max_candidates > 0 and rank >= max_candidates:
                break

        top_prediction = ranked[0].node.id if ranked else None

        return RankingResult(
            error_node_id=error_node_id,
            ranked_candidates=ranked,
            total_candidates=len(backward_nodes),
            top_prediction=top_prediction
        )

    def _compute_all_depths(
        self,
        graph: CausalGraph,
        error_node_id: str,
        backward_nodes: List[Node]
    ) -> Dict[str, int]:
        """
        Compute depth from error node for all candidates.

        Uses BFS from error node.
        """
        depths = {error_node_id: 0}
        visited = {error_node_id}
        queue = [(error_node_id, 0)]

        while queue:
            current_id, depth = queue.pop(0)

            for pred_id in graph._graph.predecessors(current_id):
                if pred_id not in visited:
                    visited.add(pred_id)
                    depths[pred_id] = depth + 1
                    queue.append((pred_id, depth + 1))

        return depths

    def _generate_explanation(self, features: NodeFeatures, score: float) -> str:
        """Generate human-readable explanation for ranking."""
        reasons = []

        # Position
        if features.relative_position > 0.3 and features.relative_position < 0.7:
            reasons.append("positioned in the middle of the trace")
        elif features.relative_position < 0.3:
            reasons.append("early in the trace")

        # Structure
        if features.out_degree > 1:
            reasons.append(f"affects {features.out_degree} downstream nodes")
        if features.is_divergence_point:
            reasons.append("data flow divergence point")

        # Content
        if features.downstream_errors > 0:
            reasons.append(f"{features.downstream_errors} downstream errors")
        if features.action_type in CausalScorer.HIGH_RISK_ACTIONS:
            reasons.append(f"high-risk action type ({features.action_type})")

        if not reasons:
            reasons.append("baseline candidate")

        return f"Score {score:.3f}: " + "; ".join(reasons)


class ImprovedAgentTrace:
    """
    Improved AgentTrace with node ranking.

    Combines backward tracing with intelligent ranking.
    """

    def __init__(self, ranker: Optional[NodeRanker] = None):
        """Initialize with optional custom ranker."""
        self.ranker = ranker or NodeRanker()

    def find_root_cause(
        self,
        graph: CausalGraph,
        error_node_id: str,
        top_k: int = 5
    ) -> RankingResult:
        """
        Find the most likely root cause(s) for an error.

        Args:
            graph: The causal graph
            error_node_id: ID of the error manifestation node
            top_k: Return top K candidates

        Returns:
            RankingResult with ranked candidates
        """
        return self.ranker.rank_candidates(graph, error_node_id, max_candidates=top_k)

    def evaluate_ranking(
        self,
        ranking_result: RankingResult,
        ground_truth_node_id: str
    ) -> Dict[str, Any]:
        """
        Evaluate ranking against ground truth.

        Args:
            ranking_result: Result from find_root_cause
            ground_truth_node_id: Actual root cause node ID

        Returns:
            Evaluation metrics
        """
        rank = ranking_result.get_rank_of(ground_truth_node_id)

        return {
            'ground_truth_id': ground_truth_node_id,
            'predicted_id': ranking_result.top_prediction,
            'ground_truth_rank': rank,
            'hit_at_1': rank == 1,
            'hit_at_3': 1 <= rank <= 3,
            'hit_at_5': 1 <= rank <= 5,
            'found': rank > 0,
            'mrr': 1 / rank if rank > 0 else 0
        }
