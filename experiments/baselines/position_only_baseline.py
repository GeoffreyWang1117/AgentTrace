"""
Position-Only Baseline for AgentTrace Benchmark.

Ranks nodes purely by position heuristic: 1 - |relative_position - 0.4| * 1.5
This makes the position bias explicitly measurable as a named baseline.
"""

from agenttrace.core.graph import CausalGraph
from agenttrace.core.node import Node


class PositionOnlyBaseline:
    """
    Baseline that ranks nodes purely by position.

    Implements the standard identify_root_cause interface.
    """

    def __init__(self, optimal_position: float = 0.4):
        self.optimal_position = optimal_position

    def identify_root_cause(
        self,
        graph: CausalGraph,
        error_node_id: str,
    ) -> list[str]:
        """
        Rank nodes by position heuristic.

        Returns list of node IDs ranked by position score (most likely first).
        """
        nodes = sorted(graph, key=lambda n: n.timestamp)
        total = len(nodes)

        scored = []
        for i, node in enumerate(nodes):
            if node.id == error_node_id:
                continue
            rel_pos = i / max(total - 1, 1)
            score = max(0, 1 - abs(rel_pos - self.optimal_position) * 1.5)
            scored.append((node.id, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [nid for nid, _ in scored]
