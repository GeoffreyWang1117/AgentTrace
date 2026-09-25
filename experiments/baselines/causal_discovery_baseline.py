"""
Causal Discovery Baseline for AgentTrace Benchmark.

Uses the PC algorithm from causal-learn to discover a causal DAG from
node content features, then finds root causes as parentless ancestors
of the error node.
"""

import numpy as np
from typing import Optional

from agenttrace.core.graph import CausalGraph
from agenttrace.core.node import NodeType


def _check_causal_learn():
    """Check if causal-learn is available."""
    try:
        from causallearn.search.ConstraintBased.PC import pc
        return True
    except ImportError:
        return False


class CausalDiscoveryBaseline:
    """
    Causal discovery baseline using the PC algorithm.

    Represents node content as feature variables, discovers causal DAG,
    then traces from error node to find root causes.

    Implements the standard identify_root_cause interface.
    """

    def __init__(self, alpha: float = 0.05):
        """
        Args:
            alpha: Significance level for PC algorithm independence tests.
        """
        self.alpha = alpha

    def identify_root_cause(
        self,
        graph: CausalGraph,
        error_node_id: str,
    ) -> list[str]:
        """
        Discover causal structure and rank root causes.

        Returns list of node IDs ranked by likelihood of being root cause.
        """
        if not _check_causal_learn():
            print("WARNING: causal-learn not installed, falling back to position")
            from experiments.baselines.position_only_baseline import PositionOnlyBaseline
            return PositionOnlyBaseline().identify_root_cause(graph, error_node_id)

        from causallearn.search.ConstraintBased.PC import pc

        nodes = sorted(graph._nodes.values(), key=lambda n: n.timestamp)
        node_ids = [n.id for n in nodes]
        n = len(nodes)

        if n < 3:
            return node_ids

        # Build feature matrix from node content
        data_matrix = self._build_feature_matrix(nodes)

        # Run PC algorithm
        try:
            cg = pc(data_matrix, alpha=self.alpha, indep_test='fisherz')
            adj_matrix = cg.G.graph  # adjacency matrix
        except Exception:
            # If PC fails, fall back to temporal ordering
            return self._temporal_fallback(nodes, error_node_id)

        # Find error node index
        error_idx = None
        for i, nid in enumerate(node_ids):
            if nid == error_node_id:
                error_idx = i
                break

        if error_idx is None:
            return node_ids

        # Trace backward in discovered DAG to find ancestors
        ancestors = self._find_ancestors(adj_matrix, error_idx)

        # Find parentless ancestors (root causes)
        root_causes = []
        other_ancestors = []
        for idx in ancestors:
            parents = self._find_parents(adj_matrix, idx)
            parent_ancestors = [p for p in parents if p in ancestors]
            if not parent_ancestors:
                root_causes.append(idx)
            else:
                other_ancestors.append(idx)

        # Rank: parentless ancestors first, then other ancestors, then rest
        ranked_indices = root_causes + other_ancestors
        remaining = [i for i in range(n) if i not in ranked_indices and i != error_idx]
        ranked_indices.extend(remaining)

        return [node_ids[i] for i in ranked_indices]

    def _build_feature_matrix(self, nodes: list) -> np.ndarray:
        """Build numeric feature matrix from nodes for PC algorithm."""
        features = []
        for node in nodes:
            # Extract numeric features
            content_len = len(str(node.data))
            type_val = list(NodeType).index(node.type) if node.type in NodeType else 0
            has_error = 1.0 if any(
                kw in str(node.data).lower()
                for kw in ['error', 'fail', 'wrong', 'invalid']
            ) else 0.0

            # Position
            features.append([
                type_val,
                content_len / 1000.0,
                has_error,
                hash(node.agent_id) % 100 / 100.0,
            ])

        return np.array(features, dtype=float)

    def _find_ancestors(self, adj_matrix, node_idx: int) -> list[int]:
        """Find all ancestors of a node in the discovered DAG."""
        n = adj_matrix.shape[0]
        visited = set()
        queue = [node_idx]

        while queue:
            current = queue.pop(0)
            parents = self._find_parents(adj_matrix, current)
            for p in parents:
                if p not in visited:
                    visited.add(p)
                    queue.append(p)

        return list(visited)

    def _find_parents(self, adj_matrix, node_idx: int) -> list[int]:
        """Find parent nodes (direct causes) in adjacency matrix."""
        n = adj_matrix.shape[0]
        parents = []
        for i in range(n):
            # Check for directed edge i -> node_idx
            if adj_matrix[i, node_idx] == -1 and adj_matrix[node_idx, i] == 1:
                parents.append(i)
            elif adj_matrix[i, node_idx] == 1 and adj_matrix[node_idx, i] == 1:
                # Undirected edge — treat as potential parent
                parents.append(i)
        return parents

    def _temporal_fallback(self, nodes: list, error_node_id: str) -> list[str]:
        """Fall back to reverse temporal ordering."""
        node_ids = [n.id for n in nodes]
        # Put nodes before error first (reversed), then after
        error_idx = None
        for i, n in enumerate(nodes):
            if n.id == error_node_id:
                error_idx = i
                break

        if error_idx is None:
            return node_ids

        before = node_ids[:error_idx]
        after = node_ids[error_idx + 1:]
        return list(reversed(before)) + after
