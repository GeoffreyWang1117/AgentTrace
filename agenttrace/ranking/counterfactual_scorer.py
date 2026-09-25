"""
Counterfactual Causal Attribution Scorer for root cause identification.

Implements three complementary causal reasoning strategies:

1. **Structural Counterfactual Impact** -- estimates the effect of
   do(node = normal) on downstream error propagation by removing the
   candidate from the graph and measuring the change in error-indicative
   descendants.

2. **Causal Mediation Score** -- quantifies how much a candidate node
   *mediates* the causal effect from upstream sources to the error node,
   measured as the fraction of source-to-error simple paths that pass
   through the candidate.

3. **Halpern-Pearl Responsibility Score** -- an approximation of
   *degree of responsibility* from the HP actual-causality framework.
   A candidate v is an actual cause if there exists a contingency set W
   such that intervening on v (while holding W fixed) would prevent the
   error.  The responsibility is 1 / (1 + |W_min|), where W_min is the
   smallest such contingency.

The three scores are combined with the existing structural score from
``CausalScorer`` via a tuneable mixing coefficient alpha.

References
----------
- Halpern & Pearl, "Causes and Explanations: A Structural-Model
  Approach", *British J. Phil. Sci.*, 2005.
- Pearl, "Direct and Indirect Effects", *UAI*, 2001.
- Chockler & Halpern, "Responsibility and Blame: A Structural-Model
  Approach", *JAIR*, 2004.
"""

from __future__ import annotations

from collections import deque

import networkx as nx

from agenttrace.core.graph import CausalGraph
from agenttrace.core.node import Node
from agenttrace.ranking.scorer import CausalScorer


# ---------------------------------------------------------------------------
# Error-keyword detection (shared with CausalScorer for consistency)
# ---------------------------------------------------------------------------
_ERROR_KEYWORDS = CausalScorer.ERROR_KEYWORDS


def _has_error_signal(node: Node) -> bool:
    """Return True if the node's data contains error-indicative keywords."""
    if node.data is None:
        return False
    text = str(node.data).lower()
    return any(kw in text for kw in _ERROR_KEYWORDS)


# ---------------------------------------------------------------------------
# Helper: BFS depths from the error node (backward along edges)
# ---------------------------------------------------------------------------
def _compute_depths_from_error(graph: CausalGraph, error_node_id: str) -> dict[str, int]:
    """BFS backward from error_node_id, returning {node_id: depth}."""
    depths: dict[str, int] = {error_node_id: 0}
    queue: deque[tuple[str, int]] = deque([(error_node_id, 0)])
    while queue:
        cid, d = queue.popleft()
        for pid in graph._graph.predecessors(cid):
            if pid not in depths:
                depths[pid] = d + 1
                queue.append((pid, d + 1))
    return depths


# ===================================================================
# CounterfactualScorer
# ===================================================================


class CounterfactualScorer:
    """Score root-cause candidates using counterfactual causal reasoning.

    Parameters
    ----------
    alpha : float, default 0.5
        Mixing weight between the counterfactual component and the
        structural component.

        ``score_final(v) = alpha * counterfactual_score(v)
                         + (1 - alpha) * structural_score(v)``

        where *counterfactual_score* is a weighted blend of the three
        causal measures (impact, mediation, responsibility) and
        *structural_score* comes from ``CausalScorer.score_node``.

    cf_weights : dict[str, float] | None
        Weights for the three counterfactual sub-scores.  Keys are
        ``"impact"``, ``"mediation"``, ``"responsibility"``.  Defaults
        to equal weighting ``{k: 1/3 for k in ...}``.

    max_paths : int, default 500
        Upper bound on the number of simple paths enumerated per
        source-error pair (guards against combinatorial explosion on
        dense graphs).

    structural_scorer : CausalScorer | None
        An existing ``CausalScorer`` instance.  If *None* a default
        one is created.
    """

    def __init__(
        self,
        alpha: float = 0.5,
        cf_weights: dict[str, float] | None = None,
        max_paths: int = 500,
        structural_scorer: CausalScorer | None = None,
    ):
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")

        self.alpha = alpha
        self.cf_weights = cf_weights or {
            "impact": 1.0 / 3,
            "mediation": 1.0 / 3,
            "responsibility": 1.0 / 3,
        }
        self.max_paths = max_paths
        self._structural_scorer = structural_scorer or CausalScorer()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score_candidates(
        self,
        graph: CausalGraph,
        error_node_id: str,
    ) -> list[tuple[str, float, dict]]:
        """Rank every ancestor of ``error_node_id`` by causal attribution.

        Parameters
        ----------
        graph : CausalGraph
            The full execution trace graph.
        error_node_id : str
            Node where the error was observed.

        Returns
        -------
        list[tuple[str, float, dict]]
            ``(node_id, combined_score, explanation)`` sorted descending
            by ``combined_score``.  ``explanation`` contains the
            individual sub-scores and any diagnostic information.
        """
        if error_node_id not in graph._nodes:
            raise ValueError(f"Error node {error_node_id} not in graph")

        # Candidates: all ancestors of the error node (excluding itself)
        ancestors = graph.trace_backward(error_node_id)
        if not ancestors:
            return []

        # Pre-compute depths for the structural scorer
        depths = _compute_depths_from_error(graph, error_node_id)

        results: list[tuple[str, float, dict]] = []

        for node in ancestors:
            nid = node.id

            # --- counterfactual sub-scores ---
            impact = self.counterfactual_impact(graph, nid, error_node_id)
            mediation = self.causal_mediation(graph, nid, error_node_id)
            responsibility = self.responsibility_score(graph, nid, error_node_id)

            w = self.cf_weights
            cf_score = w["impact"] * impact + w["mediation"] * mediation + w["responsibility"] * responsibility

            # --- structural sub-score ---
            features = self._structural_scorer.extract_features(graph, node, error_node_id, depths)
            structural = self._structural_scorer.score_node(features)

            # --- combined ---
            combined = self.alpha * cf_score + (1 - self.alpha) * structural

            explanation = {
                "counterfactual_impact": round(impact, 4),
                "causal_mediation": round(mediation, 4),
                "responsibility": round(responsibility, 4),
                "counterfactual_combined": round(cf_score, 4),
                "structural_score": round(structural, 4),
                "alpha": self.alpha,
            }
            results.append((nid, combined, explanation))

        # Sort descending by score
        results.sort(key=lambda t: t[1], reverse=True)
        return results

    # ------------------------------------------------------------------
    # 1. Structural Counterfactual Impact
    # ------------------------------------------------------------------

    def counterfactual_impact(
        self,
        graph: CausalGraph,
        node_id: str,
        error_node_id: str,
    ) -> float:
        """Estimate the impact of ``do(node = normal)`` on error propagation.

        Intuition
        ---------
        If we remove *node_id* from the graph (simulating an intervention
        that sets its output to a benign value), how many downstream
        error-indicative nodes become unreachable from the remaining
        predecessors of *node_id*?

        Formally, let

        * ``D_before``  = set of descendants of *node_id* that carry an
          error signal,
        * ``D_after``   = subset of ``D_before`` still reachable from at
          least one predecessor of *node_id* in the graph with *node_id*
          removed.

        Then::

            impact(v) = |D_before - D_after| / max(|D_before|, 1)

        A score of 1.0 means that *all* downstream error nodes depend
        exclusively on *node_id*; a score of 0.0 means every one of
        them has an alternative causal path.

        Returns
        -------
        float
            Value in [0, 1].
        """
        if node_id not in graph._nodes:
            return 0.0

        # D_before: error-carrying descendants of node_id
        descendants = graph.trace_forward(node_id)
        d_before = {n.id for n in descendants if _has_error_signal(n)}

        if not d_before:
            # No downstream error signals at all -- check if the error
            # node itself is reachable and counts.
            if error_node_id in {n.id for n in descendants}:
                d_before = {error_node_id}
            else:
                return 0.0

        # Build the counterfactual reachability set:
        # From each predecessor of node_id, do a forward BFS on the
        # graph *without* node_id and collect reachable nodes.
        predecessors = list(graph._graph.predecessors(node_id))
        reachable_after: set[str] = set()

        removed = {node_id}
        for pred_id in predecessors:
            # BFS forward from pred_id, skipping node_id
            visited: set[str] = set()
            queue: deque[str] = deque([pred_id])
            while queue:
                cid = queue.popleft()
                if cid in visited or cid in removed:
                    continue
                visited.add(cid)
                for succ in graph._graph.successors(cid):
                    if succ not in visited and succ not in removed:
                        queue.append(succ)
            reachable_after |= visited

        d_after = d_before & reachable_after
        eliminated = len(d_before) - len(d_after)
        return eliminated / max(len(d_before), 1)

    # ------------------------------------------------------------------
    # 2. Causal Mediation Score
    # ------------------------------------------------------------------

    def causal_mediation(
        self,
        graph: CausalGraph,
        node_id: str,
        error_node_id: str,
    ) -> float:
        """Fraction of source-to-error causal paths that pass through *node_id*.

        Intuition
        ---------
        A high mediation score means that most causal explanations for
        the error must "go through" this node.  Formally:

        Let *S* be the set of root-cause source nodes (nodes with no
        predecessors that are ancestors of the error).  For each
        source *s*, enumerate all simple paths ``s -> ... -> error``.
        The mediation score is::

            mediation(v) = (# paths through v) / (# total paths)

        Path enumeration is capped at ``self.max_paths`` per source to
        keep computation tractable.

        Returns
        -------
        float
            Value in [0, 1].
        """
        if node_id not in graph._nodes or error_node_id not in graph._nodes:
            return 0.0

        # Find source nodes (roots of the backward trace from error)
        sources = graph.find_root_causes(error_node_id)
        if not sources:
            return 0.0

        # Use DAG DP path counting instead of enumerating all_simple_paths
        # (polynomial O(V+E) instead of exponential)
        total_paths = 0
        paths_through_node = 0

        for source in sources:
            sid = source.id
            if sid == error_node_id:
                continue

            # Count total paths from source to error via topological DP
            n_total = self._count_paths_dp(graph, sid, error_node_id)
            if n_total == 0:
                continue

            # Count paths from source to error that pass through node_id:
            # = (paths source→node_id) × (paths node_id→error)
            n_to_node = self._count_paths_dp(graph, sid, node_id)
            n_from_node = self._count_paths_dp(graph, node_id, error_node_id)
            n_through = n_to_node * n_from_node

            total_paths += n_total
            paths_through_node += min(n_through, n_total)

        if total_paths == 0:
            return 0.0

        return min(1.0, paths_through_node / total_paths)

    # ------------------------------------------------------------------
    # 3. Halpern-Pearl Responsibility Score
    # ------------------------------------------------------------------

    def responsibility_score(
        self,
        graph: CausalGraph,
        node_id: str,
        error_node_id: str,
    ) -> float:
        """Halpern-Pearl style *degree of responsibility*.

        Background
        ----------
        Under the HP framework, a variable *v* is an **actual cause**
        of outcome *phi* (here, the error at ``error_node_id``) if:

        (AC1) Both *v* and *phi* are true (the node and the error
              actually occurred).
        (AC2) There exists a *contingency* -- a set of variables *W*
              whose values we hold fixed at their current values --
              such that changing *v* (``do(v = normal)``) would make
              *phi* false.
        (AC3) *v* is minimal (no proper subset of the conjunction
              satisfies AC1-AC2).

        The **degree of responsibility** is ``1 / (1 + |W_min|)`` where
        ``|W_min|`` is the size of the *smallest* contingency set *W*
        satisfying AC2.

        Approximation
        -------------
        Exact computation is NP-hard in general.  We approximate by
        working on the *backward trace* from the error node:

        1. Build the sub-DAG *G_back* of all ancestors of the error
           (plus the error node itself).
        2. For each subset size ``k = 0, 1, 2, ...`` (up to a cap),
           check whether there exists a set *W* of ``k`` *other*
           ancestor nodes such that removing ``{node_id} union W``
           disconnects *every* source from the error node.
        3. If disconnection is achieved with contingency size *k*,
           return ``1 / (1 + k)``.

        For ``k = 0`` this reduces to checking whether ``node_id`` is
        a *cut vertex* between sources and the error.

        The search is bounded by ``max_contingency_size`` (default 3)
        to keep the combinatorics manageable.

        Returns
        -------
        float
            Value in [0, 1].  1.0 means the node alone (no contingency
            needed) disconnects all sources from error.
        """
        max_contingency_size = 3  # keeps worst case O(n^3)

        if node_id not in graph._nodes or error_node_id not in graph._nodes:
            return 0.0

        # Must be an ancestor
        ancestor_ids = {n.id for n in graph.trace_backward(error_node_id)}
        if node_id not in ancestor_ids:
            return 0.0

        # Source nodes (roots in the backward trace)
        sources = graph.find_root_causes(error_node_id)
        source_ids = {s.id for s in sources}
        if not source_ids:
            return 0.0

        # Build the sub-DAG of ancestors + error node
        subgraph_nodes = ancestor_ids | {error_node_id}
        sub_g: nx.DiGraph = graph._graph.subgraph(subgraph_nodes).copy()

        # Helper: check whether any source can reach error_node_id in
        # sub_g after removing a set of nodes.
        def _sources_reach_error(removed: set[str]) -> bool:
            g_view = sub_g.copy()
            g_view.remove_nodes_from(removed)
            for sid in source_ids:
                if sid in removed:
                    continue
                if sid in g_view and nx.has_path(g_view, sid, error_node_id):
                    return True
            return False

        # Other candidate nodes for the contingency set
        other_ancestors = sorted(ancestor_ids - {node_id})

        for k in range(0, max_contingency_size + 1):
            if k == 0:
                # Check if removing node_id alone disconnects sources
                if not _sources_reach_error({node_id}):
                    return 1.0  # responsibility = 1/(1+0)
            else:
                # Try all combinations of size k from other_ancestors
                for combo in _combinations(other_ancestors, k):
                    removed = {node_id} | set(combo)
                    if not _sources_reach_error(removed):
                        return 1.0 / (1.0 + k)

        # Could not disconnect within the budget -- assign a small
        # baseline proportional to how "central" the node is on
        # source-to-error paths (falls back to a mild mediation signal).
        return 0.0

    # ------------------------------------------------------------------
    # Utility: DP path counting on DAG — O(V+E)
    # ------------------------------------------------------------------

    @staticmethod
    def _count_paths_dp(
        graph: CausalGraph,
        source_id: str,
        target_id: str,
    ) -> int:
        """Count the number of directed paths from source to target in the DAG.

        Uses topological DP in O(V+E) instead of exponential path enumeration.
        Returns 0 if no path exists or if source == target (returns 1 in that case).
        """
        if source_id == target_id:
            return 1
        if source_id not in graph._graph or target_id not in graph._graph:
            return 0

        # BFS to find reachable subgraph from source
        reachable = set()
        queue = deque([source_id])
        while queue:
            nid = queue.popleft()
            if nid in reachable:
                continue
            reachable.add(nid)
            for succ in graph._graph.successors(nid):
                if succ not in reachable:
                    queue.append(succ)

        if target_id not in reachable:
            return 0

        # Topological sort of reachable subgraph
        sub = graph._graph.subgraph(reachable)
        try:
            topo_order = list(nx.topological_sort(sub))
        except nx.NetworkXUnfeasible:
            return 0  # cycle detected

        # DP: count[v] = number of paths from source to v
        count = {nid: 0 for nid in topo_order}
        count[source_id] = 1

        for nid in topo_order:
            if count[nid] == 0:
                continue
            for succ in sub.successors(nid):
                count[succ] += count[nid]

        return count.get(target_id, 0)


# ---------------------------------------------------------------------------
# Utility: lightweight combinations generator (avoids itertools import
# for very small k, but we use itertools when available)
# ---------------------------------------------------------------------------


def _combinations(items: list[str], k: int):
    """Yield all k-element combinations of *items*."""
    from itertools import combinations as _itertools_combinations

    yield from _itertools_combinations(items, k)
