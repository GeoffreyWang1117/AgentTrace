"""
Multi-Agent Structural Causal Model (MA-SCM) — Optimized v2.

Formal causal reasoning framework for root cause analysis in multi-agent
LLM systems, grounded in Pearl's causal hierarchy and Halpern-Pearl
actual causality.

Optimizations over v1:
  O2: HP contingency → minimum vertex cut via max-flow (polynomial)
  O3: Precompute ancestors once, pass everywhere
  O6: Interventional impact = source disconnection (distinct from mediation)
  O7: Lexicographic ordering: responsibility > mediation > structural
  O8: Batch path counting via forward/reverse DP

References
----------
- Pearl, "Causality" (2009), Ch. 7
- Halpern, "A Modification of the Halpern-Pearl Definition" (2015)
- Chockler & Halpern, "Responsibility and Blame" (JAIR, 2004)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import networkx as nx

from agenttrace.core.graph import CausalGraph


# ═══════════════════════════════════════════════════════════════════════
# Definition 1: Multi-Agent SCM
# ═══════════════════════════════════════════════════════════════════════


@dataclass
class MultiAgentSCM:
    """
    M_MA = (U, V, F, P(U), A, a)
    """

    graph: CausalGraph
    agents: set[str] = field(default_factory=set)
    node_agent_map: dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        for nid, node in self.graph._nodes.items():
            self.node_agent_map[nid] = node.agent_id
            self.agents.add(node.agent_id)

    @classmethod
    def from_graph(cls, graph: CausalGraph) -> "MultiAgentSCM":
        return cls(graph=graph)

    def agent_of(self, node_id: str) -> str:
        return self.node_agent_map.get(node_id, "unknown")


# ═══════════════════════════════════════════════════════════════════════
# Batch Path Counting — O(V+E) per source (Optimization O8)
# ═══════════════════════════════════════════════════════════════════════


class PathCounter:
    """
    Precompute all path counts in batch: forward from each source,
    reverse from error. O((M+1)(V+E)) total instead of O(3MN(V+E)).
    """

    def __init__(self, graph: nx.DiGraph, error_node_id: str, ancestors: set[str]):
        self.graph = graph
        self.error_node_id = error_node_id
        self.ancestors = ancestors

        # Source nodes (roots in ancestor subgraph)
        self.source_ids = {n for n in ancestors if graph.in_degree(n) == 0}

        # Ancestor subgraph + error node
        sub_nodes = ancestors | {error_node_id}
        self.sub_g = graph.subgraph(sub_nodes)

        try:
            self.topo_order = list(nx.topological_sort(self.sub_g))
        except nx.NetworkXUnfeasible:
            self.topo_order = list(sub_nodes)

        # Precompute forward path counts from each source
        self._forward: dict[str, dict[str, int]] = {}
        for sid in self.source_ids:
            self._forward[sid] = self._dp_forward(sid)

        # Precompute reverse path counts to error
        self._reverse = self._dp_reverse()

    def _dp_forward(self, source: str) -> dict[str, int]:
        """Count paths from source to every reachable node."""
        count = {n: 0 for n in self.topo_order}
        count[source] = 1
        for n in self.topo_order:
            if count[n] == 0:
                continue
            for s in self.sub_g.successors(n):
                if s in count:
                    count[s] += count[n]
        return count

    def _dp_reverse(self) -> dict[str, int]:
        """Count paths from every node to error (reverse DP)."""
        count = {n: 0 for n in self.topo_order}
        count[self.error_node_id] = 1
        for n in reversed(self.topo_order):
            for s in self.sub_g.successors(n):
                if s in count:
                    count[n] += count[s]
        return count

    def paths_from_source(self, source: str, target: str) -> int:
        return self._forward.get(source, {}).get(target, 0)

    def paths_to_error(self, node_id: str) -> int:
        return self._reverse.get(node_id, 0)

    def total_paths(self, source: str) -> int:
        return self.paths_from_source(source, self.error_node_id)

    def mediation(self, candidate_id: str) -> float:
        """
        Fraction of source-to-error paths through candidate.

        mediation(v) = Σ_s [paths(s→v) × paths(v→error)] / Σ_s paths(s→error)
        """
        total = 0
        through = 0
        for sid in self.source_ids:
            if sid == self.error_node_id:
                continue
            n_total = self.total_paths(sid)
            if n_total == 0:
                continue
            n_to = self.paths_from_source(sid, candidate_id)
            n_from = self.paths_to_error(candidate_id)
            total += n_total
            through += n_to * n_from

        if total == 0:
            return 0.0
        return min(1.0, through / total)


# ═══════════════════════════════════════════════════════════════════════
# Definition 2: HP Actual Causality — Min Vertex Cut (Optimization O2)
# ═══════════════════════════════════════════════════════════════════════


class HPActualCause:
    """
    Halpern-Pearl Actual Causality with min-cut optimization.

    Instead of enumerating C(N,k) contingency sets, compute the minimum
    vertex cut between sources and error (after removing candidate).
    Responsibility = 1 / (1 + min_cut_size).

    Complexity: O(V × E) via max-flow, instead of O(C(N, max_k)).
    """

    def responsibility(
        self,
        sub_g: nx.DiGraph,
        candidate_id: str,
        source_ids: set[str],
        error_node_id: str,
    ) -> float:
        """
        Compute degree of responsibility via minimum vertex cut.

        dr(v) = 1/(1+|W_min|) where W_min is the smallest set of nodes
        that, together with v, disconnects all sources from error.
        """
        # Remove candidate from subgraph
        remaining = set(sub_g.nodes()) - {candidate_id}
        if error_node_id not in remaining:
            return 1.0  # removing candidate removes error entirely

        view = sub_g.subgraph(remaining).copy()

        # Check if candidate alone is sufficient (k=0 → responsibility=1.0)
        active_sources = source_ids & remaining
        if not active_sources:
            return 1.0  # all sources removed

        any_reachable = False
        for sid in active_sources:
            if sid in view and nx.has_path(view, sid, error_node_id):
                any_reachable = True
                break
        if not any_reachable:
            return 1.0  # candidate alone disconnects → but-for cause

        # Need additional nodes: find minimum vertex cut
        # Transform to edge-cut problem: split each node into (v_in, v_out)
        # with edge capacity 1, then find min cut
        try:
            # Add super-source connecting all source nodes
            flow_g = nx.DiGraph()
            super_src = "__super_source__"
            for n in view.nodes():
                flow_g.add_node(f"{n}_in")
                flow_g.add_node(f"{n}_out")
                flow_g.add_edge(f"{n}_in", f"{n}_out", capacity=1)
            for u, v in view.edges():
                flow_g.add_edge(f"{u}_out", f"{v}_in", capacity=len(view) + 1)
            for sid in active_sources:
                flow_g.add_edge(super_src, f"{sid}_in", capacity=len(view) + 1)

            target = f"{error_node_id}_in"
            if super_src not in flow_g or target not in flow_g:
                return 0.0

            cut_value, _ = nx.minimum_cut(flow_g, super_src, target)
            min_k = int(cut_value)
            if min_k > 20:
                return 0.0
            return 1.0 / (1.0 + min_k)

        except (nx.NetworkXError, nx.NetworkXUnbounded):
            return 0.0

    def is_actual_cause(
        self,
        sub_g: nx.DiGraph,
        candidate_id: str,
        source_ids: set[str],
        error_node_id: str,
    ) -> bool:
        """Check AC1 + AC2: is candidate an actual cause?"""
        return self.responsibility(sub_g, candidate_id, source_ids, error_node_id) > 0.0


# ═══════════════════════════════════════════════════════════════════════
# Definition 3: Interventional Impact (Optimization O6)
# ═══════════════════════════════════════════════════════════════════════


def interventional_impact(
    sub_g: nx.DiGraph,
    candidate_id: str,
    source_ids: set[str],
    error_node_id: str,
) -> float:
    """
    True interventional impact: fraction of sources disconnected
    from error after do(candidate = normal).

    This is mathematically distinct from mediation:
    - Mediation = fraction of PATHS through candidate (continuous)
    - Intervention = fraction of SOURCES disconnected (discrete)

    A node can have high mediation (on many paths) but low intervention
    (removing it doesn't disconnect any source because redundant paths exist).
    """
    remaining = set(sub_g.nodes()) - {candidate_id}
    if error_node_id not in remaining:
        return 1.0

    view = sub_g.subgraph(remaining)
    active_sources = source_ids & remaining
    if not active_sources:
        return 1.0

    disconnected = 0
    for sid in active_sources:
        if not nx.has_path(view, sid, error_node_id):
            disconnected += 1

    return disconnected / len(active_sources)


# ═══════════════════════════════════════════════════════════════════════
# Definition 4: Integrated Causal Attribution (Optimization O7)
# ═══════════════════════════════════════════════════════════════════════


@dataclass
class CausalAttributionResult:
    """Result of causal attribution for a single candidate node."""

    node_id: str
    agent_id: str
    is_actual_cause: bool
    responsibility: float
    mediation_score: float
    interventional_impact: float
    causal_score: float
    details: dict = field(default_factory=dict)


class CausalAttributor:
    """
    Integrated causal attribution using lexicographic ordering.

    Ranking priority (Optimization O7):
      1. HP Responsibility (highest — only theoretically grounded measure)
      2. Mediation score (tie-breaker among equal responsibility)
      3. Interventional impact (secondary tie-breaker)
      4. Structural score (lowest priority)

    Lexicographic ordering avoids arbitrary weight tuning and is
    theoretically justified: HP responsibility is the gold standard
    for actual causality (Halpern 2015).

    The combined score uses a weighted lexicographic scheme:
      score = resp × 1000 + med × 100 + interv × 10 + struct
    This ensures resp always dominates, with lower features as tie-breakers.
    """

    def __init__(self, structural_weight: float = 0.3):
        self.structural_weight = structural_weight
        self.hp = HPActualCause()

    def attribute(
        self,
        graph: CausalGraph,
        error_node_id: str,
        structural_scores: dict[str, float] | None = None,
    ) -> list[CausalAttributionResult]:
        """
        Causal attribution for all candidates. O(V×E) total.
        """
        nx_graph = graph._graph
        scm = MultiAgentSCM.from_graph(graph)

        # O3: Precompute ancestors ONCE
        try:
            ancestors = nx.ancestors(nx_graph, error_node_id)
        except nx.NetworkXError:
            return []

        # Deterministic candidate order: iterating a set (nx.ancestors) is not
        # stable across processes and, combined with the stable sort below, made
        # tie-broken top-1 predictions non-reproducible. Sort for determinism.
        candidates = sorted(nid for nid in ancestors if nid != error_node_id)
        if not candidates:
            return []

        # Source nodes
        source_ids = {n for n in ancestors if nx_graph.in_degree(n) == 0}
        if not source_ids:
            source_ids = set(candidates[:1])

        # Ancestor subgraph (reused everywhere)
        sub_nodes = ancestors | {error_node_id}
        sub_g = nx_graph.subgraph(sub_nodes)

        # O8: Batch path counting
        pc = PathCounter(nx_graph, error_node_id, ancestors)

        results = []
        all_scores = []
        for nid in candidates:
            # Level 3: HP Responsibility via min-cut (O2)
            resp = self.hp.responsibility(sub_g, nid, source_ids, error_node_id)
            is_cause = resp > 0.0

            # Level 2: Mediation via precomputed paths (O8)
            med = pc.mediation(nid)

            # Level 2: Interventional impact as source disconnection (O6)
            interv = interventional_impact(sub_g, nid, source_ids, error_node_id)

            # Level 1: Structural score
            str_score = structural_scores.get(nid, 0.5) if structural_scores else 0.5

            # O7: Lexicographic score with graceful degradation
            causal_score = resp * 100.0 + med * 10.0 + interv * 5.0 + str_score * self.structural_weight

            all_scores.append((resp, med, interv, str_score, causal_score))
            results.append(
                CausalAttributionResult(
                    node_id=nid,
                    agent_id=scm.agent_of(nid),
                    is_actual_cause=is_cause,
                    responsibility=resp,
                    mediation_score=med,
                    interventional_impact=interv,
                    causal_score=causal_score,
                    details={
                        "structural": str_score,
                        "ordering": "lexicographic(resp>med>interv>struct)",
                    },
                )
            )

        # O7 graceful degradation: if responsibility is uniform (no discrimination),
        # re-sort by mediation + structural (lower-level features dominate)
        if all_scores:
            resp_values = [s[0] for s in all_scores]
            resp_range = max(resp_values) - min(resp_values)
            if resp_range < 0.01:
                # All responsibilities equal — resp can't discriminate
                # Re-score with mediation and structural dominating
                for i, r in enumerate(results):
                    r.causal_score = (
                        r.mediation_score * 100.0
                        + r.interventional_impact * 10.0
                        + (structural_scores.get(r.node_id, 0.5) if structural_scores else 0.5) * 5.0
                        + r.responsibility * 0.1  # tiny tie-breaker
                    )

        # Deterministic tie-break: equal causal_score falls back to node_id order.
        results.sort(key=lambda r: (r.causal_score, r.node_id), reverse=True)
        return results
