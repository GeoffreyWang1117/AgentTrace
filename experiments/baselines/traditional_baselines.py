"""
Traditional Causal Tracing Baselines.

Implements classic approaches for root cause identification:
1. PageRank-based scoring
2. First divergence point
3. Anomaly detection (content-based)
4. Random walk with restart
5. Betweenness centrality
"""

import json
import random
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import networkx as nx

from agenttrace.core.graph import CausalGraph
from agenttrace.core.node import Node


@dataclass
class BaselineResult:
    """Result from a baseline method."""
    scenario_id: str
    method: str
    predicted_node_id: Optional[str]
    predicted_step: Optional[int]
    confidence: float
    all_scores: Dict[str, float]


class PageRankBaseline:
    """
    Use PageRank to identify important nodes.

    The intuition is that root causes are "important" nodes that
    many other nodes depend on.
    """

    def __init__(self, damping: float = 0.85, reverse: bool = True):
        """
        Args:
            damping: PageRank damping factor
            reverse: If True, use reversed graph (find sources of influence)
        """
        self.damping = damping
        self.reverse = reverse

    def identify_root_cause(
        self,
        graph: CausalGraph,
        error_node_id: str
    ) -> BaselineResult:
        """Find root cause using PageRank."""
        g = graph._graph

        if self.reverse:
            g = g.reverse()

        # Run PageRank
        try:
            scores = nx.pagerank(g, alpha=self.damping)
        except Exception:
            scores = {n: 1.0 / len(g) for n in g.nodes()}

        # Get backward nodes from error
        backward_nodes = graph.trace_backward(error_node_id)
        backward_ids = {n.id for n in backward_nodes}

        # Filter to backward trace and sort by score
        filtered_scores = {
            nid: score
            for nid, score in scores.items()
            if nid in backward_ids
        }

        if not filtered_scores:
            return BaselineResult(
                scenario_id="",
                method="PageRank",
                predicted_node_id=None,
                predicted_step=None,
                confidence=0.0,
                all_scores={}
            )

        # Get top scoring node
        top_node_id = max(filtered_scores, key=filtered_scores.get)
        top_node = graph.get_node(top_node_id)
        step = top_node.data.get('step') if isinstance(top_node.data, dict) else None

        return BaselineResult(
            scenario_id="",
            method="PageRank",
            predicted_node_id=top_node_id,
            predicted_step=step,
            confidence=filtered_scores[top_node_id],
            all_scores=filtered_scores
        )


class FirstDivergenceBaseline:
    """
    Find the first point where execution diverges from expected.

    Looks for nodes where:
    1. Output goes to multiple downstream nodes
    2. Content suggests a decision/branch point
    """

    def identify_root_cause(
        self,
        graph: CausalGraph,
        error_node_id: str
    ) -> BaselineResult:
        """Find first divergence point."""
        backward_nodes = graph.trace_backward(error_node_id)

        # Sort by step (earliest first)
        backward_sorted = sorted(
            backward_nodes,
            key=lambda n: n.data.get('step', 999) if isinstance(n.data, dict) else 999
        )

        # Find first node with multiple outgoing edges
        for node in backward_sorted:
            out_degree = len(list(graph._graph.successors(node.id)))
            if out_degree > 1:
                step = node.data.get('step') if isinstance(node.data, dict) else None
                return BaselineResult(
                    scenario_id="",
                    method="FirstDivergence",
                    predicted_node_id=node.id,
                    predicted_step=step,
                    confidence=0.8,
                    all_scores={}
                )

        # Fallback: return earliest node
        if backward_sorted:
            node = backward_sorted[0]
            step = node.data.get('step') if isinstance(node.data, dict) else None
            return BaselineResult(
                scenario_id="",
                method="FirstDivergence",
                predicted_node_id=node.id,
                predicted_step=step,
                confidence=0.5,
                all_scores={}
            )

        return BaselineResult(
            scenario_id="",
            method="FirstDivergence",
            predicted_node_id=None,
            predicted_step=None,
            confidence=0.0,
            all_scores={}
        )


class AnomalyScoreBaseline:
    """
    Score nodes by how "anomalous" their content is.

    Looks for unusual patterns in:
    1. Content keywords
    2. Data format/structure
    3. Agent behavior patterns
    """

    ANOMALY_KEYWORDS = {
        'unexpected', 'unusual', 'strange', 'weird', 'different',
        'changed', 'modified', 'altered', 'corrupted', 'invalid',
        'wrong', 'incorrect', 'mismatch', 'conflict', 'violation'
    }

    def identify_root_cause(
        self,
        graph: CausalGraph,
        error_node_id: str
    ) -> BaselineResult:
        """Find most anomalous node."""
        backward_nodes = graph.trace_backward(error_node_id)

        scores = {}
        for node in backward_nodes:
            score = self._compute_anomaly_score(node, graph)
            scores[node.id] = score

        if not scores:
            return BaselineResult(
                scenario_id="",
                method="AnomalyScore",
                predicted_node_id=None,
                predicted_step=None,
                confidence=0.0,
                all_scores={}
            )

        top_node_id = max(scores, key=scores.get)
        top_node = graph.get_node(top_node_id)
        step = top_node.data.get('step') if isinstance(top_node.data, dict) else None

        return BaselineResult(
            scenario_id="",
            method="AnomalyScore",
            predicted_node_id=top_node_id,
            predicted_step=step,
            confidence=scores[top_node_id],
            all_scores=scores
        )

    def _compute_anomaly_score(self, node: Node, graph: CausalGraph) -> float:
        """Compute anomaly score for a node."""
        score = 0.0
        data_str = str(node.data).lower()

        # Keyword presence
        for kw in self.ANOMALY_KEYWORDS:
            if kw in data_str:
                score += 0.2

        # Structural anomaly: unusual in/out degree
        in_degree = len(list(graph._graph.predecessors(node.id)))
        out_degree = len(list(graph._graph.successors(node.id)))

        # High out degree is unusual
        if out_degree > 2:
            score += 0.2

        # No predecessors but not first node (orphan)
        if in_degree == 0:
            all_timestamps = [n.timestamp for n in graph._nodes.values()]
            if node.timestamp != min(all_timestamps):
                score += 0.3

        return min(1.0, score)


class BetweennessBaseline:
    """
    Use betweenness centrality to identify critical nodes.

    Nodes with high betweenness are on many paths between other nodes,
    making them critical for information flow.
    """

    def identify_root_cause(
        self,
        graph: CausalGraph,
        error_node_id: str
    ) -> BaselineResult:
        """Find node with highest betweenness centrality."""
        try:
            centrality = nx.betweenness_centrality(graph._graph)
        except Exception:
            centrality = {}

        # Filter to backward trace
        backward_nodes = graph.trace_backward(error_node_id)
        backward_ids = {n.id for n in backward_nodes}

        filtered = {
            nid: score
            for nid, score in centrality.items()
            if nid in backward_ids
        }

        if not filtered:
            return BaselineResult(
                scenario_id="",
                method="Betweenness",
                predicted_node_id=None,
                predicted_step=None,
                confidence=0.0,
                all_scores={}
            )

        top_node_id = max(filtered, key=filtered.get)
        top_node = graph.get_node(top_node_id)
        step = top_node.data.get('step') if isinstance(top_node.data, dict) else None

        return BaselineResult(
            scenario_id="",
            method="Betweenness",
            predicted_node_id=top_node_id,
            predicted_step=step,
            confidence=filtered[top_node_id],
            all_scores=filtered
        )


class RandomWalkBaseline:
    """
    Random walk with restart from error node.

    Simulates tracing backward with some randomness,
    counting how often we visit each node.
    """

    def __init__(self, num_walks: int = 100, walk_length: int = 10, restart_prob: float = 0.15):
        self.num_walks = num_walks
        self.walk_length = walk_length
        self.restart_prob = restart_prob

    def identify_root_cause(
        self,
        graph: CausalGraph,
        error_node_id: str
    ) -> BaselineResult:
        """Find root cause via random walk."""
        visit_counts = defaultdict(int)

        for _ in range(self.num_walks):
            current = error_node_id

            for _ in range(self.walk_length):
                # Random restart
                if random.random() < self.restart_prob:
                    current = error_node_id
                    continue

                # Get predecessors
                predecessors = list(graph._graph.predecessors(current))
                if not predecessors:
                    break

                # Random walk to predecessor
                current = random.choice(predecessors)
                visit_counts[current] += 1

        if not visit_counts:
            return BaselineResult(
                scenario_id="",
                method="RandomWalk",
                predicted_node_id=None,
                predicted_step=None,
                confidence=0.0,
                all_scores={}
            )

        # Normalize scores
        total_visits = sum(visit_counts.values())
        scores = {nid: count / total_visits for nid, count in visit_counts.items()}

        top_node_id = max(scores, key=scores.get)
        top_node = graph.get_node(top_node_id)
        step = top_node.data.get('step') if isinstance(top_node.data, dict) else None

        return BaselineResult(
            scenario_id="",
            method="RandomWalk",
            predicted_node_id=top_node_id,
            predicted_step=step,
            confidence=scores[top_node_id],
            all_scores=scores
        )


class EarliestNodeBaseline:
    """
    Simple baseline: pick the earliest node in the backward trace.

    Based on the intuition that root causes are early in the execution.
    """

    def identify_root_cause(
        self,
        graph: CausalGraph,
        error_node_id: str
    ) -> BaselineResult:
        """Find earliest node."""
        backward_nodes = graph.trace_backward(error_node_id)

        if not backward_nodes:
            return BaselineResult(
                scenario_id="",
                method="EarliestNode",
                predicted_node_id=None,
                predicted_step=None,
                confidence=0.0,
                all_scores={}
            )

        # Sort by step
        sorted_nodes = sorted(
            backward_nodes,
            key=lambda n: n.data.get('step', 999) if isinstance(n.data, dict) else 999
        )

        earliest = sorted_nodes[0]
        step = earliest.data.get('step') if isinstance(earliest.data, dict) else None

        return BaselineResult(
            scenario_id="",
            method="EarliestNode",
            predicted_node_id=earliest.id,
            predicted_step=step,
            confidence=0.5,
            all_scores={}
        )


def evaluate_all_baselines(
    traces: Dict[str, Any],
    ground_truth: Dict[str, Any]
) -> Dict[str, Dict[str, Any]]:
    """Evaluate all traditional baselines."""

    baselines = {
        'PageRank': PageRankBaseline(),
        'FirstDivergence': FirstDivergenceBaseline(),
        'AnomalyScore': AnomalyScoreBaseline(),
        'Betweenness': BetweennessBaseline(),
        'RandomWalk': RandomWalkBaseline(),
        'EarliestNode': EarliestNodeBaseline()
    }

    results = {name: {'correct': 0, 'total': 0, 'hit_at_3': 0} for name in baselines}

    for scenario_id, trace in traces.items():
        if scenario_id not in ground_truth:
            continue

        gt = ground_truth[scenario_id]
        gt_root = gt.get('root_cause_node_id')
        gt_step = gt.get('root_cause_step')
        error_node = gt.get('error_node_id')

        if not gt_root or not error_node:
            continue

        graph = CausalGraph.from_json(trace['trace_json'])

        for name, baseline in baselines.items():
            result = baseline.identify_root_cause(graph, error_node)

            # Check by node ID
            is_correct = result.predicted_node_id == gt_root

            # Also check by step (more lenient)
            if not is_correct and result.predicted_step == gt_step:
                is_correct = True

            results[name]['total'] += 1
            if is_correct:
                results[name]['correct'] += 1

            # Check hit@3 (if scores available)
            if result.all_scores:
                sorted_scores = sorted(result.all_scores.items(), key=lambda x: -x[1])
                top3_ids = [x[0] for x in sorted_scores[:3]]
                if gt_root in top3_ids:
                    results[name]['hit_at_3'] += 1
            elif is_correct:
                results[name]['hit_at_3'] += 1

    # Calculate rates
    for name in results:
        total = results[name]['total']
        if total > 0:
            results[name]['hit_at_1_rate'] = results[name]['correct'] / total
            results[name]['hit_at_3_rate'] = results[name]['hit_at_3'] / total
        else:
            results[name]['hit_at_1_rate'] = 0
            results[name]['hit_at_3_rate'] = 0

    return results


def main():
    """Run traditional baseline evaluation."""
    print("="*70)
    print("TRADITIONAL BASELINE EVALUATION")
    print("="*70)

    # Load blind benchmark
    blind_dir = Path("data/blind_benchmark")
    traces_dir = blind_dir / "traces"
    gt_file = blind_dir / "ground_truth" / "all_ground_truth.json"

    with open(gt_file) as f:
        ground_truth = json.load(f)

    traces = {}
    for trace_file in traces_dir.glob("*_trace.json"):
        with open(trace_file) as f:
            trace = json.load(f)
            traces[trace['scenario_id']] = trace

    print(f"\nLoaded {len(traces)} traces")

    # Evaluate
    results = evaluate_all_baselines(traces, ground_truth)

    # Print results
    print("\n" + "-"*60)
    print("{:<20} {:>12} {:>12} {:>12}".format(
        "Method", "Hit@1", "Hit@3", "Total"
    ))
    print("-"*60)

    for name, res in sorted(results.items(), key=lambda x: -x[1]['hit_at_1_rate']):
        print("{:<20} {:>12.1%} {:>12.1%} {:>12}".format(
            name,
            res['hit_at_1_rate'],
            res['hit_at_3_rate'],
            res['total']
        ))

    # Save results
    output_dir = Path("data/results")
    with open(output_dir / "traditional_baselines_results.json", 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_dir / 'traditional_baselines_results.json'}")


if __name__ == "__main__":
    main()
