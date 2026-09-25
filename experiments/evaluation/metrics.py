"""
Evaluation Metrics for AgentTrace Benchmark.

This module implements various metrics for evaluating causal chain
inference accuracy and debugging effectiveness.
"""

import json
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional
from collections import defaultdict

from agenttrace.core.graph import CausalGraph
from agenttrace.core.node import Node


@dataclass
class CausalChainMetrics:
    """Metrics for causal chain inference evaluation."""
    # Edge-level metrics
    precision: float      # TP / (TP + FP)
    recall: float         # TP / (TP + FN)
    f1: float            # 2 * P * R / (P + R)

    # Chain-level metrics
    chain_accuracy: float  # Percentage of complete chains matched

    # Detailed counts
    true_positives: int
    false_positives: int
    false_negatives: int


@dataclass
class RootCauseMetrics:
    """Metrics for root cause localization evaluation."""
    hit_at_1: float      # Top prediction is correct
    hit_at_3: float      # Correct answer in top 3
    hit_at_5: float      # Correct answer in top 5
    mrr: float           # Mean Reciprocal Rank

    # Detailed results
    total_samples: int
    correct_at_1: int
    correct_at_3: int
    correct_at_5: int


@dataclass
class DebuggingEfficiencyMetrics:
    """Metrics for debugging efficiency evaluation."""
    avg_steps_to_root_cause: float
    median_steps_to_root_cause: float
    accuracy: float  # Percentage that found correct root cause
    avg_nodes_examined: float  # Cognitive load proxy


@dataclass
class EvaluationResult:
    """Complete evaluation results for a method."""
    method_name: str
    causal_chain_metrics: CausalChainMetrics
    root_cause_metrics: RootCauseMetrics
    debugging_efficiency: Optional[DebuggingEfficiencyMetrics]

    # Per-domain breakdown
    domain_breakdown: dict[str, dict]

    # Summary statistics
    total_scenarios: int
    successful_evaluations: int


class CausalChainEvaluator:
    """Evaluates causal chain inference accuracy."""

    @staticmethod
    def compute_edge_metrics(
        predicted_edges: set[tuple[str, str]],
        ground_truth_edges: set[tuple[str, str]]
    ) -> CausalChainMetrics:
        """
        Compute precision, recall, F1 for predicted edges.

        Args:
            predicted_edges: Set of (source_id, target_id) tuples
            ground_truth_edges: Set of (source_id, target_id) tuples

        Returns:
            CausalChainMetrics with computed values
        """
        tp = len(predicted_edges & ground_truth_edges)
        fp = len(predicted_edges - ground_truth_edges)
        fn = len(ground_truth_edges - predicted_edges)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        return CausalChainMetrics(
            precision=precision,
            recall=recall,
            f1=f1,
            chain_accuracy=1.0 if predicted_edges == ground_truth_edges else 0.0,
            true_positives=tp,
            false_positives=fp,
            false_negatives=fn
        )

    @staticmethod
    def evaluate_trace(
        predicted_graph: CausalGraph,
        ground_truth_labels: dict
    ) -> CausalChainMetrics:
        """
        Evaluate predicted causal graph against ground truth.

        Args:
            predicted_graph: The CausalGraph produced by AgentTrace
            ground_truth_labels: Dict with 'causal_edges' list

        Returns:
            CausalChainMetrics
        """
        # Extract predicted edges
        predicted_edges = set()
        for edge in predicted_graph.get_all_edges():
            predicted_edges.add((edge.source_id, edge.target_id))

        # Extract ground truth edges
        gt_edges = set()
        for edge in ground_truth_labels.get("causal_edges", []):
            gt_edges.add((edge["source_node_id"], edge["target_node_id"]))

        return CausalChainEvaluator.compute_edge_metrics(predicted_edges, gt_edges)


class RootCauseEvaluator:
    """Evaluates root cause localization accuracy."""

    @staticmethod
    def compute_metrics(
        predictions: list[tuple[str, list[str]]],  # (scenario_id, ranked_node_ids)
        ground_truths: dict[str, str]  # scenario_id -> correct_node_id
    ) -> RootCauseMetrics:
        """
        Compute Hit@K and MRR metrics.

        Args:
            predictions: List of (scenario_id, ranked list of predicted node IDs)
            ground_truths: Dict mapping scenario_id to correct root cause node ID

        Returns:
            RootCauseMetrics
        """
        correct_at_1 = 0
        correct_at_3 = 0
        correct_at_5 = 0
        reciprocal_ranks = []

        for scenario_id, ranked_predictions in predictions:
            if scenario_id not in ground_truths:
                continue

            correct_answer = ground_truths[scenario_id]

            # Find rank of correct answer
            try:
                rank = ranked_predictions.index(correct_answer) + 1
            except ValueError:
                rank = float('inf')

            # Update counts
            if rank == 1:
                correct_at_1 += 1
            if rank <= 3:
                correct_at_3 += 1
            if rank <= 5:
                correct_at_5 += 1

            # Reciprocal rank
            rr = 1.0 / rank if rank != float('inf') else 0.0
            reciprocal_ranks.append(rr)

        total = len(predictions)
        mrr = sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0.0

        return RootCauseMetrics(
            hit_at_1=correct_at_1 / total if total > 0 else 0.0,
            hit_at_3=correct_at_3 / total if total > 0 else 0.0,
            hit_at_5=correct_at_5 / total if total > 0 else 0.0,
            mrr=mrr,
            total_samples=total,
            correct_at_1=correct_at_1,
            correct_at_3=correct_at_3,
            correct_at_5=correct_at_5
        )


class MetricsAggregator:
    """Aggregates metrics across multiple evaluations."""

    def __init__(self):
        self.chain_metrics: list[CausalChainMetrics] = []
        self.root_cause_metrics: list[RootCauseMetrics] = []
        self.domain_metrics: dict[str, list] = defaultdict(list)

    def add_chain_metrics(
        self,
        metrics: CausalChainMetrics,
        domain: Optional[str] = None
    ):
        """Add chain metrics to aggregator."""
        self.chain_metrics.append(metrics)
        if domain:
            self.domain_metrics[domain].append(("chain", metrics))

    def add_root_cause_metrics(
        self,
        metrics: RootCauseMetrics,
        domain: Optional[str] = None
    ):
        """Add root cause metrics to aggregator."""
        self.root_cause_metrics.append(metrics)
        if domain:
            self.domain_metrics[domain].append(("root_cause", metrics))

    def aggregate_chain_metrics(self) -> CausalChainMetrics:
        """Compute aggregate chain metrics."""
        if not self.chain_metrics:
            return CausalChainMetrics(0, 0, 0, 0, 0, 0, 0)

        n = len(self.chain_metrics)
        return CausalChainMetrics(
            precision=sum(m.precision for m in self.chain_metrics) / n,
            recall=sum(m.recall for m in self.chain_metrics) / n,
            f1=sum(m.f1 for m in self.chain_metrics) / n,
            chain_accuracy=sum(m.chain_accuracy for m in self.chain_metrics) / n,
            true_positives=sum(m.true_positives for m in self.chain_metrics),
            false_positives=sum(m.false_positives for m in self.chain_metrics),
            false_negatives=sum(m.false_negatives for m in self.chain_metrics)
        )

    def aggregate_root_cause_metrics(self) -> RootCauseMetrics:
        """Compute aggregate root cause metrics."""
        if not self.root_cause_metrics:
            return RootCauseMetrics(0, 0, 0, 0, 0, 0, 0, 0)

        n = len(self.root_cause_metrics)
        total_samples = sum(m.total_samples for m in self.root_cause_metrics)

        return RootCauseMetrics(
            hit_at_1=sum(m.hit_at_1 for m in self.root_cause_metrics) / n,
            hit_at_3=sum(m.hit_at_3 for m in self.root_cause_metrics) / n,
            hit_at_5=sum(m.hit_at_5 for m in self.root_cause_metrics) / n,
            mrr=sum(m.mrr for m in self.root_cause_metrics) / n,
            total_samples=total_samples,
            correct_at_1=sum(m.correct_at_1 for m in self.root_cause_metrics),
            correct_at_3=sum(m.correct_at_3 for m in self.root_cause_metrics),
            correct_at_5=sum(m.correct_at_5 for m in self.root_cause_metrics)
        )

    def get_domain_breakdown(self) -> dict[str, dict]:
        """Get metrics broken down by domain."""
        breakdown = {}

        for domain, metrics_list in self.domain_metrics.items():
            chain_metrics = [m for t, m in metrics_list if t == "chain"]
            rc_metrics = [m for t, m in metrics_list if t == "root_cause"]

            domain_summary = {"count": len(metrics_list)}

            if chain_metrics:
                domain_summary["avg_f1"] = sum(m.f1 for m in chain_metrics) / len(chain_metrics)
                domain_summary["avg_precision"] = sum(m.precision for m in chain_metrics) / len(chain_metrics)
                domain_summary["avg_recall"] = sum(m.recall for m in chain_metrics) / len(chain_metrics)

            if rc_metrics:
                domain_summary["avg_hit_at_1"] = sum(m.hit_at_1 for m in rc_metrics) / len(rc_metrics)
                domain_summary["avg_mrr"] = sum(m.mrr for m in rc_metrics) / len(rc_metrics)

            breakdown[domain] = domain_summary

        return breakdown


@dataclass
class TwoStageResult:
    """Results from two-stage evaluation."""
    # Stage 1: Edge inference quality
    edge_metrics: CausalChainMetrics
    # Stage 2: Ranking quality on inferred graph
    ranking_metrics: RootCauseMetrics


@dataclass
class MultiRootCauseMetrics:
    """Metrics for multi-root-cause evaluation."""
    ndcg_at_3: float
    ndcg_at_5: float
    recall_at_3: float
    recall_at_5: float
    total_samples: int


class TwoStageEvaluator:
    """
    Evaluates AgentTrace in two independent stages:
    1. Edge inference quality (inferred edges vs ground truth edges)
    2. Ranking quality (root cause ranking on the inferred graph)
    """

    @staticmethod
    def evaluate(
        predicted_edges: set[tuple[str, str]],
        ground_truth_edges: set[tuple[str, str]],
        predictions: list[tuple[str, list[str]]],
        ground_truths: dict[str, str],
    ) -> TwoStageResult:
        """
        Run both stages of evaluation.

        Args:
            predicted_edges: Inferred edges as (source, target) pairs
            ground_truth_edges: Ground truth edges
            predictions: Ranking predictions per scenario
            ground_truths: Ground truth root cause per scenario
        """
        edge_metrics = CausalChainEvaluator.compute_edge_metrics(
            predicted_edges, ground_truth_edges
        )
        ranking_metrics = RootCauseEvaluator.compute_metrics(
            predictions, ground_truths
        )
        return TwoStageResult(
            edge_metrics=edge_metrics,
            ranking_metrics=ranking_metrics,
        )


def compute_ndcg_at_k(
    ranked_ids: list[str],
    relevant_ids: set[str],
    k: int,
) -> float:
    """
    Compute Normalized Discounted Cumulative Gain at K.

    Args:
        ranked_ids: Ranked list of predicted node IDs
        relevant_ids: Set of ground truth root cause node IDs
        k: Cutoff
    """
    import math

    def dcg(ranks: list[float], k: int) -> float:
        return sum(
            rel / math.log2(i + 2)
            for i, rel in enumerate(ranks[:k])
        )

    # Actual gains
    gains = [1.0 if nid in relevant_ids else 0.0 for nid in ranked_ids[:k]]
    actual_dcg = dcg(gains, k)

    # Ideal gains
    ideal_gains = sorted(gains, reverse=True)
    # But we might have fewer relevant items — ideal is all relevant first
    n_relevant = min(len(relevant_ids), k)
    ideal = [1.0] * n_relevant + [0.0] * (k - n_relevant)
    ideal_dcg = dcg(ideal, k)

    if ideal_dcg == 0:
        return 0.0
    return actual_dcg / ideal_dcg


def compute_recall_at_k(
    ranked_ids: list[str],
    relevant_ids: set[str],
    k: int,
) -> float:
    """Compute Recall@K: fraction of relevant items found in top K."""
    if not relevant_ids:
        return 0.0
    found = sum(1 for nid in ranked_ids[:k] if nid in relevant_ids)
    return found / len(relevant_ids)


class MultiRootCauseEvaluator:
    """Evaluates scenarios with multiple root causes."""

    @staticmethod
    def compute_metrics(
        predictions: list[tuple[str, list[str]]],
        ground_truths: dict[str, set[str]],
    ) -> MultiRootCauseMetrics:
        """
        Compute NDCG@K and Recall@K for multi-root scenarios.

        Args:
            predictions: (scenario_id, ranked_node_ids) pairs
            ground_truths: scenario_id -> set of root cause node IDs
        """
        ndcg_3_scores = []
        ndcg_5_scores = []
        recall_3_scores = []
        recall_5_scores = []

        for scenario_id, ranked in predictions:
            if scenario_id not in ground_truths:
                continue
            relevant = ground_truths[scenario_id]

            ndcg_3_scores.append(compute_ndcg_at_k(ranked, relevant, 3))
            ndcg_5_scores.append(compute_ndcg_at_k(ranked, relevant, 5))
            recall_3_scores.append(compute_recall_at_k(ranked, relevant, 3))
            recall_5_scores.append(compute_recall_at_k(ranked, relevant, 5))

        n = len(ndcg_3_scores)
        return MultiRootCauseMetrics(
            ndcg_at_3=sum(ndcg_3_scores) / n if n else 0.0,
            ndcg_at_5=sum(ndcg_5_scores) / n if n else 0.0,
            recall_at_3=sum(recall_3_scores) / n if n else 0.0,
            recall_at_5=sum(recall_5_scores) / n if n else 0.0,
            total_samples=n,
        )


def generate_latex_table(results: list[EvaluationResult]) -> str:
    """Generate LaTeX table from evaluation results."""
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Comparison of Causal Chain Inference Methods}",
        r"\label{tab:results}",
        r"\begin{tabular}{lccccccc}",
        r"\toprule",
        r"Method & Precision & Recall & F1 & Hit@1 & Hit@3 & MRR \\",
        r"\midrule"
    ]

    for result in results:
        cm = result.causal_chain_metrics
        rm = result.root_cause_metrics
        lines.append(
            f"{result.method_name} & "
            f"{cm.precision:.3f} & {cm.recall:.3f} & {cm.f1:.3f} & "
            f"{rm.hit_at_1:.3f} & {rm.hit_at_3:.3f} & {rm.mrr:.3f} \\\\"
        )

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}"
    ])

    return "\n".join(lines)


def save_results(
    results: list[EvaluationResult],
    output_dir: Path = Path("data/results")
) -> Path:
    """Save evaluation results to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save as JSON
    json_path = output_dir / "evaluation_results.json"
    with open(json_path, 'w') as f:
        json.dump(
            [asdict(r) for r in results],
            f,
            indent=2
        )

    # Save LaTeX table
    latex_path = output_dir / "results_table.tex"
    with open(latex_path, 'w') as f:
        f.write(generate_latex_table(results))

    return json_path
