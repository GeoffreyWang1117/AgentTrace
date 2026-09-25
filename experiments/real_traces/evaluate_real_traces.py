"""
Evaluate AgentTrace on Real Traces.

Tests the ranking algorithm on traces from real multi-agent workflows.
"""

import json
from pathlib import Path
from typing import Dict, Any, List
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from agenttrace.core.graph import CausalGraph
from agenttrace.ranking.ranker import NodeRanker, ImprovedAgentTrace


def load_real_traces():
    """Load real traces and ground truth."""
    traces_dir = Path("data/real_traces")

    # Load ground truth
    gt_file = traces_dir / "ground_truth.json"
    with open(gt_file) as f:
        ground_truth = json.load(f)

    # Load traces
    traces = {}
    for trace_file in traces_dir.glob("*_real_trace.json"):
        with open(trace_file) as f:
            trace = json.load(f)
            traces[trace['scenario_id']] = trace

    return traces, ground_truth


def evaluate_on_real_traces(
    traces: Dict[str, Any],
    ground_truth: Dict[str, Any]
) -> Dict[str, Any]:
    """Evaluate AgentTrace on real traces."""

    improved = ImprovedAgentTrace()

    results = {
        'total': 0,
        'hit_at_1': 0,
        'hit_at_3': 0,
        'found': 0,
        'by_workflow': {},
        'details': []
    }

    for scenario_id, trace in traces.items():
        if scenario_id not in ground_truth:
            continue

        gt = ground_truth[scenario_id]
        gt_root = gt.get('root_cause_node_id')
        error_node = gt.get('error_node_id')
        workflow = gt.get('workflow', 'unknown')

        if not gt_root or not error_node:
            continue

        # Initialize workflow stats
        if workflow not in results['by_workflow']:
            results['by_workflow'][workflow] = {
                'total': 0, 'hit_at_1': 0, 'hit_at_3': 0
            }

        # Load graph and run ranking
        graph = CausalGraph.from_json(trace['trace_json'])
        ranking_result = improved.find_root_cause(graph, error_node, top_k=5)

        rank = ranking_result.get_rank_of(gt_root)
        predicted = ranking_result.top_prediction

        # Update stats
        results['total'] += 1
        results['by_workflow'][workflow]['total'] += 1

        if rank == 1:
            results['hit_at_1'] += 1
            results['by_workflow'][workflow]['hit_at_1'] += 1
        if 1 <= rank <= 3:
            results['hit_at_3'] += 1
            results['by_workflow'][workflow]['hit_at_3'] += 1
        if rank > 0:
            results['found'] += 1

        # Store details
        results['details'].append({
            'scenario_id': scenario_id,
            'workflow': workflow,
            'bug_type': gt.get('bug_type'),
            'ground_truth_node': gt_root,
            'predicted_node': predicted,
            'rank': rank,
            'correct': rank == 1
        })

    # Calculate rates
    if results['total'] > 0:
        results['hit_at_1_rate'] = results['hit_at_1'] / results['total']
        results['hit_at_3_rate'] = results['hit_at_3'] / results['total']
        results['recall'] = results['found'] / results['total']

    for workflow in results['by_workflow']:
        wf = results['by_workflow'][workflow]
        if wf['total'] > 0:
            wf['hit_at_1_rate'] = wf['hit_at_1'] / wf['total']
            wf['hit_at_3_rate'] = wf['hit_at_3'] / wf['total']

    return results


def main():
    """Run evaluation on real traces."""
    print("="*70)
    print("EVALUATING AGENTTRACE ON REAL TRACES")
    print("="*70)

    # Load data
    print("\nLoading real traces...")
    traces, ground_truth = load_real_traces()
    print(f"Loaded {len(traces)} traces, {len(ground_truth)} ground truth entries")

    # Evaluate
    print("\nRunning evaluation...")
    results = evaluate_on_real_traces(traces, ground_truth)

    # Print results
    print("\n" + "-"*60)
    print("OVERALL RESULTS")
    print("-"*60)
    print(f"Total traces: {results['total']}")
    print(f"Hit@1: {results.get('hit_at_1_rate', 0):.1%} ({results['hit_at_1']}/{results['total']})")
    print(f"Hit@3: {results.get('hit_at_3_rate', 0):.1%} ({results['hit_at_3']}/{results['total']})")
    print(f"Recall: {results.get('recall', 0):.1%}")

    print("\n" + "-"*60)
    print("BY WORKFLOW")
    print("-"*60)
    for workflow, stats in results['by_workflow'].items():
        print(f"\n{workflow.upper()}:")
        print(f"  Total: {stats['total']}")
        print(f"  Hit@1: {stats.get('hit_at_1_rate', 0):.1%}")
        print(f"  Hit@3: {stats.get('hit_at_3_rate', 0):.1%}")

    print("\n" + "-"*60)
    print("DETAILS")
    print("-"*60)
    for detail in results['details']:
        status = "✓" if detail['correct'] else "✗"
        print(f"{status} {detail['scenario_id']}: rank={detail['rank']}, bug={detail['bug_type']}")

    # Save results
    output_dir = Path("data/results")
    with open(output_dir / "real_traces_evaluation.json", 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_dir / 'real_traces_evaluation.json'}")

    print("\n" + "="*70)
    print("EVALUATION COMPLETE")
    print("="*70)

    return results


if __name__ == "__main__":
    main()
