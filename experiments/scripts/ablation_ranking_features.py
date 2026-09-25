"""
Ablation Study on Ranking Features.

Analyzes the contribution of different feature groups:
1. Position features
2. Structure features
3. Content features
4. Flow features
5. Confidence features
"""

import json
from pathlib import Path
from typing import Dict, Any, List
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from agenttrace.core.graph import CausalGraph
from agenttrace.ranking.scorer import CausalScorer
from agenttrace.ranking.ranker import NodeRanker, ImprovedAgentTrace


def load_blind_data():
    """Load blind benchmark data."""
    blind_dir = Path("data/blind_benchmark")

    with open(blind_dir / "ground_truth" / "all_ground_truth.json") as f:
        ground_truth = json.load(f)

    traces = {}
    for f in (blind_dir / "traces").glob("*_trace.json"):
        with open(f) as file:
            trace = json.load(file)
            traces[trace['scenario_id']] = trace

    return traces, ground_truth


def evaluate_with_weights(
    traces: Dict[str, Any],
    ground_truth: Dict[str, Any],
    weights: Dict[str, float]
) -> Dict[str, float]:
    """Evaluate ranking with specific feature weights."""
    scorer = CausalScorer(weights=weights)
    ranker = NodeRanker(scorer=scorer)
    improved = ImprovedAgentTrace(ranker=ranker)

    hit_at_1 = 0
    hit_at_3 = 0
    total = 0
    mrr_sum = 0

    for scenario_id, trace in traces.items():
        if scenario_id not in ground_truth:
            continue

        gt = ground_truth[scenario_id]
        gt_root = gt.get('root_cause_node_id')
        error_node = gt.get('error_node_id')

        if not gt_root or not error_node:
            continue

        graph = CausalGraph.from_json(trace['trace_json'])
        result = improved.find_root_cause(graph, error_node, top_k=10)

        rank = result.get_rank_of(gt_root)
        total += 1

        if rank == 1:
            hit_at_1 += 1
        if 1 <= rank <= 3:
            hit_at_3 += 1
        if rank > 0:
            mrr_sum += 1 / rank

    return {
        'hit_at_1': hit_at_1 / total if total > 0 else 0,
        'hit_at_3': hit_at_3 / total if total > 0 else 0,
        'mrr': mrr_sum / total if total > 0 else 0,
        'total': total
    }


def run_ablation():
    """Run ablation study on feature groups."""
    print("="*70)
    print("ABLATION STUDY: RANKING FEATURES")
    print("="*70)

    traces, ground_truth = load_blind_data()
    print(f"\nLoaded {len(traces)} traces")

    # Default weights
    default_weights = {
        'position': 0.25,
        'structure': 0.20,
        'content': 0.25,
        'flow': 0.20,
        'confidence': 0.10
    }

    results = {}

    # Full model
    print("\n1. Full Model (all features)")
    full_result = evaluate_with_weights(traces, ground_truth, default_weights)
    results['Full Model'] = full_result
    print(f"   Hit@1: {full_result['hit_at_1']:.1%}, Hit@3: {full_result['hit_at_3']:.1%}")

    # Ablation: Remove each feature group
    feature_groups = ['position', 'structure', 'content', 'flow', 'confidence']

    print("\n2. Ablation (remove one feature at a time)")
    for feature in feature_groups:
        ablated_weights = default_weights.copy()
        ablated_weights[feature] = 0

        # Renormalize
        total_weight = sum(ablated_weights.values())
        if total_weight > 0:
            ablated_weights = {k: v / total_weight for k, v in ablated_weights.items()}

        result = evaluate_with_weights(traces, ground_truth, ablated_weights)
        results[f'w/o {feature}'] = result

        drop = full_result['hit_at_1'] - result['hit_at_1']
        print(f"   w/o {feature:12}: Hit@1: {result['hit_at_1']:.1%} (Δ={drop:+.1%})")

    # Single feature only
    print("\n3. Single Feature Only")
    for feature in feature_groups:
        single_weights = {k: 0 for k in default_weights}
        single_weights[feature] = 1.0

        result = evaluate_with_weights(traces, ground_truth, single_weights)
        results[f'only {feature}'] = result

        print(f"   only {feature:12}: Hit@1: {result['hit_at_1']:.1%}")

    # Alternative weight configurations
    print("\n4. Alternative Weight Configurations")

    configs = {
        'position-heavy': {'position': 0.5, 'structure': 0.15, 'content': 0.15, 'flow': 0.15, 'confidence': 0.05},
        'structure-heavy': {'position': 0.15, 'structure': 0.5, 'content': 0.15, 'flow': 0.15, 'confidence': 0.05},
        'content-heavy': {'position': 0.15, 'structure': 0.15, 'content': 0.5, 'flow': 0.15, 'confidence': 0.05},
        'balanced': {'position': 0.2, 'structure': 0.2, 'content': 0.2, 'flow': 0.2, 'confidence': 0.2}
    }

    for config_name, weights in configs.items():
        result = evaluate_with_weights(traces, ground_truth, weights)
        results[config_name] = result
        print(f"   {config_name:15}: Hit@1: {result['hit_at_1']:.1%}")

    # Generate summary table
    print("\n" + "="*70)
    print("ABLATION SUMMARY")
    print("="*70)
    print("\n{:<20} {:>10} {:>10} {:>10}".format("Configuration", "Hit@1", "Hit@3", "MRR"))
    print("-"*50)

    for name, res in results.items():
        print("{:<20} {:>10.1%} {:>10.1%} {:>10.3f}".format(
            name, res['hit_at_1'], res['hit_at_3'], res['mrr']
        ))

    # Feature importance (by drop when removed)
    print("\n" + "-"*50)
    print("FEATURE IMPORTANCE (by Hit@1 drop when removed)")
    print("-"*50)

    importance = []
    for feature in feature_groups:
        drop = full_result['hit_at_1'] - results[f'w/o {feature}']['hit_at_1']
        importance.append((feature, drop))

    importance.sort(key=lambda x: -x[1])
    for feature, drop in importance:
        bar = "█" * int(drop * 100)
        print(f"{feature:12}: {drop:+.1%} {bar}")

    # Save results
    output_dir = Path("data/results")
    with open(output_dir / "ablation_ranking_features.json", 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_dir / 'ablation_ranking_features.json'}")


if __name__ == "__main__":
    run_ablation()
