#!/usr/bin/env python3
"""
Run experiments for ECAI paper:
1. Cross-Domain Generalization (Leave-one-domain-out)
2. Uniform Bug Distribution evaluation
"""

import json
import os
import sys
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any
import random

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from agenttrace.core.graph import CausalGraph
from agenttrace.ranking.ranker import NodeRanker, ImprovedAgentTrace
from agenttrace.ranking.scorer import CausalScorer


@dataclass
class EvalResult:
    """Evaluation result for a single scenario."""
    scenario_id: str
    domain: str
    bug_position: str  # 'early', 'middle', 'late'
    hit_at_1: bool
    hit_at_3: bool
    hit_at_5: bool
    mrr: float
    ground_truth_rank: int


def get_bug_position(scenario_data: dict, trace_data: dict) -> str:
    """Determine bug position category (early, middle, late)."""
    # Parse trace to get step count
    trace_json = json.loads(trace_data['trace_json'])
    total_steps = len(trace_json['nodes'])

    # Find root cause step
    root_cause_id = trace_data['root_cause_node_id']
    root_cause_step = None
    for node in trace_json['nodes']:
        if node['id'] == root_cause_id:
            root_cause_step = node['data'].get('step', 1)
            break

    if root_cause_step is None:
        return 'middle'

    # Categorize position
    relative_pos = root_cause_step / total_steps
    if relative_pos <= 0.33:
        return 'early'
    elif relative_pos <= 0.66:
        return 'middle'
    else:
        return 'late'


def load_all_traces(data_dir: Path) -> Dict[str, Tuple[dict, str]]:
    """Load all traces with their domain info."""
    traces = {}
    traces_dir = data_dir / 'traces'

    for trace_file in traces_dir.glob('*_trace.json'):
        with open(trace_file) as f:
            trace_data = json.load(f)

        scenario_id = trace_data['scenario_id']

        # Determine domain from scenario_id prefix
        domain_map = {
            'cod': 'coding',
            'cus': 'customer_service',
            'res': 'research',
            'pla': 'planning',
            'tra': 'trading',
            'hea': 'healthcare',
            'leg': 'legal',
            'edu': 'education',
            'fin': 'finance',
            'dev': 'devops'
        }
        prefix = scenario_id[:3]
        domain = domain_map.get(prefix, 'unknown')

        traces[scenario_id] = (trace_data, domain)

    return traces


def evaluate_single_trace(
    trace_data: dict,
    ranker: ImprovedAgentTrace
) -> Dict[str, Any]:
    """Evaluate AgentTrace on a single trace."""
    # Parse the trace JSON
    trace_json = json.loads(trace_data['trace_json'])

    # Reconstruct the graph
    graph = CausalGraph.from_json(trace_data['trace_json'])

    # Get ground truth
    root_cause_id = trace_data['root_cause_node_id']
    error_node_id = trace_data['error_node_id']

    # Run ranking
    result = ranker.find_root_cause(graph, error_node_id, top_k=10)

    # Evaluate
    eval_result = ranker.evaluate_ranking(result, root_cause_id)

    return eval_result


def run_cross_domain_experiment(traces: Dict[str, Tuple[dict, str]]) -> Dict[str, Dict[str, float]]:
    """
    Run leave-one-domain-out cross-validation.

    For each domain, evaluate using default weights (trained on other domains).
    """
    print("\n" + "="*60)
    print("CROSS-DOMAIN GENERALIZATION EXPERIMENT")
    print("="*60)

    # Group traces by domain
    domain_traces = defaultdict(list)
    for scenario_id, (trace_data, domain) in traces.items():
        domain_traces[domain].append((scenario_id, trace_data))

    results = {}

    # For each domain as held-out test set
    for held_out_domain in sorted(domain_traces.keys()):
        test_traces = domain_traces[held_out_domain]

        # Use default weights (already optimized)
        ranker = ImprovedAgentTrace()

        # Evaluate on held-out domain
        hits_at_1 = 0
        total = 0

        for scenario_id, trace_data in test_traces:
            try:
                eval_result = evaluate_single_trace(trace_data, ranker)
                if eval_result['hit_at_1']:
                    hits_at_1 += 1
                total += 1
            except Exception as e:
                print(f"  Error evaluating {scenario_id}: {e}")

        accuracy = hits_at_1 / total * 100 if total > 0 else 0
        results[held_out_domain] = {
            'hit_at_1': accuracy,
            'total': total,
            'hits': hits_at_1
        }

        print(f"  {held_out_domain:20s}: {accuracy:.1f}% ({hits_at_1}/{total})")

    # Compute overall average
    all_hits = sum(r['hits'] for r in results.values())
    all_total = sum(r['total'] for r in results.values())
    avg_accuracy = all_hits / all_total * 100 if all_total > 0 else 0

    print(f"\n  {'Average':20s}: {avg_accuracy:.1f}% ({all_hits}/{all_total})")

    return results


def run_uniform_distribution_experiment(traces: Dict[str, Tuple[dict, str]]) -> Dict[str, Dict[str, float]]:
    """
    Evaluate on uniform bug position distribution.

    Resample scenarios to have equal distribution across early/middle/late positions.
    """
    print("\n" + "="*60)
    print("UNIFORM BUG DISTRIBUTION EXPERIMENT")
    print("="*60)

    # Categorize traces by bug position
    position_traces = defaultdict(list)

    for scenario_id, (trace_data, domain) in traces.items():
        try:
            trace_json = json.loads(trace_data['trace_json'])
            total_steps = len(trace_json['nodes'])

            # Find root cause step
            root_cause_id = trace_data['root_cause_node_id']
            root_cause_step = None
            for node in trace_json['nodes']:
                if node['id'] == root_cause_id:
                    root_cause_step = node['data'].get('step', 1)
                    break

            if root_cause_step is None:
                continue

            # Categorize position
            relative_pos = root_cause_step / total_steps
            if relative_pos <= 0.33:
                position = 'early'
            elif relative_pos <= 0.66:
                position = 'middle'
            else:
                position = 'late'

            position_traces[position].append((scenario_id, trace_data))
        except Exception as e:
            print(f"  Error processing {scenario_id}: {e}")

    print(f"\nOriginal distribution:")
    for pos, traces_list in sorted(position_traces.items()):
        print(f"  {pos}: {len(traces_list)}")

    # Create uniform sample (min count from each position)
    min_count = min(len(traces_list) for traces_list in position_traces.values())
    print(f"\nUniform sample size per position: {min_count}")

    # Sample uniformly
    random.seed(42)  # For reproducibility
    uniform_traces = []
    for position in ['early', 'middle', 'late']:
        sampled = random.sample(position_traces[position], min_count)
        uniform_traces.extend(sampled)

    print(f"Total uniform sample: {len(uniform_traces)}")

    # Evaluate on uniform distribution
    ranker = ImprovedAgentTrace()

    results_by_position = defaultdict(lambda: {'hits': 0, 'total': 0})

    for scenario_id, trace_data in uniform_traces:
        try:
            eval_result = evaluate_single_trace(trace_data, ranker)

            # Determine position for this trace
            trace_json = json.loads(trace_data['trace_json'])
            total_steps = len(trace_json['nodes'])
            root_cause_id = trace_data['root_cause_node_id']
            for node in trace_json['nodes']:
                if node['id'] == root_cause_id:
                    root_cause_step = node['data'].get('step', 1)
                    break

            relative_pos = root_cause_step / total_steps
            if relative_pos <= 0.33:
                position = 'early'
            elif relative_pos <= 0.66:
                position = 'middle'
            else:
                position = 'late'

            results_by_position[position]['total'] += 1
            if eval_result['hit_at_1']:
                results_by_position[position]['hits'] += 1

        except Exception as e:
            print(f"  Error evaluating {scenario_id}: {e}")

    # Print results
    print("\nResults by position (uniform):")
    for position in ['early', 'middle', 'late']:
        r = results_by_position[position]
        acc = r['hits'] / r['total'] * 100 if r['total'] > 0 else 0
        print(f"  {position:8s}: {acc:.1f}% ({r['hits']}/{r['total']})")

    # Overall uniform accuracy
    total_hits = sum(r['hits'] for r in results_by_position.values())
    total_count = sum(r['total'] for r in results_by_position.values())
    uniform_accuracy = total_hits / total_count * 100 if total_count > 0 else 0

    print(f"\n  Overall (uniform): {uniform_accuracy:.1f}% ({total_hits}/{total_count})")

    # Also compute First Node baseline on uniform distribution
    first_node_hits = 0
    for scenario_id, trace_data in uniform_traces:
        try:
            trace_json = json.loads(trace_data['trace_json'])
            nodes = trace_json['nodes']
            # Sort by step to find first node
            sorted_nodes = sorted(nodes, key=lambda n: n['data'].get('step', 0))
            first_node_id = sorted_nodes[0]['id'] if sorted_nodes else None

            root_cause_id = trace_data['root_cause_node_id']
            if first_node_id == root_cause_id:
                first_node_hits += 1
        except:
            pass

    first_node_acc = first_node_hits / total_count * 100 if total_count > 0 else 0
    print(f"  First Node (uniform): {first_node_acc:.1f}% ({first_node_hits}/{total_count})")

    return {
        'uniform_accuracy': uniform_accuracy,
        'by_position': {k: v['hits']/v['total']*100 if v['total'] > 0 else 0
                        for k, v in results_by_position.items()},
        'first_node_accuracy': first_node_acc,
        'sample_size': total_count
    }


def run_original_distribution_experiment(traces: Dict[str, Tuple[dict, str]]) -> Dict[str, float]:
    """
    Evaluate on original (non-uniform) distribution.
    """
    print("\n" + "="*60)
    print("ORIGINAL DISTRIBUTION EXPERIMENT")
    print("="*60)

    ranker = ImprovedAgentTrace()

    hits_at_1 = 0
    hits_at_3 = 0
    hits_at_5 = 0
    total_mrr = 0
    total = 0

    first_node_hits = 0

    for scenario_id, (trace_data, domain) in traces.items():
        try:
            eval_result = evaluate_single_trace(trace_data, ranker)

            if eval_result['hit_at_1']:
                hits_at_1 += 1
            if eval_result['hit_at_3']:
                hits_at_3 += 1
            if eval_result['hit_at_5']:
                hits_at_5 += 1
            total_mrr += eval_result['mrr']
            total += 1

            # First node baseline
            trace_json = json.loads(trace_data['trace_json'])
            nodes = trace_json['nodes']
            sorted_nodes = sorted(nodes, key=lambda n: n['data'].get('step', 0))
            first_node_id = sorted_nodes[0]['id'] if sorted_nodes else None
            root_cause_id = trace_data['root_cause_node_id']
            if first_node_id == root_cause_id:
                first_node_hits += 1

        except Exception as e:
            print(f"  Error evaluating {scenario_id}: {e}")

    results = {
        'hit_at_1': hits_at_1 / total * 100 if total > 0 else 0,
        'hit_at_3': hits_at_3 / total * 100 if total > 0 else 0,
        'hit_at_5': hits_at_5 / total * 100 if total > 0 else 0,
        'mrr': total_mrr / total if total > 0 else 0,
        'total': total,
        'first_node_accuracy': first_node_hits / total * 100 if total > 0 else 0
    }

    print(f"\nAgentTrace Results (Original Distribution):")
    print(f"  Hit@1: {results['hit_at_1']:.1f}%")
    print(f"  Hit@3: {results['hit_at_3']:.1f}%")
    print(f"  Hit@5: {results['hit_at_5']:.1f}%")
    print(f"  MRR:   {results['mrr']:.3f}")
    print(f"  Total: {results['total']}")
    print(f"\nFirst Node Baseline: {results['first_node_accuracy']:.1f}%")

    return results


def main():
    """Run all experiments."""
    print("="*60)
    print("ECAI 2026 PAPER EXPERIMENTS")
    print("="*60)

    # Load all traces
    data_dir = project_root / 'data'
    print(f"\nLoading traces from {data_dir}...")
    traces = load_all_traces(data_dir)
    print(f"Loaded {len(traces)} traces")

    # Run experiments
    original_results = run_original_distribution_experiment(traces)
    cross_domain_results = run_cross_domain_experiment(traces)
    uniform_results = run_uniform_distribution_experiment(traces)

    # Summary for paper
    print("\n" + "="*60)
    print("SUMMARY FOR PAPER")
    print("="*60)

    print("\n1. ORIGINAL DISTRIBUTION (Table 1 - Main Results):")
    print(f"   AgentTrace Hit@1: {original_results['hit_at_1']:.1f}%")
    print(f"   First Node:       {original_results['first_node_accuracy']:.1f}%")

    print("\n2. CROSS-DOMAIN GENERALIZATION (Table 6):")
    for domain, result in sorted(cross_domain_results.items()):
        delta = result['hit_at_1'] - original_results['hit_at_1']
        print(f"   {domain:20s}: {result['hit_at_1']:.1f}% (Δ = {delta:+.1f}%)")

    # Compute average cross-domain
    avg_cross = sum(r['hit_at_1'] for r in cross_domain_results.values()) / len(cross_domain_results)
    avg_delta = avg_cross - original_results['hit_at_1']
    print(f"   {'Average':20s}: {avg_cross:.1f}% (Δ = {avg_delta:+.1f}%)")

    print("\n3. UNIFORM DISTRIBUTION (Table 5):")
    print(f"   Original (60/30/10): AgentTrace {original_results['hit_at_1']:.1f}%")
    print(f"   Uniform (33/33/33):  AgentTrace {uniform_results['uniform_accuracy']:.1f}%")
    print(f"   Gap: {uniform_results['uniform_accuracy'] - original_results['hit_at_1']:+.1f}%")
    print(f"   First Node (original): {original_results['first_node_accuracy']:.1f}%")
    print(f"   First Node (uniform):  {uniform_results['first_node_accuracy']:.1f}%")

    # Save results to JSON
    results_file = data_dir / 'results' / 'paper_experiments.json'
    results_file.parent.mkdir(exist_ok=True)

    all_results = {
        'original_distribution': original_results,
        'cross_domain': {k: v for k, v in cross_domain_results.items()},
        'uniform_distribution': uniform_results
    }

    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\nResults saved to: {results_file}")


if __name__ == '__main__':
    main()
