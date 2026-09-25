"""
Evaluate all methods on Blind Benchmark.

This script runs evaluation on the blinded data where bug markers
have been removed, providing a fair comparison.
"""

import json
import asyncio
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Any, List, Optional
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from agenttrace.core.graph import CausalGraph
from agenttrace.ranking.ranker import NodeRanker, ImprovedAgentTrace


@dataclass
class EvaluationResult:
    """Result for a single scenario evaluation."""
    scenario_id: str
    method: str
    ground_truth_node_id: str
    predicted_node_id: Optional[str]
    rank: int
    score: float
    hit_at_1: bool
    hit_at_3: bool
    hit_at_5: bool
    mrr: float


def load_blind_data():
    """Load blind benchmark data and ground truth."""
    blind_dir = Path("data/blind_benchmark")
    traces_dir = blind_dir / "traces"
    gt_file = blind_dir / "ground_truth" / "all_ground_truth.json"

    # Load ground truth
    with open(gt_file) as f:
        ground_truth = json.load(f)

    # Load traces
    traces = {}
    for trace_file in traces_dir.glob("*_trace.json"):
        with open(trace_file) as f:
            trace = json.load(f)
            scenario_id = trace['scenario_id']
            traces[scenario_id] = trace

    return traces, ground_truth


def evaluate_original_agenttrace(
    traces: Dict[str, Any],
    ground_truth: Dict[str, Any]
) -> List[EvaluationResult]:
    """
    Evaluate original AgentTrace (backward trace ordering).

    This is the baseline - just uses causal distance ordering.
    """
    results = []

    for scenario_id, trace in traces.items():
        if scenario_id not in ground_truth:
            continue

        gt = ground_truth[scenario_id]
        gt_root = gt.get('root_cause_node_id')
        error_node = gt.get('error_node_id')

        if not gt_root or not error_node:
            continue

        # Load graph
        graph = CausalGraph.from_json(trace['trace_json'])

        # Original backward trace
        backward_nodes = graph.trace_backward(error_node)
        backward_ids = [n.id for n in backward_nodes]

        # Check rank
        if gt_root in backward_ids:
            rank = backward_ids.index(gt_root) + 1
            predicted = backward_ids[0] if backward_ids else None
        else:
            rank = -1
            predicted = None

        results.append(EvaluationResult(
            scenario_id=scenario_id,
            method='AgentTrace-Original',
            ground_truth_node_id=gt_root,
            predicted_node_id=predicted,
            rank=rank,
            score=0.0,
            hit_at_1=rank == 1,
            hit_at_3=1 <= rank <= 3,
            hit_at_5=1 <= rank <= 5,
            mrr=1/rank if rank > 0 else 0
        ))

    return results


def evaluate_improved_agenttrace(
    traces: Dict[str, Any],
    ground_truth: Dict[str, Any]
) -> List[EvaluationResult]:
    """
    Evaluate improved AgentTrace with node ranking.
    """
    results = []
    improved = ImprovedAgentTrace()

    for scenario_id, trace in traces.items():
        if scenario_id not in ground_truth:
            continue

        gt = ground_truth[scenario_id]
        gt_root = gt.get('root_cause_node_id')
        error_node = gt.get('error_node_id')

        if not gt_root or not error_node:
            continue

        # Load graph
        graph = CausalGraph.from_json(trace['trace_json'])

        # Improved ranking
        ranking_result = improved.find_root_cause(graph, error_node, top_k=10)

        rank = ranking_result.get_rank_of(gt_root)
        predicted = ranking_result.top_prediction
        score = 0.0

        # Get score for ground truth
        for rc in ranking_result.ranked_candidates:
            if rc.node.id == gt_root:
                score = rc.score
                break

        results.append(EvaluationResult(
            scenario_id=scenario_id,
            method='AgentTrace-Ranked',
            ground_truth_node_id=gt_root,
            predicted_node_id=predicted,
            rank=rank,
            score=score,
            hit_at_1=rank == 1,
            hit_at_3=1 <= rank <= 3,
            hit_at_5=1 <= rank <= 5,
            mrr=1/rank if rank > 0 else 0
        ))

    return results


def evaluate_llm_baseline_blind(
    traces: Dict[str, Any],
    ground_truth: Dict[str, Any],
    scenarios_dir: Path
) -> List[EvaluationResult]:
    """
    Evaluate LLM-Direct baseline on blind data.

    Uses GPT-4o to identify root cause without explicit markers.
    """
    from openai import OpenAI

    client = OpenAI()
    results = []

    PROMPT = '''You are an expert debugger analyzing a multi-agent system trace.

The trace shows a series of agent interactions that resulted in an error or failure.
Your task is to identify which step is the ROOT CAUSE of the problem.

## Trace Data
{trace_data}

## Task Description
{task_description}

Analyze the trace carefully:
1. Find where the error/failure manifests
2. Trace backward to find where things first went wrong
3. Identify the step that is the SOURCE of the problem (not just where it shows up)

Return ONLY the step number (as an integer) that is the root cause.
Just the number, nothing else.'''

    # Load scenarios for task descriptions
    blind_scenarios_dir = Path("data/blind_benchmark/scenarios")

    for scenario_id, trace in list(traces.items())[:50]:  # Sample 50 for cost
        if scenario_id not in ground_truth:
            continue

        gt = ground_truth[scenario_id]
        gt_root = gt.get('root_cause_node_id')
        gt_step = gt.get('root_cause_step')

        if not gt_root or not gt_step:
            continue

        # Get task description from scenario
        domain = gt.get('domain', 'coding')
        scenario_file = blind_scenarios_dir / domain / f"{scenario_id}.json"

        task_desc = "Multi-agent task execution"
        if scenario_file.exists():
            with open(scenario_file) as f:
                scenario = json.load(f)
                task_desc = scenario.get('task_description', task_desc)

        # Format trace for LLM
        trace_json = json.loads(trace['trace_json'])
        trace_summary = []
        for node in trace_json.get('nodes', []):
            data = node.get('data', {})
            step = data.get('step', '?')
            action = data.get('action', 'unknown')
            agent = node.get('agent_id', 'unknown')
            content = data.get('content', {})
            trace_summary.append(f"Step {step}: [{agent}] {action} - {json.dumps(content)[:100]}")

        trace_data = "\n".join(trace_summary)

        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",  # Use mini for cost efficiency
                messages=[
                    {"role": "user", "content": PROMPT.format(
                        trace_data=trace_data,
                        task_description=task_desc
                    )}
                ],
                max_tokens=10,
                temperature=0
            )

            predicted_step = response.choices[0].message.content.strip()

            # Check if predicted step matches ground truth
            try:
                predicted_step_int = int(predicted_step)
                is_correct = predicted_step_int == gt_step
            except ValueError:
                is_correct = False
                predicted_step_int = -1

            results.append(EvaluationResult(
                scenario_id=scenario_id,
                method='LLM-Blind',
                ground_truth_node_id=gt_root,
                predicted_node_id=f"step_{predicted_step}",
                rank=1 if is_correct else -1,
                score=0.0,
                hit_at_1=is_correct,
                hit_at_3=is_correct,
                hit_at_5=is_correct,
                mrr=1.0 if is_correct else 0.0
            ))

        except Exception as e:
            print(f"  Error for {scenario_id}: {e}")
            continue

    return results


def aggregate_results(results: List[EvaluationResult]) -> Dict[str, Any]:
    """Aggregate evaluation results."""
    if not results:
        return {}

    total = len(results)
    hit_at_1 = sum(1 for r in results if r.hit_at_1)
    hit_at_3 = sum(1 for r in results if r.hit_at_3)
    hit_at_5 = sum(1 for r in results if r.hit_at_5)
    found = sum(1 for r in results if r.rank > 0)
    mrr = sum(r.mrr for r in results) / total

    return {
        'total': total,
        'hit_at_1': hit_at_1,
        'hit_at_1_rate': hit_at_1 / total,
        'hit_at_3': hit_at_3,
        'hit_at_3_rate': hit_at_3 / total,
        'hit_at_5': hit_at_5,
        'hit_at_5_rate': hit_at_5 / total,
        'found': found,
        'recall': found / total,
        'mrr': mrr
    }


def main():
    """Run blind benchmark evaluation."""
    print("="*70)
    print("BLIND BENCHMARK EVALUATION")
    print("="*70)

    print("\nLoading blind benchmark data...")
    traces, ground_truth = load_blind_data()
    print(f"Loaded {len(traces)} traces, {len(ground_truth)} ground truth entries")

    all_results = {}

    # 1. Original AgentTrace
    print("\n" + "-"*50)
    print("Evaluating: AgentTrace-Original (causal distance ordering)")
    print("-"*50)
    original_results = evaluate_original_agenttrace(traces, ground_truth)
    original_agg = aggregate_results(original_results)
    all_results['AgentTrace-Original'] = original_agg
    print(f"  Hit@1: {original_agg['hit_at_1_rate']:.1%}")
    print(f"  Hit@3: {original_agg['hit_at_3_rate']:.1%}")
    print(f"  Hit@5: {original_agg['hit_at_5_rate']:.1%}")
    print(f"  MRR: {original_agg['mrr']:.3f}")

    # 2. Improved AgentTrace with ranking
    print("\n" + "-"*50)
    print("Evaluating: AgentTrace-Ranked (with node scoring)")
    print("-"*50)
    ranked_results = evaluate_improved_agenttrace(traces, ground_truth)
    ranked_agg = aggregate_results(ranked_results)
    all_results['AgentTrace-Ranked'] = ranked_agg
    print(f"  Hit@1: {ranked_agg['hit_at_1_rate']:.1%}")
    print(f"  Hit@3: {ranked_agg['hit_at_3_rate']:.1%}")
    print(f"  Hit@5: {ranked_agg['hit_at_5_rate']:.1%}")
    print(f"  MRR: {ranked_agg['mrr']:.3f}")

    # 3. LLM Baseline on blind data (sample)
    print("\n" + "-"*50)
    print("Evaluating: LLM-Blind (GPT-4o-mini on blinded traces)")
    print("-"*50)
    try:
        llm_results = evaluate_llm_baseline_blind(
            traces, ground_truth,
            Path("data/blind_benchmark/scenarios")
        )
        llm_agg = aggregate_results(llm_results)
        all_results['LLM-Blind'] = llm_agg
        print(f"  Hit@1: {llm_agg['hit_at_1_rate']:.1%}")
        print(f"  Sample size: {llm_agg['total']}")
    except Exception as e:
        print(f"  Skipped due to error: {e}")

    # Summary comparison
    print("\n" + "="*70)
    print("SUMMARY COMPARISON")
    print("="*70)

    print("\n{:<25} {:>10} {:>10} {:>10} {:>10}".format(
        "Method", "Hit@1", "Hit@3", "Hit@5", "MRR"
    ))
    print("-" * 65)

    for method, agg in all_results.items():
        print("{:<25} {:>10.1%} {:>10.1%} {:>10.1%} {:>10.3f}".format(
            method,
            agg['hit_at_1_rate'],
            agg['hit_at_3_rate'],
            agg['hit_at_5_rate'],
            agg['mrr']
        ))

    # Calculate improvement
    if 'AgentTrace-Original' in all_results and 'AgentTrace-Ranked' in all_results:
        orig = all_results['AgentTrace-Original']
        ranked = all_results['AgentTrace-Ranked']
        improvement = ranked['hit_at_1_rate'] - orig['hit_at_1_rate']
        print(f"\nHit@1 Improvement: {improvement:+.1%}")

    # Save results
    output_dir = Path("data/results")
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "blind_benchmark_results.json", 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\nResults saved to: {output_dir / 'blind_benchmark_results.json'}")

    print("\n" + "="*70)
    print("BLIND BENCHMARK EVALUATION COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
