"""
Run All Baselines and Generate Comprehensive Comparison.

Baselines:
1. LLM-Direct: Simple prompting
2. Chain-of-Thought (CoT): Step-by-step reasoning
3. ReAct: Reasoning + Acting pattern
4. Log-Only: First error heuristic
5. Random: Chance performance
"""

import json
import asyncio
from pathlib import Path
from datetime import datetime
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from agenttrace.core.graph import CausalGraph
from experiments.benchmark.scenario_generator import ScenarioDomain
from experiments.baselines.cot_baseline import run_cot_baseline, CoTAnalyzer
from experiments.baselines.react_baseline import run_react_baseline, ReActAnalyzer


def load_data():
    """Load traces and scenarios."""
    traces_dir = Path("data/traces")
    scenarios_dir = Path("data/scenarios")

    traces = {}
    scenarios = {}

    for trace_file in traces_dir.glob("*_trace.json"):
        with open(trace_file) as f:
            trace = json.load(f)
            traces[trace['scenario_id']] = trace

    for domain in ScenarioDomain:
        domain_dir = scenarios_dir / domain.value
        if domain_dir.exists():
            for f in domain_dir.glob("*.json"):
                with open(f) as file:
                    scenario = json.load(file)
                    scenarios[scenario['scenario_id']] = scenario

    return traces, scenarios


def evaluate_agenttrace(traces):
    """Evaluate AgentTrace."""
    results = {'hit_at_1': 0, 'hit_at_3': 0, 'hit_at_5': 0, 'total': 0, 'mrr_sum': 0}

    for sid, trace in traces.items():
        gt_root = trace.get('root_cause_node_id')
        error_node = trace.get('error_node_id')

        if not gt_root or not error_node:
            continue

        results['total'] += 1
        graph = CausalGraph.from_json(trace['trace_json'])

        try:
            backward_nodes = graph.trace_backward(error_node)
            backward_ids = [n.id for n in backward_nodes]

            if gt_root in backward_ids:
                rank = backward_ids.index(gt_root) + 1
                results['mrr_sum'] += 1.0 / rank

                if rank == 1:
                    results['hit_at_1'] += 1
                if rank <= 3:
                    results['hit_at_3'] += 1
                if rank <= 5:
                    results['hit_at_5'] += 1
        except Exception:
            pass

    total = results['total']
    return {
        'method': 'AgentTrace',
        'hit_at_1': results['hit_at_1'] / total if total else 0,
        'hit_at_3': results['hit_at_3'] / total if total else 0,
        'hit_at_5': results['hit_at_5'] / total if total else 0,
        'mrr': results['mrr_sum'] / total if total else 0,
        'total': total
    }


def evaluate_llm_baseline(results_dir: Path, scenarios: dict, method_name: str):
    """Evaluate an LLM-based baseline."""
    results = {'correct': 0, 'total': 0}

    for sid, scenario in scenarios.items():
        # Try different file patterns
        for pattern in [f"{sid}_llm_direct.json", f"{sid}_cot.json", f"{sid}_react.json"]:
            result_file = results_dir / pattern
            if result_file.exists():
                break
        else:
            continue

        with open(result_file) as f:
            llm_result = json.load(f)

        results['total'] += 1

        # Check if correct
        root_step = scenario.get('root_cause_step', 0)
        bug_type = scenario.get('bug_type', '')

        identified = llm_result.get('identified_root_cause', '').lower()

        # Check for step or bug type mention
        step_match = f"step {root_step}" in identified
        bug_match = bug_type.replace('_', ' ') in identified

        # Also check reasoning steps for CoT
        reasoning = str(llm_result.get('reasoning_steps', [])).lower()
        step_in_reasoning = f"step {root_step}" in reasoning

        if step_match or bug_match or step_in_reasoning:
            results['correct'] += 1

    total = results['total']
    return {
        'method': method_name,
        'hit_at_1': results['correct'] / total if total else 0,
        'hit_at_3': results['correct'] / total if total else 0,
        'hit_at_5': results['correct'] / total if total else 0,
        'total': total
    }


async def run_new_baselines(traces, scenarios):
    """Run CoT and ReAct baselines."""
    print("\n" + "="*60)
    print("Running Chain-of-Thought Baseline...")
    print("="*60)

    cot_results = await run_cot_baseline(max_concurrent=10)
    print(f"CoT: Analyzed {len(cot_results)} traces")

    print("\n" + "="*60)
    print("Running ReAct Baseline...")
    print("="*60)

    react_results = await run_react_baseline(max_concurrent=10)
    print(f"ReAct: Analyzed {len(react_results)} traces")

    return cot_results, react_results


def generate_comparison_table(all_results):
    """Generate comprehensive comparison table."""
    table = r"""
\begin{table*}[t]
\centering
\caption{Comprehensive Baseline Comparison for Root Cause Localization}
\label{tab:all_baselines}
\begin{tabular}{llcccc}
\toprule
\textbf{Category} & \textbf{Method} & \textbf{Hit@1} & \textbf{Hit@3} & \textbf{Hit@5} & \textbf{MRR} \\
\midrule
\multirow{2}{*}{Structured} & \textbf{AgentTrace (Ours)} & """

    at = all_results.get('AgentTrace', {})
    table += f"{at.get('hit_at_1', 0):.1%} & {at.get('hit_at_3', 0):.1%} & "
    table += f"{at.get('hit_at_5', 0):.1%} & {at.get('mrr', 0):.3f} \\\\\n"

    table += r"& Log-Only & "
    lo = all_results.get('Log-Only', {})
    table += f"{lo.get('hit_at_1', 0):.1%} & {lo.get('hit_at_3', 0):.1%} & "
    table += f"{lo.get('hit_at_5', 0):.1%} & - \\\\\n"

    table += r"\midrule" + "\n"
    table += r"\multirow{3}{*}{LLM-based} & LLM-Direct (GPT-4o) & "
    ld = all_results.get('LLM-Direct', {})
    table += f"{ld.get('hit_at_1', 0):.1%} & {ld.get('hit_at_3', 0):.1%} & "
    table += f"{ld.get('hit_at_5', 0):.1%} & - \\\\\n"

    table += r"& Chain-of-Thought & "
    cot = all_results.get('CoT', {})
    table += f"{cot.get('hit_at_1', 0):.1%} & {cot.get('hit_at_3', 0):.1%} & "
    table += f"{cot.get('hit_at_5', 0):.1%} & - \\\\\n"

    table += r"& ReAct & "
    react = all_results.get('ReAct', {})
    table += f"{react.get('hit_at_1', 0):.1%} & {react.get('hit_at_3', 0):.1%} & "
    table += f"{react.get('hit_at_5', 0):.1%} & - \\\\\n"

    table += r"\midrule" + "\n"
    table += r"Chance & Random & "
    rand = all_results.get('Random', {})
    table += f"{rand.get('hit_at_1', 0):.1%} & {rand.get('hit_at_3', 0):.1%} & "
    table += f"{rand.get('hit_at_5', 0):.1%} & - \\\\\n"

    table += r"""
\bottomrule
\end{tabular}
\end{table*}
"""
    return table


async def main():
    """Run all baselines and generate comparison."""
    print("="*70)
    print("COMPREHENSIVE BASELINE EVALUATION")
    print("="*70)

    traces, scenarios = load_data()
    print(f"\nLoaded {len(traces)} traces, {len(scenarios)} scenarios")

    all_results = {}

    # 1. AgentTrace
    print("\nEvaluating AgentTrace...")
    at_results = evaluate_agenttrace(traces)
    all_results['AgentTrace'] = at_results
    print(f"  Hit@3: {at_results['hit_at_3']:.1%}")

    # 2. Run new baselines (CoT, ReAct)
    await run_new_baselines(traces, scenarios)

    # 3. Evaluate LLM-Direct
    print("\nEvaluating LLM-Direct...")
    ld_results = evaluate_llm_baseline(
        Path("data/results/llm_direct"), scenarios, "LLM-Direct"
    )
    all_results['LLM-Direct'] = ld_results
    print(f"  Hit@1: {ld_results['hit_at_1']:.1%}")

    # 4. Evaluate CoT
    print("\nEvaluating Chain-of-Thought...")
    cot_results = evaluate_llm_baseline(
        Path("data/results/cot"), scenarios, "CoT"
    )
    all_results['CoT'] = cot_results
    print(f"  Hit@1: {cot_results['hit_at_1']:.1%}")

    # 5. Evaluate ReAct
    print("\nEvaluating ReAct...")
    react_results = evaluate_llm_baseline(
        Path("data/results/react"), scenarios, "ReAct"
    )
    all_results['ReAct'] = react_results
    print(f"  Hit@1: {react_results['hit_at_1']:.1%}")

    # 6. Log-Only (from fair evaluation)
    print("\nLoading Log-Only results...")
    fair_results_file = Path("data/results/fair_evaluation_results.json")
    if fair_results_file.exists():
        with open(fair_results_file) as f:
            fair_results = json.load(f)
        for r in fair_results:
            if 'Log-Only' in r['method']:
                all_results['Log-Only'] = r

    # 7. Random baseline
    import numpy as np
    random_hit1 = np.mean([1/t.get('node_count', 5) for t in traces.values()])
    random_hit3 = np.mean([min(3/t.get('node_count', 5), 1) for t in traces.values()])
    random_hit5 = np.mean([min(5/t.get('node_count', 5), 1) for t in traces.values()])
    all_results['Random'] = {
        'method': 'Random',
        'hit_at_1': random_hit1,
        'hit_at_3': random_hit3,
        'hit_at_5': random_hit5
    }

    # Generate table
    table = generate_comparison_table(all_results)

    output_dir = Path("data/results")
    with open(output_dir / "all_baselines_table.tex", 'w') as f:
        f.write(table)
    print(f"\nTable saved to: {output_dir / 'all_baselines_table.tex'}")

    # Save results
    with open(output_dir / "all_baselines_results.json", 'w') as f:
        json.dump(all_results, f, indent=2)

    # Print summary
    print("\n" + "="*70)
    print("COMPREHENSIVE COMPARISON SUMMARY")
    print("="*70)
    print(f"\n{'Method':<25} {'Hit@1':>10} {'Hit@3':>10} {'Hit@5':>10}")
    print("-" * 55)

    for method in ['AgentTrace', 'CoT', 'ReAct', 'LLM-Direct', 'Log-Only', 'Random']:
        r = all_results.get(method, {})
        print(f"{method:<25} {r.get('hit_at_1', 0):>10.1%} {r.get('hit_at_3', 0):>10.1%} {r.get('hit_at_5', 0):>10.1%}")

    print("\n" + "="*70)


if __name__ == "__main__":
    asyncio.run(main())
