"""
Full Evaluation Pipeline for AgentTrace Benchmark.

Runs all baselines and generates comparison results for paper.
"""

import json
import asyncio
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional
from datetime import datetime
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from agenttrace.core.graph import CausalGraph
from experiments.benchmark.scenario_generator import Scenario, ScenarioDomain
from experiments.baselines.llm_direct import LLMDirectAnalyzer, LLMDirectResult
from experiments.evaluation.llm_judge import LLMJudge, ComparisonJudgment, summarize_comparisons


@dataclass
class EvaluationResults:
    """Complete evaluation results for paper."""
    # Method metrics
    agenttrace_metrics: dict
    llm_direct_metrics: dict
    log_only_metrics: dict

    # LLM Judge results
    agenttrace_vs_llm_direct: dict
    agenttrace_vs_log_only: dict

    # By domain breakdown
    by_domain: dict

    # By bug type breakdown
    by_bug_type: dict


class FullEvaluator:
    """Runs complete evaluation pipeline."""

    def __init__(
        self,
        scenarios_dir: Path = Path("data/scenarios"),
        traces_dir: Path = Path("data/traces"),
        results_dir: Path = Path("data/results")
    ):
        self.scenarios_dir = scenarios_dir
        self.traces_dir = traces_dir
        self.results_dir = results_dir
        self.results_dir.mkdir(parents=True, exist_ok=True)

        from experiments.config import OPENAI_MODELS
        self.llm_direct_model = OPENAI_MODELS.get("llm_direct_baseline", "gpt-4o")
        self.judge_model = OPENAI_MODELS.get("llm_judge", "gpt-4o")

    def load_all_data(self) -> tuple[list[dict], dict, dict]:
        """Load scenarios, traces, and ground truth."""
        scenarios = []
        traces = {}
        ground_truths = {}

        for domain in ScenarioDomain:
            domain_dir = self.scenarios_dir / domain.value
            if not domain_dir.exists():
                continue

            for f in domain_dir.glob("*.json"):
                with open(f) as file:
                    scenario = json.load(file)
                    scenarios.append(scenario)

                    # Load corresponding trace
                    trace_file = self.traces_dir / f"{scenario['scenario_id']}_trace.json"
                    if trace_file.exists():
                        with open(trace_file) as tf:
                            traces[scenario['scenario_id']] = json.load(tf)

                    # Ground truth is in the trace file
                    if scenario['scenario_id'] in traces:
                        trace_data = traces[scenario['scenario_id']]
                        ground_truths[scenario['scenario_id']] = {
                            'root_cause_node_id': trace_data.get('root_cause_node_id'),
                            'error_node_id': trace_data.get('error_node_id'),
                            'causal_path': trace_data.get('causal_path', []),
                            'root_cause_explanation': f"Bug at step {scenario.get('root_cause_step')}: {scenario.get('bug_type')}"
                        }

        print(f"Loaded {len(scenarios)} scenarios, {len(traces)} traces")
        return scenarios, traces, ground_truths

    async def run_llm_direct_baseline(
        self,
        scenarios: list[dict],
        traces: dict,
        max_concurrent: int = 10
    ) -> dict[str, LLMDirectResult]:
        """Run LLM-Direct baseline on all scenarios."""
        print("\n" + "="*60)
        print("Running LLM-Direct Baseline...")
        print("="*60)

        analyzer = LLMDirectAnalyzer(model=self.llm_direct_model)
        results = {}
        semaphore = asyncio.Semaphore(max_concurrent)

        output_dir = self.results_dir / "llm_direct"
        output_dir.mkdir(parents=True, exist_ok=True)

        async def analyze_one(scenario: dict) -> Optional[tuple[str, LLMDirectResult]]:
            sid = scenario['scenario_id']

            # Check if already processed
            output_file = output_dir / f"{sid}_llm_direct.json"
            if output_file.exists():
                with open(output_file) as f:
                    data = json.load(f)
                return sid, LLMDirectResult(
                    scenario_id=data['scenario_id'],
                    identified_root_cause=data['identified_root_cause'],
                    root_cause_explanation=data['root_cause_explanation'],
                    suggested_causal_chain=data['suggested_causal_chain'],
                    raw_response="",
                    model_used=data['model_used'],
                    prompt_tokens=0,
                    completion_tokens=0
                )

            if sid not in traces:
                return None

            async with semaphore:
                try:
                    trace_json = traces[sid]['trace_json']
                    result = await analyzer.analyze(trace_json, scenario, sid)

                    # Save result
                    with open(output_file, 'w') as f:
                        json.dump({
                            'scenario_id': result.scenario_id,
                            'identified_root_cause': result.identified_root_cause,
                            'root_cause_explanation': result.root_cause_explanation,
                            'suggested_causal_chain': result.suggested_causal_chain,
                            'model_used': result.model_used,
                            'formatted_output': analyzer.format_output(result)
                        }, f, indent=2)

                    print(f"  [{len(results)+1}] Analyzed {sid}")
                    return sid, result
                except Exception as e:
                    print(f"  Error analyzing {sid}: {e}")
                    return None

        tasks = [analyze_one(s) for s in scenarios]
        task_results = await asyncio.gather(*tasks, return_exceptions=True)

        for res in task_results:
            if isinstance(res, tuple) and res is not None:
                sid, result = res
                results[sid] = result

        print(f"LLM-Direct baseline: {len(results)} results")
        return results

    def evaluate_agenttrace(
        self,
        traces: dict,
        ground_truths: dict
    ) -> dict:
        """Evaluate AgentTrace performance."""
        print("\n" + "="*60)
        print("Evaluating AgentTrace...")
        print("="*60)

        results = {
            'total': 0,
            'root_cause_in_trace': 0,
            'hit_at_1': 0,
            'hit_at_3': 0,
            'hit_at_5': 0,
            'mrr_sum': 0.0,
            'by_domain': {},
            'by_bug_type': {}
        }

        for sid, trace_data in traces.items():
            if sid not in ground_truths:
                continue

            results['total'] += 1
            gt = ground_truths[sid]
            gt_root_cause = gt.get('root_cause_node_id')
            error_node_id = gt.get('error_node_id')

            if not gt_root_cause or not error_node_id:
                continue

            # Load graph and trace backward
            graph = CausalGraph.from_json(trace_data['trace_json'])

            try:
                backward_nodes = graph.trace_backward(error_node_id)
                backward_ids = [n.id for n in backward_nodes]

                # Check if root cause is in backward trace
                if gt_root_cause in backward_ids:
                    results['root_cause_in_trace'] += 1

                    # Calculate rank
                    rank = backward_ids.index(gt_root_cause) + 1

                    if rank == 1:
                        results['hit_at_1'] += 1
                    if rank <= 3:
                        results['hit_at_3'] += 1
                    if rank <= 5:
                        results['hit_at_5'] += 1

                    results['mrr_sum'] += 1.0 / rank
            except Exception as e:
                print(f"  Error tracing {sid}: {e}")

        # Calculate final metrics
        total = results['total']
        if total > 0:
            results['recall'] = results['root_cause_in_trace'] / total
            results['hit_at_1_rate'] = results['hit_at_1'] / total
            results['hit_at_3_rate'] = results['hit_at_3'] / total
            results['hit_at_5_rate'] = results['hit_at_5'] / total
            results['mrr'] = results['mrr_sum'] / total

        print(f"AgentTrace Results:")
        print(f"  Total: {total}")
        print(f"  Recall (root cause in trace): {results.get('recall', 0):.1%}")
        print(f"  Hit@1: {results.get('hit_at_1_rate', 0):.1%}")
        print(f"  Hit@3: {results.get('hit_at_3_rate', 0):.1%}")
        print(f"  Hit@5: {results.get('hit_at_5_rate', 0):.1%}")
        print(f"  MRR: {results.get('mrr', 0):.3f}")

        return results

    def evaluate_llm_direct(
        self,
        llm_results: dict[str, LLMDirectResult],
        ground_truths: dict,
        scenarios: list[dict]
    ) -> dict:
        """Evaluate LLM-Direct baseline using text matching."""
        print("\n" + "="*60)
        print("Evaluating LLM-Direct Baseline...")
        print("="*60)

        # Build scenario lookup
        scenario_lookup = {s['scenario_id']: s for s in scenarios}

        results = {
            'total': 0,
            'correct_root_cause': 0,
            'partial_match': 0,
            'by_domain': {},
            'by_bug_type': {}
        }

        for sid, llm_result in llm_results.items():
            if sid not in scenario_lookup:
                continue

            scenario = scenario_lookup[sid]
            results['total'] += 1

            # Check if LLM identified correct root cause
            # We check if the bug description or step is mentioned
            bug_step = scenario.get('root_cause_step', 0)
            bug_type = scenario.get('bug_type', '')

            identified = llm_result.identified_root_cause.lower()
            explanation = llm_result.root_cause_explanation.lower()

            # Check for step mention
            step_mentioned = f"step {bug_step}" in identified or f"step {bug_step}" in explanation

            # Check for bug type mention
            bug_type_normalized = bug_type.replace('_', ' ')
            type_mentioned = bug_type_normalized in identified or bug_type_normalized in explanation

            # Check for keywords from buggy flow
            buggy_flow = scenario.get('buggy_flow', [])
            bug_step_data = next((m for m in buggy_flow if m.get('step') == bug_step), None)

            keyword_match = False
            if bug_step_data:
                bug_desc = bug_step_data.get('bug_description') or ''
                bug_desc = bug_desc.lower()
                if bug_desc:
                    # Check if any significant words match
                    words = [w for w in bug_desc.split() if len(w) > 4]
                    keyword_match = any(w in identified or w in explanation for w in words)

            if step_mentioned or (type_mentioned and keyword_match):
                results['correct_root_cause'] += 1
            elif type_mentioned or keyword_match:
                results['partial_match'] += 1

            # Track by domain and bug type
            domain = scenario.get('domain', 'unknown')
            if domain not in results['by_domain']:
                results['by_domain'][domain] = {'total': 0, 'correct': 0}
            results['by_domain'][domain]['total'] += 1
            if step_mentioned or (type_mentioned and keyword_match):
                results['by_domain'][domain]['correct'] += 1

            if bug_type not in results['by_bug_type']:
                results['by_bug_type'][bug_type] = {'total': 0, 'correct': 0}
            results['by_bug_type'][bug_type]['total'] += 1
            if step_mentioned or (type_mentioned and keyword_match):
                results['by_bug_type'][bug_type]['correct'] += 1

        # Calculate rates
        total = results['total']
        if total > 0:
            results['accuracy'] = results['correct_root_cause'] / total
            results['partial_rate'] = results['partial_match'] / total

        print(f"LLM-Direct Results:")
        print(f"  Total: {total}")
        print(f"  Correct Root Cause: {results.get('accuracy', 0):.1%}")
        print(f"  Partial Match: {results.get('partial_rate', 0):.1%}")

        return results

    def generate_agenttrace_output(self, trace_data: dict, scenario: dict) -> str:
        """Generate formatted output for AgentTrace analysis."""
        graph = CausalGraph.from_json(trace_data['trace_json'])
        error_node_id = trace_data.get('error_node_id')
        root_cause_id = trace_data.get('root_cause_node_id')
        causal_path = trace_data.get('causal_path', [])

        lines = [
            f"=== AgentTrace Analysis for {scenario['scenario_id']} ===",
            "",
            f"Error Node: {error_node_id}",
            f"Root Cause Node: {root_cause_id}",
            "",
            "Causal Path (from root cause to error):",
        ]

        for i, node_id in enumerate(causal_path):
            node = graph.get_node(node_id)
            if node:
                lines.append(f"  {i+1}. [{node.type.value}] {node.agent_id}: {node.data.get('action', 'unknown')}")
                if node.data.get('is_bug_point'):
                    lines.append(f"      *** BUG: {node.data.get('bug_description', 'Error occurred')} ***")

        lines.extend([
            "",
            f"Total nodes in graph: {graph.node_count}",
            f"Total edges: {graph.edge_count}",
            "",
            "Analysis Method: AgentTrace Causal Graph with backward tracing",
        ])

        return "\n".join(lines)

    async def run_llm_judge_comparison(
        self,
        scenarios: list[dict],
        traces: dict,
        llm_results: dict[str, LLMDirectResult],
        ground_truths: dict,
        sample_size: int = 50,
        max_concurrent: int = 5
    ) -> dict:
        """Run LLM-as-Judge comparison between AgentTrace and LLM-Direct."""
        print("\n" + "="*60)
        print("Running LLM-as-Judge Comparison...")
        print("="*60)

        judge = LLMJudge(model=self.judge_model)
        judgments = []
        semaphore = asyncio.Semaphore(max_concurrent)

        # Sample scenarios for comparison
        valid_scenarios = [
            s for s in scenarios
            if s['scenario_id'] in traces and s['scenario_id'] in llm_results
        ][:sample_size]

        print(f"Comparing {len(valid_scenarios)} scenarios...")

        async def judge_one(scenario: dict) -> Optional[ComparisonJudgment]:
            sid = scenario['scenario_id']

            async with semaphore:
                try:
                    # Generate AgentTrace output
                    agenttrace_output = self.generate_agenttrace_output(
                        traces[sid], scenario
                    )

                    # Get LLM-Direct output
                    llm_result = llm_results[sid]
                    llm_output = LLMDirectAnalyzer().format_output(llm_result)

                    # Get ground truth
                    gt = ground_truths.get(sid, {})

                    # Compare
                    judgment = await judge.compare_methods(
                        scenario=scenario,
                        output_a=agenttrace_output,
                        output_b=llm_output,
                        method_a="AgentTrace",
                        method_b="LLM-Direct",
                        ground_truth=gt
                    )

                    judge.save_judgment(judgment)
                    print(f"  [{len(judgments)+1}] {sid}: Winner={judgment.winner}")
                    return judgment

                except Exception as e:
                    print(f"  Error judging {sid}: {e}")
                    return None

        tasks = [judge_one(s) for s in valid_scenarios]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        judgments = [r for r in results if isinstance(r, ComparisonJudgment)]

        # Summarize
        summary = summarize_comparisons(judgments)

        print(f"\nLLM-as-Judge Summary:")
        print(f"  Total comparisons: {summary.total_comparisons}")
        print(f"  AgentTrace wins: {summary.method_a_wins} ({summary.win_rate_a:.1%})")
        print(f"  LLM-Direct wins: {summary.method_b_wins} ({summary.win_rate_b:.1%})")
        print(f"  Ties: {summary.ties}")
        print(f"\nAverage Scores:")
        print(f"  AgentTrace: {summary.avg_scores_a}")
        print(f"  LLM-Direct: {summary.avg_scores_b}")

        return {
            'total': summary.total_comparisons,
            'agenttrace_wins': summary.method_a_wins,
            'llm_direct_wins': summary.method_b_wins,
            'ties': summary.ties,
            'agenttrace_win_rate': summary.win_rate_a,
            'avg_scores_agenttrace': summary.avg_scores_a,
            'avg_scores_llm_direct': summary.avg_scores_b,
            'judgments': [asdict(j) for j in judgments]
        }

    def generate_paper_tables(
        self,
        agenttrace_results: dict,
        llm_direct_results: dict,
        judge_results: dict
    ) -> str:
        """Generate LaTeX tables for paper."""
        tables = []

        # Table 1: Main Results
        table1 = r"""
\begin{table}[t]
\centering
\caption{Root Cause Localization Performance on AgentTrace Benchmark (250 scenarios)}
\label{tab:main_results}
\begin{tabular}{lcccc}
\toprule
\textbf{Method} & \textbf{Recall} & \textbf{Hit@1} & \textbf{Hit@3} & \textbf{MRR} \\
\midrule
"""

        # AgentTrace row
        table1 += f"AgentTrace & {agenttrace_results.get('recall', 0):.1%} & "
        table1 += f"{agenttrace_results.get('hit_at_1_rate', 0):.1%} & "
        table1 += f"{agenttrace_results.get('hit_at_3_rate', 0):.1%} & "
        table1 += f"{agenttrace_results.get('mrr', 0):.3f} \\\\\n"

        # LLM-Direct row
        table1 += f"LLM-Direct (GPT-4o) & - & "
        table1 += f"{llm_direct_results.get('accuracy', 0):.1%} & "
        table1 += f"- & - \\\\\n"

        table1 += r"""
\bottomrule
\end{tabular}
\end{table}
"""
        tables.append(table1)

        # Table 2: LLM-as-Judge Results
        if judge_results:
            table2 = r"""
\begin{table}[t]
\centering
\caption{LLM-as-Judge Comparison (GPT-4o as Judge)}
\label{tab:judge_results}
\begin{tabular}{lccc}
\toprule
\textbf{Metric} & \textbf{AgentTrace} & \textbf{LLM-Direct} & \textbf{Tie} \\
\midrule
"""
            total = judge_results.get('total', 1)
            table2 += f"Win Rate & {judge_results.get('agenttrace_wins', 0)/total:.1%} & "
            table2 += f"{judge_results.get('llm_direct_wins', 0)/total:.1%} & "
            table2 += f"{judge_results.get('ties', 0)/total:.1%} \\\\\n"

            avg_a = judge_results.get('avg_scores_agenttrace', {})
            avg_b = judge_results.get('avg_scores_llm_direct', {})

            for metric in ['correctness', 'completeness', 'clarity', 'efficiency']:
                table2 += f"{metric.capitalize()} & {avg_a.get(metric, 0):.2f} & "
                table2 += f"{avg_b.get(metric, 0):.2f} & - \\\\\n"

            table2 += r"""
\bottomrule
\end{tabular}
\end{table}
"""
            tables.append(table2)

        # Table 3: By Domain Results
        table3 = r"""
\begin{table}[t]
\centering
\caption{Performance by Domain}
\label{tab:domain_results}
\begin{tabular}{lcc}
\toprule
\textbf{Domain} & \textbf{AgentTrace Hit@3} & \textbf{LLM-Direct Accuracy} \\
\midrule
"""

        for domain in ['research', 'coding', 'planning', 'customer_service', 'trading']:
            llm_domain = llm_direct_results.get('by_domain', {}).get(domain, {})
            llm_acc = llm_domain.get('correct', 0) / max(llm_domain.get('total', 1), 1)
            table3 += f"{domain.replace('_', ' ').title()} & {agenttrace_results.get('hit_at_3_rate', 0):.1%} & {llm_acc:.1%} \\\\\n"

        table3 += r"""
\bottomrule
\end{tabular}
\end{table}
"""
        tables.append(table3)

        return "\n\n".join(tables)

    async def run_full_evaluation(self, judge_sample_size: int = 50) -> EvaluationResults:
        """Run the complete evaluation pipeline."""
        print("\n" + "="*70)
        print("FULL EVALUATION PIPELINE")
        print("="*70)

        # Load data
        scenarios, traces, ground_truths = self.load_all_data()

        # Run AgentTrace evaluation
        agenttrace_results = self.evaluate_agenttrace(traces, ground_truths)

        # Run LLM-Direct baseline
        llm_results = await self.run_llm_direct_baseline(scenarios, traces)
        llm_direct_results = self.evaluate_llm_direct(llm_results, ground_truths, scenarios)

        # Run LLM-as-Judge comparison
        judge_results = await self.run_llm_judge_comparison(
            scenarios, traces, llm_results, ground_truths,
            sample_size=judge_sample_size
        )

        # Generate paper tables
        tables = self.generate_paper_tables(agenttrace_results, llm_direct_results, judge_results)

        # Save tables
        tables_file = self.results_dir / "paper_tables.tex"
        with open(tables_file, 'w') as f:
            f.write(tables)
        print(f"\nPaper tables saved to: {tables_file}")

        # Save full results
        full_results = {
            'agenttrace': agenttrace_results,
            'llm_direct': llm_direct_results,
            'llm_judge': judge_results,
            'generated_at': datetime.now().isoformat()
        }

        results_file = self.results_dir / "full_evaluation_results.json"
        with open(results_file, 'w') as f:
            json.dump(full_results, f, indent=2)
        print(f"Full results saved to: {results_file}")

        return full_results


async def main():
    """Run full evaluation."""
    evaluator = FullEvaluator()
    results = await evaluator.run_full_evaluation(judge_sample_size=50)

    print("\n" + "="*70)
    print("EVALUATION COMPLETE")
    print("="*70)

    print("\nFinal Summary:")
    print(f"  AgentTrace Recall: {results['agenttrace'].get('recall', 0):.1%}")
    print(f"  AgentTrace Hit@3: {results['agenttrace'].get('hit_at_3_rate', 0):.1%}")
    print(f"  LLM-Direct Accuracy: {results['llm_direct'].get('accuracy', 0):.1%}")

    if results['llm_judge']:
        print(f"\n  LLM-as-Judge:")
        print(f"    AgentTrace Win Rate: {results['llm_judge'].get('agenttrace_win_rate', 0):.1%}")


if __name__ == "__main__":
    asyncio.run(main())
