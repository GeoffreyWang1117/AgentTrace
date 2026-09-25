#!/usr/bin/env python3
"""
Main script to run all AgentTrace benchmark experiments.

Usage:
    python -m experiments.scripts.run_experiments --phase all
    python -m experiments.scripts.run_experiments --phase generate
    python -m experiments.scripts.run_experiments --phase evaluate
"""

import argparse
import asyncio
import json
from pathlib import Path
from datetime import datetime

# Import experiment modules
from experiments.benchmark.scenario_generator import (
    ScenarioGenerator,
    ScenarioValidator,
    ScenarioDomain,
    Complexity,
    BugType,
    generate_all_scenarios
)
from experiments.benchmark.trace_collector import (
    TraceCollector,
    collect_all_benchmark_traces
)
from experiments.benchmark.ground_truth_labeler import (
    GroundTruthLabeler,
    label_all_traces
)
from experiments.baselines.log_only import run_log_only_baseline, LogOnlyAnalyzer
from experiments.baselines.llm_direct import run_llm_direct_baseline, LLMDirectAnalyzer
from experiments.evaluation.metrics import (
    CausalChainEvaluator,
    RootCauseEvaluator,
    MetricsAggregator,
    EvaluationResult,
    save_results
)
from experiments.evaluation.llm_judge import (
    LLMJudge,
    run_comparison_evaluation,
    summarize_comparisons
)
from experiments.config import (
    DATA_DIR,
    SCENARIOS_DIR,
    TRACES_DIR,
    LABELS_DIR,
    RESULTS_DIR,
    BENCHMARK_CONFIG
)


async def phase_generate(args):
    """Phase 1: Generate scenarios."""
    print("\n" + "="*60)
    print("PHASE 1: GENERATING SCENARIOS")
    print("="*60)

    if args.single_test:
        # Generate just one scenario for testing
        generator = ScenarioGenerator(output_dir=SCENARIOS_DIR)
        print("\nGenerating single test scenario...")

        scenario = await generator.generate_scenario(
            domain=ScenarioDomain.RESEARCH,
            complexity=Complexity.MEDIUM,
            bug_type=BugType.DATA_CORRUPTION,
            index=0
        )

        validator = ScenarioValidator()
        is_valid, errors = validator.validate(scenario)

        if is_valid:
            filepath = generator.save_scenario(scenario)
            print(f"✓ Generated and saved: {filepath}")
            print(f"  Scenario ID: {scenario.scenario_id}")
            print(f"  Agents: {[a.id for a in scenario.agents]}")
            print(f"  Steps: {len(scenario.buggy_flow)}")
        else:
            print(f"✗ Validation failed: {errors}")
    else:
        # Generate full benchmark
        count = args.count or BENCHMARK_CONFIG["scenarios_per_domain"]
        await generate_all_scenarios(
            output_dir=SCENARIOS_DIR,
            scenarios_per_domain=count
        )

    print("\n✓ Scenario generation complete")


async def phase_collect(args):
    """Phase 2: Collect traces."""
    print("\n" + "="*60)
    print("PHASE 2: COLLECTING TRACES")
    print("="*60)

    results = await collect_all_benchmark_traces(
        scenarios_dir=SCENARIOS_DIR,
        traces_dir=TRACES_DIR,
        use_llm=not args.no_llm,
        max_concurrent=args.max_concurrent
    )

    total = sum(len(v) for v in results.values())
    print(f"\n✓ Collected {total} traces")


async def phase_label(args):
    """Phase 3: Generate ground truth labels."""
    print("\n" + "="*60)
    print("PHASE 3: GENERATING GROUND TRUTH LABELS")
    print("="*60)

    results = await label_all_traces(
        scenarios_dir=SCENARIOS_DIR,
        traces_dir=TRACES_DIR,
        labels_dir=LABELS_DIR,
        cross_validate=not args.no_cross_validate,
        max_concurrent=args.max_concurrent
    )

    total = sum(len(v) for v in results.values())
    print(f"\n✓ Labeled {total} traces")


async def phase_baselines(args):
    """Phase 4: Run baseline methods."""
    print("\n" + "="*60)
    print("PHASE 4: RUNNING BASELINE METHODS")
    print("="*60)

    # Log-only baseline
    print("\nRunning Log-Only baseline...")
    log_only_results = run_log_only_baseline(
        traces_dir=str(TRACES_DIR),
        output_dir=str(RESULTS_DIR / "log_only")
    )
    print(f"  ✓ Log-Only: {len(log_only_results)} results")

    # LLM-Direct baseline
    print("\nRunning LLM-Direct baseline...")
    llm_direct_results = await run_llm_direct_baseline(
        scenarios_dir=str(SCENARIOS_DIR),
        traces_dir=str(TRACES_DIR),
        output_dir=str(RESULTS_DIR / "llm_direct"),
        max_concurrent=args.max_concurrent
    )
    print(f"  ✓ LLM-Direct: {len(llm_direct_results)} results")

    print("\n✓ Baselines complete")


async def phase_agenttrace(args):
    """Phase 5: Run AgentTrace analysis."""
    print("\n" + "="*60)
    print("PHASE 5: RUNNING AGENTTRACE ANALYSIS")
    print("="*60)

    from agenttrace.core.graph import CausalGraph
    from agenttrace.query.interface import QueryInterface

    output_dir = RESULTS_DIR / "agenttrace"
    output_dir.mkdir(parents=True, exist_ok=True)

    trace_files = list(TRACES_DIR.glob("*_trace.json"))
    print(f"Processing {len(trace_files)} traces...")

    for trace_file in trace_files:
        with open(trace_file) as f:
            trace_data = json.load(f)

        scenario_id = trace_data["scenario_id"]
        trace_json = trace_data["trace_json"]

        # Load graph and run AgentTrace analysis
        graph = CausalGraph.from_json(trace_json)
        query = QueryInterface(graph)

        # Find errors and trace back
        error_nodes = [n for n in graph if n.type.value == "error"]

        analysis = {
            "scenario_id": scenario_id,
            "node_count": graph.node_count,
            "edge_count": graph.edge_count,
            "errors_found": len(error_nodes),
            "root_causes": [],
            "causal_chains": []
        }

        for error in error_nodes[:3]:  # Limit to first 3 errors
            root_causes = graph.find_root_causes(error.id)
            if root_causes:
                analysis["root_causes"].append({
                    "error_id": error.id,
                    "root_cause_ids": [rc.id for rc in root_causes],
                    "root_cause_data": [str(rc.data)[:100] for rc in root_causes]
                })

            # Get causal chain
            chain = graph.trace_backward(error.id)
            analysis["causal_chains"].append({
                "error_id": error.id,
                "chain_length": len(chain),
                "chain_node_ids": [n.id for n in chain[:10]]  # Limit
            })

        # Format output
        formatted = format_agenttrace_output(analysis, graph)
        analysis["formatted_output"] = formatted

        # Save
        output_file = output_dir / f"{scenario_id}_agenttrace.json"
        with open(output_file, 'w') as f:
            json.dump(analysis, f, indent=2)

    print(f"\n✓ AgentTrace analysis complete: {len(trace_files)} traces")


def format_agenttrace_output(analysis: dict, graph) -> str:
    """Format AgentTrace output for LLM judge."""
    lines = [
        f"=== AgentTrace Analysis for {analysis['scenario_id']} ===",
        "",
        f"Graph Statistics:",
        f"  Nodes: {analysis['node_count']}",
        f"  Edges: {analysis['edge_count']}",
        f"  Errors Found: {analysis['errors_found']}",
        "",
        "Root Cause Analysis:",
    ]

    for rc in analysis.get("root_causes", []):
        lines.append(f"  Error {rc['error_id'][:8]}... traced to:")
        for i, (rc_id, rc_data) in enumerate(zip(rc["root_cause_ids"], rc["root_cause_data"])):
            lines.append(f"    {i+1}. {rc_id[:8]}...: {rc_data}")

    lines.extend([
        "",
        "Causal Chains:",
    ])

    for chain in analysis.get("causal_chains", []):
        lines.append(f"  Error {chain['error_id'][:8]}...: {chain['chain_length']} nodes in chain")

    lines.extend([
        "",
        "Analysis Method: AgentTrace causal graph with backward tracing",
    ])

    return "\n".join(lines)


async def phase_evaluate(args):
    """Phase 6: Evaluate all methods."""
    print("\n" + "="*60)
    print("PHASE 6: EVALUATING METHODS")
    print("="*60)

    # Load ground truth labels
    ground_truths = {}
    for label_file in (LABELS_DIR / "gpt4").glob("*_labels.json"):
        with open(label_file) as f:
            data = json.load(f)
        ground_truths[data["scenario_id"]] = data

    print(f"Loaded {len(ground_truths)} ground truth labels")

    # Load method outputs
    methods = {
        "log_only": RESULTS_DIR / "log_only",
        "llm_direct": RESULTS_DIR / "llm_direct",
        "agenttrace": RESULTS_DIR / "agenttrace"
    }

    method_outputs = {}
    for method_name, method_dir in methods.items():
        method_outputs[method_name] = {}
        if method_dir.exists():
            for result_file in method_dir.glob("*.json"):
                with open(result_file) as f:
                    data = json.load(f)
                method_outputs[method_name][data["scenario_id"]] = data.get("formatted_output", "")

    # Run LLM-as-Judge comparisons
    if not args.skip_judge:
        print("\nRunning LLM-as-Judge comparisons...")

        # Load scenarios
        scenarios = []
        for domain in ScenarioDomain:
            domain_dir = SCENARIOS_DIR / domain.value
            if domain_dir.exists():
                for f in domain_dir.glob("*.json"):
                    with open(f) as file:
                        scenarios.append(json.load(file))

        # Compare AgentTrace vs Log-Only
        if "agenttrace" in method_outputs and "log_only" in method_outputs:
            judgments = await run_comparison_evaluation(
                scenarios=scenarios,
                outputs_a=method_outputs["agenttrace"],
                outputs_b=method_outputs["log_only"],
                ground_truths=ground_truths,
                method_a="AgentTrace",
                method_b="Log-Only",
                max_concurrent=args.max_concurrent
            )
            summary = summarize_comparisons(judgments)
            print(f"\n  AgentTrace vs Log-Only:")
            print(f"    AgentTrace wins: {summary.method_a_wins}")
            print(f"    Log-Only wins: {summary.method_b_wins}")
            print(f"    Ties: {summary.ties}")

        # Compare AgentTrace vs LLM-Direct
        if "agenttrace" in method_outputs and "llm_direct" in method_outputs:
            judgments = await run_comparison_evaluation(
                scenarios=scenarios,
                outputs_a=method_outputs["agenttrace"],
                outputs_b=method_outputs["llm_direct"],
                ground_truths=ground_truths,
                method_a="AgentTrace",
                method_b="LLM-Direct",
                max_concurrent=args.max_concurrent
            )
            summary = summarize_comparisons(judgments)
            print(f"\n  AgentTrace vs LLM-Direct:")
            print(f"    AgentTrace wins: {summary.method_a_wins}")
            print(f"    LLM-Direct wins: {summary.method_b_wins}")
            print(f"    Ties: {summary.ties}")

    print("\n✓ Evaluation complete")


async def run_all_phases(args):
    """Run all experiment phases."""
    start_time = datetime.now()

    if not args.skip_generate:
        await phase_generate(args)

    if not args.skip_collect:
        await phase_collect(args)

    if not args.skip_label:
        await phase_label(args)

    if not args.skip_baselines:
        await phase_baselines(args)

    if not args.skip_agenttrace:
        await phase_agenttrace(args)

    if not args.skip_evaluate:
        await phase_evaluate(args)

    end_time = datetime.now()
    duration = end_time - start_time

    print("\n" + "="*60)
    print("ALL EXPERIMENTS COMPLETE")
    print("="*60)
    print(f"Total time: {duration}")


def main():
    parser = argparse.ArgumentParser(
        description="Run AgentTrace benchmark experiments"
    )

    parser.add_argument(
        "--phase",
        choices=["all", "generate", "collect", "label", "baselines", "agenttrace", "evaluate"],
        default="all",
        help="Which phase to run"
    )

    parser.add_argument("--single-test", action="store_true",
                        help="Generate single test scenario")
    parser.add_argument("--count", type=int,
                        help="Scenarios per domain")
    parser.add_argument("--no-llm", action="store_true",
                        help="Skip LLM simulation in trace collection")
    parser.add_argument("--no-cross-validate", action="store_true",
                        help="Skip Claude cross-validation")
    parser.add_argument("--max-concurrent", type=int, default=5,
                        help="Max concurrent API calls")
    parser.add_argument("--skip-judge", action="store_true",
                        help="Skip LLM-as-Judge evaluation")

    # Skip flags for partial runs
    parser.add_argument("--skip-generate", action="store_true")
    parser.add_argument("--skip-collect", action="store_true")
    parser.add_argument("--skip-label", action="store_true")
    parser.add_argument("--skip-baselines", action="store_true")
    parser.add_argument("--skip-agenttrace", action="store_true")
    parser.add_argument("--skip-evaluate", action="store_true")

    args = parser.parse_args()

    async def run():
        if args.phase == "all":
            await run_all_phases(args)
        elif args.phase == "generate":
            await phase_generate(args)
        elif args.phase == "collect":
            await phase_collect(args)
        elif args.phase == "label":
            await phase_label(args)
        elif args.phase == "baselines":
            await phase_baselines(args)
        elif args.phase == "agenttrace":
            await phase_agenttrace(args)
        elif args.phase == "evaluate":
            await phase_evaluate(args)

    asyncio.run(run())


if __name__ == "__main__":
    main()
