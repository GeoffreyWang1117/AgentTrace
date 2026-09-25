"""
LLM-Direct Baseline for AgentTrace Benchmark.

This baseline represents using LLM directly on raw logs/traces:
- No structured causal graph
- LLM analyzes raw trace data to identify root cause
- Represents "just ask GPT-4" approach
"""

import json
import asyncio
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

from openai import AsyncOpenAI


@dataclass
class LLMDirectResult:
    """Result of LLM-direct analysis."""
    scenario_id: str

    # LLM's analysis
    identified_root_cause: str
    root_cause_explanation: str
    suggested_causal_chain: list[str]

    # Raw LLM response
    raw_response: str

    # Metadata
    model_used: str
    prompt_tokens: int
    completion_tokens: int


LLM_DIRECT_PROMPT = '''You are an expert debugger analyzing a multi-agent system execution trace.

## Task Context
{task_description}

## Graph Structure (Adjacency List)
{adjacency_list}

## Node Summaries
{node_summaries}

## Raw Execution Trace
{trace_data}

## Your Task
Analyze this trace step by step using chain-of-thought reasoning:

1. First, identify the error manifestation — which node shows the final failure?
2. Trace backward through the adjacency list to find predecessor nodes.
3. For each predecessor, examine whether its content could have caused the downstream error.
4. Identify the root cause — the earliest node whose incorrect behavior propagated to the error.
5. Describe the full causal chain from root cause to error.

Think carefully before answering. Consider:
- Which agents interacted and what data was passed?
- Where did the data first become incorrect?
- Is the bug a data corruption, missing context, wrong routing, or logic error?

## Output Format
Return a JSON object:
{{
    "reasoning": "Your step-by-step chain-of-thought reasoning",
    "root_cause": "Description of the root cause",
    "root_cause_node_id": "The node ID of the root cause",
    "root_cause_explanation": "Detailed explanation of why this is the root cause",
    "causal_chain": ["step1 description", "step2 description", ...],
    "causal_chain_node_ids": ["node_id_1", "node_id_2", ...],
    "responsible_agents": ["agent1", "agent2"]
}}

Analyze the trace carefully:'''


class LLMDirectAnalyzer:
    """
    Analyzes traces by directly prompting an LLM.

    This baseline shows the limitations of using LLM without
    structured causal graph representation.
    """

    def __init__(self, model: str = "gpt-4-turbo-preview"):
        self.client = AsyncOpenAI()
        self.model = model

    async def analyze(
        self,
        trace_json: str,
        scenario: dict,
        scenario_id: str
    ) -> LLMDirectResult:
        """Analyze a trace using direct LLM prompting with structured context."""

        # Parse trace to build adjacency list and node summaries
        try:
            trace_obj = json.loads(trace_json) if isinstance(trace_json, str) else trace_json
            nodes = trace_obj.get("nodes", [])
            edges = trace_obj.get("edges", [])

            # Build adjacency list
            adj_lines = []
            for edge in edges:
                adj_lines.append(
                    f"  {edge['source_id'][:12]}... -> {edge['target_id'][:12]}... "
                    f"[{edge.get('type', 'unknown')}]"
                )
            adjacency_list = "\n".join(adj_lines[:50]) or "(no edges)"

            # Build node summaries with predecessors/successors
            successors = {}
            predecessors = {}
            for edge in edges:
                successors.setdefault(edge['source_id'], []).append(edge['target_id'])
                predecessors.setdefault(edge['target_id'], []).append(edge['source_id'])

            summary_lines = []
            for node in nodes:
                nid = node['id']
                data = node.get('data', {})
                preds = [p[:12] + "..." for p in predecessors.get(nid, [])]
                succs = [s[:12] + "..." for s in successors.get(nid, [])]
                summary_lines.append(
                    f"  Node {nid[:12]}... | Agent: {node.get('agent_id', '?')} | "
                    f"Type: {node.get('type', '?')} | "
                    f"Action: {data.get('action', '?')} | "
                    f"Predecessors: {preds} | Successors: {succs}"
                )
            node_summaries = "\n".join(summary_lines[:40]) or "(no nodes)"
        except (json.JSONDecodeError, TypeError):
            adjacency_list = "(parse error)"
            node_summaries = "(parse error)"

        # Format trace data (truncate if too long)
        trace_data = trace_json[:6000] if len(trace_json) > 6000 else trace_json

        prompt = LLM_DIRECT_PROMPT.format(
            task_description=scenario.get("task_description", "Unknown task"),
            adjacency_list=adjacency_list,
            node_summaries=node_summaries,
            trace_data=trace_data,
        )

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert in debugging multi-agent systems. "
                               "Analyze traces and identify root causes. "
                               "Always respond with valid JSON."
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            response_format={"type": "json_object"}
        )

        content = response.choices[0].message.content
        data = json.loads(content)

        return LLMDirectResult(
            scenario_id=scenario_id,
            identified_root_cause=data.get("root_cause", "Unknown"),
            root_cause_explanation=data.get("root_cause_explanation", ""),
            suggested_causal_chain=data.get("causal_chain", []),
            raw_response=content,
            model_used=self.model,
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens
        )

    def format_output(self, result: LLMDirectResult) -> str:
        """Format result for LLM judge evaluation."""
        lines = [
            f"=== LLM-Direct Analysis for {result.scenario_id} ===",
            "",
            f"Model: {result.model_used}",
            "",
            f"Identified Root Cause: {result.identified_root_cause}",
            "",
            "Explanation:",
            result.root_cause_explanation,
            "",
            "Suggested Causal Chain:",
        ]

        for i, step in enumerate(result.suggested_causal_chain):
            lines.append(f"  {i+1}. {step}")

        lines.extend([
            "",
            "Analysis Method: Direct LLM prompting on raw trace",
        ])

        return "\n".join(lines)


async def run_llm_direct_baseline(
    scenarios_dir: str = "data/scenarios",
    traces_dir: str = "data/traces",
    output_dir: str = "data/results/llm_direct",
    max_concurrent: int = 5
) -> dict[str, LLMDirectResult]:
    """Run LLM-direct baseline on all traces."""
    from experiments.benchmark.scenario_generator import Scenario, ScenarioDomain

    scenarios_path = Path(scenarios_dir)
    traces_path = Path(traces_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    analyzer = LLMDirectAnalyzer()
    results = {}
    semaphore = asyncio.Semaphore(max_concurrent)

    # Collect all trace files
    trace_files = list(traces_path.glob("*_trace.json"))

    async def process_trace(trace_file: Path) -> Optional[tuple[str, LLMDirectResult]]:
        async with semaphore:
            with open(trace_file) as f:
                trace_data = json.load(f)

            scenario_id = trace_data["scenario_id"]
            trace_json = trace_data["trace_json"]

            # Find corresponding scenario
            scenario = None
            for domain in ScenarioDomain:
                scenario_file = scenarios_path / domain.value / f"{scenario_id}.json"
                if scenario_file.exists():
                    with open(scenario_file) as f:
                        scenario = json.load(f)
                    break

            if not scenario:
                print(f"Scenario not found for {scenario_id}")
                return None

            print(f"  Analyzing {scenario_id}...")
            result = await analyzer.analyze(trace_json, scenario, scenario_id)

            # Save result
            output_file = output_path / f"{scenario_id}_llm_direct.json"
            with open(output_file, 'w') as f:
                json.dump({
                    "scenario_id": result.scenario_id,
                    "identified_root_cause": result.identified_root_cause,
                    "root_cause_explanation": result.root_cause_explanation,
                    "suggested_causal_chain": result.suggested_causal_chain,
                    "model_used": result.model_used,
                    "formatted_output": analyzer.format_output(result)
                }, f, indent=2)

            return scenario_id, result

    tasks = [process_trace(f) for f in trace_files]
    task_results = await asyncio.gather(*tasks, return_exceptions=True)

    for res in task_results:
        if isinstance(res, tuple):
            scenario_id, result = res
            results[scenario_id] = result
        elif isinstance(res, Exception):
            print(f"Error: {res}")

    return results


if __name__ == "__main__":
    import asyncio

    async def main():
        results = await run_llm_direct_baseline()
        print(f"Analyzed {len(results)} traces with LLM-direct baseline")

    asyncio.run(main())
