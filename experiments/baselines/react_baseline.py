"""
ReAct-style Baseline for AgentTrace Benchmark.

This baseline uses the Reasoning + Acting pattern:
- Interleaves reasoning with information gathering
- Asks targeted questions about the trace
- Iteratively refines the root cause hypothesis
"""

import json
import asyncio
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

from openai import AsyncOpenAI


@dataclass
class ReActResult:
    """Result of ReAct analysis."""
    scenario_id: str
    thought_action_pairs: list[dict]
    final_answer: str
    identified_root_cause: str
    confidence: float
    raw_response: str
    model_used: str
    iterations: int


REACT_PROMPT = '''You are a debugger using the ReAct (Reasoning + Acting) approach to analyze a multi-agent system trace.

## Available Actions
- EXAMINE(node_id): Look at details of a specific node
- TRACE_BACKWARD(node_id): Find what caused this node
- TRACE_FORWARD(node_id): Find what this node affected
- CHECK_AGENT(agent_id): See all actions by an agent
- FIND_ERRORS: List all error nodes
- CONCLUDE: Make final determination

## Trace Data
{trace_data}

## Task
{task_description}

## Instructions
Use the ReAct pattern: alternate between Thought and Action.

Format each step as:
Thought: [Your reasoning about what to investigate next]
Action: [One of the available actions]
Observation: [What you learned]

Continue until you can CONCLUDE with the root cause.

## Output Format
Return a JSON object:
{{
    "steps": [
        {{"thought": "...", "action": "...", "observation": "..."}},
        ...
    ],
    "final_thought": "Based on my analysis...",
    "root_cause": "The root cause is...",
    "confidence": 0.0-1.0,
    "responsible_agent": "agent_id"
}}

Begin your ReAct analysis:'''


class ReActAnalyzer:
    """Analyzes traces using ReAct pattern."""

    def __init__(self, model: str = None):
        from experiments.config import OPENAI_MODELS
        self.client = AsyncOpenAI()
        self.model = model or OPENAI_MODELS.get("llm_direct_baseline", "gpt-4o")

    async def analyze(
        self,
        trace_json: str,
        scenario: dict,
        scenario_id: str
    ) -> ReActResult:
        """Analyze trace using ReAct pattern."""

        trace_data = trace_json[:8000] if len(trace_json) > 8000 else trace_json

        prompt = REACT_PROMPT.format(
            trace_data=trace_data,
            task_description=scenario.get("task_description", "Unknown task")
        )

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert debugger using the ReAct pattern. "
                               "Think step by step, take actions, and respond with valid JSON."
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            response_format={"type": "json_object"}
        )

        content = response.choices[0].message.content
        data = json.loads(content)

        return ReActResult(
            scenario_id=scenario_id,
            thought_action_pairs=data.get("steps", []),
            final_answer=data.get("final_thought", ""),
            identified_root_cause=data.get("root_cause", "Unknown"),
            confidence=data.get("confidence", 0.5),
            raw_response=content,
            model_used=self.model,
            iterations=len(data.get("steps", []))
        )

    def format_output(self, result: ReActResult) -> str:
        """Format result for evaluation."""
        lines = [
            f"=== ReAct Analysis for {result.scenario_id} ===",
            "",
            f"Iterations: {result.iterations}",
            "",
            "Reasoning Trace:",
        ]

        for i, step in enumerate(result.thought_action_pairs[:5]):
            lines.append(f"  [{i+1}] Thought: {step.get('thought', '')[:80]}...")
            lines.append(f"      Action: {step.get('action', '')}")

        lines.extend([
            "",
            f"Final Analysis: {result.final_answer[:200]}...",
            "",
            f"Root Cause: {result.identified_root_cause}",
            f"Confidence: {result.confidence:.0%}",
            "",
            f"Analysis Method: ReAct with {result.model_used}",
        ])

        return "\n".join(lines)


async def run_react_baseline(
    scenarios_dir: str = "data/scenarios",
    traces_dir: str = "data/traces",
    output_dir: str = "data/results/react",
    max_concurrent: int = 10
) -> dict[str, ReActResult]:
    """Run ReAct baseline on all traces."""
    from experiments.benchmark.scenario_generator import ScenarioDomain

    scenarios_path = Path(scenarios_dir)
    traces_path = Path(traces_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    analyzer = ReActAnalyzer()
    results = {}
    semaphore = asyncio.Semaphore(max_concurrent)

    trace_files = list(traces_path.glob("*_trace.json"))

    async def process_trace(trace_file: Path) -> Optional[tuple[str, ReActResult]]:
        scenario_id = trace_file.stem.replace("_trace", "")
        output_file = output_path / f"{scenario_id}_react.json"

        if output_file.exists():
            with open(output_file) as f:
                data = json.load(f)
            return scenario_id, ReActResult(
                scenario_id=data['scenario_id'],
                thought_action_pairs=data.get('thought_action_pairs', []),
                final_answer=data.get('final_answer', ''),
                identified_root_cause=data['identified_root_cause'],
                confidence=data.get('confidence', 0.5),
                raw_response="",
                model_used=data.get('model_used', 'gpt-4o'),
                iterations=data.get('iterations', 0)
            )

        async with semaphore:
            with open(trace_file) as f:
                trace_data = json.load(f)

            scenario_id = trace_data["scenario_id"]
            trace_json = trace_data["trace_json"]

            scenario = None
            for domain in ScenarioDomain:
                scenario_file = scenarios_path / domain.value / f"{scenario_id}.json"
                if scenario_file.exists():
                    with open(scenario_file) as f:
                        scenario = json.load(f)
                    break

            if not scenario:
                return None

            try:
                result = await analyzer.analyze(trace_json, scenario, scenario_id)

                with open(output_file, 'w') as f:
                    json.dump({
                        'scenario_id': result.scenario_id,
                        'thought_action_pairs': result.thought_action_pairs,
                        'final_answer': result.final_answer,
                        'identified_root_cause': result.identified_root_cause,
                        'confidence': result.confidence,
                        'model_used': result.model_used,
                        'iterations': result.iterations,
                        'formatted_output': analyzer.format_output(result)
                    }, f, indent=2)

                print(f"  Analyzed {scenario_id}")
                return scenario_id, result
            except Exception as e:
                print(f"  Error analyzing {scenario_id}: {e}")
                return None

    tasks = [process_trace(f) for f in trace_files]
    task_results = await asyncio.gather(*tasks, return_exceptions=True)

    for res in task_results:
        if isinstance(res, tuple) and res is not None:
            scenario_id, result = res
            results[scenario_id] = result

    return results


if __name__ == "__main__":
    async def main():
        print("Running ReAct Baseline...")
        results = await run_react_baseline()
        print(f"Analyzed {len(results)} traces with ReAct baseline")

    asyncio.run(main())
