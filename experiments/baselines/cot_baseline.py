"""
Chain-of-Thought (CoT) Baseline for AgentTrace Benchmark.

This baseline uses step-by-step reasoning to analyze traces:
- Breaks down the trace analysis into steps
- Uses structured prompting for root cause identification
- Represents a more sophisticated LLM baseline than direct prompting
"""

import json
import asyncio
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

from openai import AsyncOpenAI


@dataclass
class CoTResult:
    """Result of CoT analysis."""
    scenario_id: str
    reasoning_steps: list[str]
    identified_root_cause: str
    confidence: float
    causal_chain: list[str]
    raw_response: str
    model_used: str


COT_PROMPT = '''You are an expert debugger analyzing a multi-agent system trace. Use step-by-step reasoning to identify the root cause.

## Trace Data
{trace_data}

## Task Description
{task_description}

## Analysis Steps
Follow these steps carefully:

**Step 1: Identify all agents involved**
List each agent and their role in the trace.

**Step 2: Trace the data flow**
For each piece of data, track how it flows between agents.

**Step 3: Identify error points**
Find where things went wrong - look for errors, unexpected outputs, or failures.

**Step 4: Trace backward from error**
Starting from the error, work backward to find what caused it.

**Step 5: Identify root cause**
Determine the earliest point where things went wrong.

**Step 6: Construct causal chain**
List the sequence of events from root cause to observed error.

## Output Format
Return a JSON object:
{{
    "reasoning_steps": [
        "Step 1: [your analysis]",
        "Step 2: [your analysis]",
        ...
    ],
    "root_cause": "Description of the root cause",
    "confidence": 0.0-1.0,
    "causal_chain": ["event1", "event2", "event3"],
    "responsible_agent": "agent_id"
}}

Think step by step:'''


class CoTAnalyzer:
    """Analyzes traces using Chain-of-Thought reasoning."""

    def __init__(self, model: str = None):
        from experiments.config import OPENAI_MODELS
        self.client = AsyncOpenAI()
        self.model = model or OPENAI_MODELS.get("llm_direct_baseline", "gpt-4o")

    async def analyze(
        self,
        trace_json: str,
        scenario: dict,
        scenario_id: str
    ) -> CoTResult:
        """Analyze trace using CoT reasoning."""

        # Truncate if needed
        trace_data = trace_json[:8000] if len(trace_json) > 8000 else trace_json

        prompt = COT_PROMPT.format(
            trace_data=trace_data,
            task_description=scenario.get("task_description", "Unknown task")
        )

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert debugger. Think step by step and respond with valid JSON."
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            response_format={"type": "json_object"}
        )

        content = response.choices[0].message.content
        data = json.loads(content)

        return CoTResult(
            scenario_id=scenario_id,
            reasoning_steps=data.get("reasoning_steps", []),
            identified_root_cause=data.get("root_cause", "Unknown"),
            confidence=data.get("confidence", 0.5),
            causal_chain=data.get("causal_chain", []),
            raw_response=content,
            model_used=self.model
        )

    def format_output(self, result: CoTResult) -> str:
        """Format result for evaluation."""
        lines = [
            f"=== Chain-of-Thought Analysis for {result.scenario_id} ===",
            "",
            "Reasoning Steps:",
        ]

        for i, step in enumerate(result.reasoning_steps[:6]):
            lines.append(f"  {step}")

        lines.extend([
            "",
            f"Root Cause: {result.identified_root_cause}",
            f"Confidence: {result.confidence:.0%}",
            "",
            "Causal Chain:",
        ])

        for i, event in enumerate(result.causal_chain):
            lines.append(f"  {i+1}. {event}")

        lines.extend([
            "",
            f"Analysis Method: Chain-of-Thought with {result.model_used}",
        ])

        return "\n".join(lines)


async def run_cot_baseline(
    scenarios_dir: str = "data/scenarios",
    traces_dir: str = "data/traces",
    output_dir: str = "data/results/cot",
    max_concurrent: int = 10
) -> dict[str, CoTResult]:
    """Run CoT baseline on all traces."""
    from experiments.benchmark.scenario_generator import ScenarioDomain

    scenarios_path = Path(scenarios_dir)
    traces_path = Path(traces_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    analyzer = CoTAnalyzer()
    results = {}
    semaphore = asyncio.Semaphore(max_concurrent)

    # Collect all trace files
    trace_files = list(traces_path.glob("*_trace.json"))

    async def process_trace(trace_file: Path) -> Optional[tuple[str, CoTResult]]:
        # Check if already processed
        scenario_id = trace_file.stem.replace("_trace", "")
        output_file = output_path / f"{scenario_id}_cot.json"

        if output_file.exists():
            with open(output_file) as f:
                data = json.load(f)
            return scenario_id, CoTResult(
                scenario_id=data['scenario_id'],
                reasoning_steps=data.get('reasoning_steps', []),
                identified_root_cause=data['identified_root_cause'],
                confidence=data.get('confidence', 0.5),
                causal_chain=data.get('causal_chain', []),
                raw_response="",
                model_used=data.get('model_used', 'gpt-4o')
            )

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
                return None

            try:
                result = await analyzer.analyze(trace_json, scenario, scenario_id)

                # Save result
                with open(output_file, 'w') as f:
                    json.dump({
                        'scenario_id': result.scenario_id,
                        'reasoning_steps': result.reasoning_steps,
                        'identified_root_cause': result.identified_root_cause,
                        'confidence': result.confidence,
                        'causal_chain': result.causal_chain,
                        'model_used': result.model_used,
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
        print("Running Chain-of-Thought Baseline...")
        results = await run_cot_baseline()
        print(f"Analyzed {len(results)} traces with CoT baseline")

    asyncio.run(main())
