"""
LLM-as-Judge Evaluator for AgentTrace Benchmark.

This module uses GPT-4 as a judge to compare debugging approaches
and evaluate the quality of causal analysis.
"""

import json
import asyncio
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Literal, Optional
from datetime import datetime

from openai import AsyncOpenAI


@dataclass
class JudgmentCriteria:
    """Scores for each evaluation criterion."""
    correctness: int      # 1-5: Did it identify correct root cause?
    completeness: int     # 1-5: Did it capture full causal chain?
    clarity: int          # 1-5: How clear is the explanation?
    efficiency: int       # 1-5: How direct was the path?


@dataclass
class ComparisonJudgment:
    """Result of comparing two debugging approaches."""
    scenario_id: str
    method_a: str
    method_b: str

    scores_a: JudgmentCriteria
    scores_b: JudgmentCriteria

    winner: Literal["A", "B", "tie"]
    explanation: str

    judge_model: str
    judged_at: str


@dataclass
class SingleMethodJudgment:
    """Judgment of a single method's output."""
    scenario_id: str
    method_name: str

    scores: JudgmentCriteria
    overall_score: float  # Average of criteria

    strengths: list[str]
    weaknesses: list[str]
    explanation: str

    judge_model: str
    judged_at: str


COMPARISON_PROMPT = '''You are an expert judge evaluating debugging tools for multi-agent systems.

## Scenario
Task: {task_description}
Expected Outcome: {expected_outcome}
Bug Type: {bug_type}
Ground Truth Root Cause: {ground_truth_root_cause}

## Debugging Output A ({method_a})
{output_a}

## Debugging Output B ({method_b})
{output_b}

## Evaluation Criteria
Rate each approach on a scale of 1-5 for:

1. **Correctness** (1-5): Did it identify the correct root cause?
   - 5: Exactly correct
   - 3: Partially correct or close
   - 1: Completely wrong

2. **Completeness** (1-5): Did it capture the full causal chain?
   - 5: All causal relationships identified
   - 3: Most relationships found
   - 1: Missing critical relationships

3. **Clarity** (1-5): How clear is the explanation?
   - 5: Crystal clear, easy to understand
   - 3: Understandable but could be clearer
   - 1: Confusing or unclear

4. **Efficiency** (1-5): How direct was the path to root cause?
   - 5: Direct path, no unnecessary steps
   - 3: Some detours but found the answer
   - 1: Very roundabout or failed to find

## Output Format
Return a JSON object:
{{
    "scores_a": {{
        "correctness": 1-5,
        "completeness": 1-5,
        "clarity": 1-5,
        "efficiency": 1-5
    }},
    "scores_b": {{
        "correctness": 1-5,
        "completeness": 1-5,
        "clarity": 1-5,
        "efficiency": 1-5
    }},
    "winner": "A" or "B" or "tie",
    "explanation": "Detailed explanation of your judgment"
}}

Evaluate fairly and objectively:'''


SINGLE_METHOD_PROMPT = '''You are an expert judge evaluating a debugging tool for multi-agent systems.

## Scenario
Task: {task_description}
Expected Outcome: {expected_outcome}
Bug Type: {bug_type}
Ground Truth Root Cause: {ground_truth_root_cause}
Ground Truth Causal Chain: {ground_truth_chain}

## Debugging Output ({method_name})
{output}

## Evaluation Criteria
Rate the output on a scale of 1-5 for:

1. **Correctness** (1-5): Did it identify the correct root cause?
2. **Completeness** (1-5): Did it capture the full causal chain?
3. **Clarity** (1-5): How clear is the explanation?
4. **Efficiency** (1-5): How direct was the path to root cause?

## Output Format
Return a JSON object:
{{
    "scores": {{
        "correctness": 1-5,
        "completeness": 1-5,
        "clarity": 1-5,
        "efficiency": 1-5
    }},
    "strengths": ["strength 1", "strength 2"],
    "weaknesses": ["weakness 1", "weakness 2"],
    "explanation": "Detailed explanation of your judgment"
}}

Evaluate objectively:'''


class LLMJudge:
    """Uses GPT-4 to judge debugging outputs."""

    def __init__(
        self,
        model: str = "gpt-4-turbo-preview",
        output_dir: Path = Path("data/results/judgments")
    ):
        self.client = AsyncOpenAI()
        self.model = model
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    async def compare_methods(
        self,
        scenario: dict,
        output_a: str,
        output_b: str,
        method_a: str,
        method_b: str,
        ground_truth: dict
    ) -> ComparisonJudgment:
        """Compare two debugging outputs using LLM as judge."""

        prompt = COMPARISON_PROMPT.format(
            task_description=scenario.get("task_description", "Unknown"),
            expected_outcome=scenario.get("expected_outcome", "Unknown"),
            bug_type=scenario.get("bug_type", "Unknown"),
            ground_truth_root_cause=ground_truth.get("root_cause_explanation", "Unknown"),
            method_a=method_a,
            output_a=output_a,
            method_b=method_b,
            output_b=output_b
        )

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": "You are an impartial judge evaluating debugging tools. "
                               "Always respond with valid JSON."
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            response_format={"type": "json_object"}
        )

        content = response.choices[0].message.content
        data = json.loads(content)

        return ComparisonJudgment(
            scenario_id=scenario.get("scenario_id", "unknown"),
            method_a=method_a,
            method_b=method_b,
            scores_a=JudgmentCriteria(**data["scores_a"]),
            scores_b=JudgmentCriteria(**data["scores_b"]),
            winner=data["winner"],
            explanation=data["explanation"],
            judge_model=self.model,
            judged_at=datetime.now().isoformat()
        )

    async def judge_single_method(
        self,
        scenario: dict,
        output: str,
        method_name: str,
        ground_truth: dict
    ) -> SingleMethodJudgment:
        """Judge a single method's output."""

        # Format causal chain for prompt
        causal_chain = ground_truth.get("causal_edges", [])
        chain_str = "\n".join([
            f"  {e.get('source_node_id', '?')} -> {e.get('target_node_id', '?')}: {e.get('explanation', '')}"
            for e in causal_chain[:10]  # Limit for context
        ])

        prompt = SINGLE_METHOD_PROMPT.format(
            task_description=scenario.get("task_description", "Unknown"),
            expected_outcome=scenario.get("expected_outcome", "Unknown"),
            bug_type=scenario.get("bug_type", "Unknown"),
            ground_truth_root_cause=ground_truth.get("root_cause_explanation", "Unknown"),
            ground_truth_chain=chain_str or "Not available",
            method_name=method_name,
            output=output
        )

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": "You are an impartial judge evaluating debugging tools. "
                               "Always respond with valid JSON."
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            response_format={"type": "json_object"}
        )

        content = response.choices[0].message.content
        data = json.loads(content)

        scores = JudgmentCriteria(**data["scores"])
        overall = (scores.correctness + scores.completeness +
                   scores.clarity + scores.efficiency) / 4.0

        return SingleMethodJudgment(
            scenario_id=scenario.get("scenario_id", "unknown"),
            method_name=method_name,
            scores=scores,
            overall_score=overall,
            strengths=data.get("strengths", []),
            weaknesses=data.get("weaknesses", []),
            explanation=data["explanation"],
            judge_model=self.model,
            judged_at=datetime.now().isoformat()
        )

    def save_judgment(self, judgment: ComparisonJudgment | SingleMethodJudgment) -> Path:
        """Save judgment to disk."""
        if isinstance(judgment, ComparisonJudgment):
            filename = f"{judgment.scenario_id}_comparison.json"
        else:
            filename = f"{judgment.scenario_id}_{judgment.method_name}_judgment.json"

        filepath = self.output_dir / filename
        with open(filepath, 'w') as f:
            json.dump(asdict(judgment), f, indent=2)

        return filepath


@dataclass
class JudgmentSummary:
    """Summary statistics for judgments."""
    total_comparisons: int
    method_a_wins: int
    method_b_wins: int
    ties: int

    avg_scores_a: dict[str, float]
    avg_scores_b: dict[str, float]

    win_rate_a: float
    win_rate_b: float


def summarize_comparisons(judgments: list[ComparisonJudgment]) -> JudgmentSummary:
    """Summarize comparison judgments."""
    if not judgments:
        return JudgmentSummary(0, 0, 0, 0, {}, {}, 0, 0)

    a_wins = sum(1 for j in judgments if j.winner == "A")
    b_wins = sum(1 for j in judgments if j.winner == "B")
    ties = sum(1 for j in judgments if j.winner == "tie")

    # Average scores
    criteria = ["correctness", "completeness", "clarity", "efficiency"]

    avg_a = {
        c: sum(getattr(j.scores_a, c) for j in judgments) / len(judgments)
        for c in criteria
    }
    avg_b = {
        c: sum(getattr(j.scores_b, c) for j in judgments) / len(judgments)
        for c in criteria
    }

    total = len(judgments)
    return JudgmentSummary(
        total_comparisons=total,
        method_a_wins=a_wins,
        method_b_wins=b_wins,
        ties=ties,
        avg_scores_a=avg_a,
        avg_scores_b=avg_b,
        win_rate_a=a_wins / total,
        win_rate_b=b_wins / total
    )


async def run_comparison_evaluation(
    scenarios: list[dict],
    outputs_a: dict[str, str],  # scenario_id -> output
    outputs_b: dict[str, str],
    ground_truths: dict[str, dict],
    method_a: str,
    method_b: str,
    max_concurrent: int = 5
) -> list[ComparisonJudgment]:
    """Run comparison evaluation for all scenarios."""
    judge = LLMJudge()
    semaphore = asyncio.Semaphore(max_concurrent)

    async def judge_one(scenario: dict) -> Optional[ComparisonJudgment]:
        async with semaphore:
            sid = scenario.get("scenario_id")
            if sid not in outputs_a or sid not in outputs_b:
                return None

            judgment = await judge.compare_methods(
                scenario=scenario,
                output_a=outputs_a[sid],
                output_b=outputs_b[sid],
                method_a=method_a,
                method_b=method_b,
                ground_truth=ground_truths.get(sid, {})
            )
            judge.save_judgment(judgment)
            return judgment

    results = await asyncio.gather(
        *[judge_one(s) for s in scenarios],
        return_exceptions=True
    )

    return [r for r in results if isinstance(r, ComparisonJudgment)]
