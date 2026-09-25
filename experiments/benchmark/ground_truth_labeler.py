"""
Ground Truth Labeler for AgentTrace Benchmark.

This module uses LLMs (GPT-4 and Claude) to generate and validate
ground truth causal relationship labels for evaluation.
"""

import json
import asyncio
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional, Literal
from datetime import datetime

from openai import AsyncOpenAI

try:
    from anthropic import AsyncAnthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

from agenttrace.core.graph import CausalGraph

from .scenario_generator import Scenario, CausalEdge
from .trace_collector import TraceResult


@dataclass
class CausalLabel:
    """A labeled causal relationship."""
    source_node_id: str
    target_node_id: str
    relationship_type: Literal["data_flow", "trigger", "state_dependency", "temporal"]
    confidence: float
    explanation: str


@dataclass
class GroundTruthLabel:
    """Complete ground truth labels for a trace."""
    scenario_id: str
    trace_id: str

    # Causal relationships
    causal_edges: list[CausalLabel]

    # Root cause analysis
    root_cause_node_id: str
    root_cause_explanation: str

    # Error propagation path
    error_propagation_path: list[str]  # Node IDs in order

    # Labeling metadata
    labeler_model: str
    labeling_timestamp: str
    raw_llm_response: Optional[str] = None


@dataclass
class CrossValidationResult:
    """Result of cross-validation between two labelers."""
    scenario_id: str
    gpt4_labels: GroundTruthLabel
    claude_labels: Optional[GroundTruthLabel]

    # Agreement metrics
    edge_agreement_rate: float  # Percentage of edges both agree on
    root_cause_agreement: bool
    path_similarity: float  # Jaccard similarity of paths

    # Cohen's Kappa for edge labels
    cohens_kappa: float


LABELING_PROMPT = '''You are an expert in debugging multi-agent systems. Analyze this execution trace and identify all causal relationships.

## Scenario Context
Task: {task_description}
Expected Outcome: {expected_outcome}
Bug Type: {bug_type}

## Execution Trace
{trace_json}

## Node Summary
{node_summary}

## Your Task
Identify all causal relationships in this trace. For each relationship, specify:
1. source_node_id: The node that causes the effect
2. target_node_id: The node that is affected
3. relationship_type: One of "data_flow", "trigger", "state_dependency", "temporal"
4. confidence: Your confidence in this relationship (0.0 to 1.0)
5. explanation: Brief explanation of why this is causal

Also identify:
1. The root cause node (the earliest node that initiated the error chain)
2. The error propagation path (sequence of node IDs from root cause to error manifestation)

## Output Format
Return a JSON object:
{{
    "causal_edges": [
        {{
            "source_node_id": "node_xxx",
            "target_node_id": "node_yyy",
            "relationship_type": "data_flow",
            "confidence": 0.95,
            "explanation": "Data from node_xxx is used as input to node_yyy"
        }}
    ],
    "root_cause_node_id": "node_xxx",
    "root_cause_explanation": "This node introduced the error because...",
    "error_propagation_path": ["node_xxx", "node_yyy", "node_zzz"]
}}

Analyze carefully and identify ALL causal relationships:'''


class GroundTruthLabeler:
    """Generates ground truth labels using LLMs."""

    def __init__(
        self,
        primary_model: str = None,
        output_dir: Path = Path("data/labels")
    ):
        from experiments.config import OPENAI_MODELS
        self.primary_model = primary_model or OPENAI_MODELS.get("ground_truth_labeling", "gpt-4o")
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.openai_client = AsyncOpenAI()
        self.anthropic_client = AsyncAnthropic() if ANTHROPIC_AVAILABLE else None

    def _summarize_nodes(self, graph: CausalGraph) -> str:
        """Create a summary of nodes for the LLM."""
        lines = []
        for node in sorted(graph, key=lambda n: n.timestamp):
            lines.append(
                f"- {node.id}: type={node.type.value}, agent={node.agent_id}, "
                f"data={json.dumps(node.data)[:100]}..."
            )
        return "\n".join(lines)

    async def label_with_gpt4(
        self,
        scenario: Scenario,
        trace_result: TraceResult
    ) -> GroundTruthLabel:
        """Generate labels using GPT-4."""
        graph = CausalGraph.from_json(trace_result.trace_json)
        node_summary = self._summarize_nodes(graph)

        prompt = LABELING_PROMPT.format(
            task_description=scenario.task_description,
            expected_outcome=scenario.expected_outcome,
            bug_type=scenario.bug_type.value,
            trace_json=trace_result.trace_json[:5000],  # Truncate if too long
            node_summary=node_summary
        )

        response = await self.openai_client.chat.completions.create(
            model=self.primary_model,
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert in causal analysis of multi-agent systems. "
                               "Always respond with valid JSON."
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            response_format={"type": "json_object"}
        )

        content = response.choices[0].message.content
        data = json.loads(content)

        return GroundTruthLabel(
            scenario_id=scenario.scenario_id,
            trace_id=trace_result.scenario_id,
            causal_edges=[CausalLabel(**e) for e in data["causal_edges"]],
            root_cause_node_id=data["root_cause_node_id"],
            root_cause_explanation=data["root_cause_explanation"],
            error_propagation_path=data["error_propagation_path"],
            labeler_model=self.primary_model,
            labeling_timestamp=datetime.now().isoformat(),
            raw_llm_response=content
        )

    async def label_with_claude(
        self,
        scenario: Scenario,
        trace_result: TraceResult,
        model: str = "claude-3-opus-20240229"
    ) -> Optional[GroundTruthLabel]:
        """Generate labels using Claude for cross-validation."""
        if not self.anthropic_client:
            print("Anthropic client not available, skipping Claude labeling")
            return None

        graph = CausalGraph.from_json(trace_result.trace_json)
        node_summary = self._summarize_nodes(graph)

        prompt = LABELING_PROMPT.format(
            task_description=scenario.task_description,
            expected_outcome=scenario.expected_outcome,
            bug_type=scenario.bug_type.value,
            trace_json=trace_result.trace_json[:5000],
            node_summary=node_summary
        )

        response = await self.anthropic_client.messages.create(
            model=model,
            max_tokens=4096,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )

        content = response.content[0].text

        # Extract JSON from response (Claude might include explanation text)
        try:
            # Try to find JSON in the response
            import re
            json_match = re.search(r'\{[\s\S]*\}', content)
            if json_match:
                data = json.loads(json_match.group())
            else:
                data = json.loads(content)
        except json.JSONDecodeError:
            print(f"Failed to parse Claude response as JSON")
            return None

        return GroundTruthLabel(
            scenario_id=scenario.scenario_id,
            trace_id=trace_result.scenario_id,
            causal_edges=[CausalLabel(**e) for e in data["causal_edges"]],
            root_cause_node_id=data["root_cause_node_id"],
            root_cause_explanation=data["root_cause_explanation"],
            error_propagation_path=data["error_propagation_path"],
            labeler_model=model,
            labeling_timestamp=datetime.now().isoformat(),
            raw_llm_response=content
        )

    def compute_agreement(
        self,
        gpt4_labels: GroundTruthLabel,
        claude_labels: GroundTruthLabel
    ) -> CrossValidationResult:
        """Compute agreement metrics between two sets of labels."""

        # Edge agreement
        gpt4_edges = {
            (e.source_node_id, e.target_node_id, e.relationship_type)
            for e in gpt4_labels.causal_edges
        }
        claude_edges = {
            (e.source_node_id, e.target_node_id, e.relationship_type)
            for e in claude_labels.causal_edges
        }

        intersection = gpt4_edges & claude_edges
        union = gpt4_edges | claude_edges

        edge_agreement = len(intersection) / len(union) if union else 1.0

        # Root cause agreement
        root_cause_agreement = (
            gpt4_labels.root_cause_node_id == claude_labels.root_cause_node_id
        )

        # Path similarity (Jaccard)
        gpt4_path = set(gpt4_labels.error_propagation_path)
        claude_path = set(claude_labels.error_propagation_path)
        path_intersection = gpt4_path & claude_path
        path_union = gpt4_path | claude_path
        path_similarity = len(path_intersection) / len(path_union) if path_union else 1.0

        # Cohen's Kappa (simplified)
        # For proper Kappa, we'd need to consider all possible node pairs
        # This is a simplified version based on agreement rate
        po = edge_agreement  # Observed agreement
        pe = 0.25  # Expected agreement by chance (assuming 4 relationship types)
        cohens_kappa = (po - pe) / (1 - pe) if po > pe else 0.0

        return CrossValidationResult(
            scenario_id=gpt4_labels.scenario_id,
            gpt4_labels=gpt4_labels,
            claude_labels=claude_labels,
            edge_agreement_rate=edge_agreement,
            root_cause_agreement=root_cause_agreement,
            path_similarity=path_similarity,
            cohens_kappa=cohens_kappa
        )

    async def label_with_cross_validation(
        self,
        scenario: Scenario,
        trace_result: TraceResult
    ) -> CrossValidationResult:
        """Label with both GPT-4 and Claude, then compute agreement."""
        gpt4_labels = await self.label_with_gpt4(scenario, trace_result)
        claude_labels = await self.label_with_claude(scenario, trace_result)

        if claude_labels:
            return self.compute_agreement(gpt4_labels, claude_labels)
        else:
            # Return with only GPT-4 labels
            return CrossValidationResult(
                scenario_id=scenario.scenario_id,
                gpt4_labels=gpt4_labels,
                claude_labels=None,
                edge_agreement_rate=1.0,
                root_cause_agreement=True,
                path_similarity=1.0,
                cohens_kappa=1.0
            )

    def save_labels(
        self,
        labels: GroundTruthLabel,
        model_name: str = "gpt4"
    ) -> Path:
        """Save labels to disk."""
        model_dir = self.output_dir / model_name
        model_dir.mkdir(parents=True, exist_ok=True)

        filepath = model_dir / f"{labels.scenario_id}_labels.json"
        with open(filepath, 'w') as f:
            json.dump(asdict(labels), f, indent=2)

        return filepath

    def save_cross_validation(self, result: CrossValidationResult) -> Path:
        """Save cross-validation result."""
        filepath = self.output_dir / f"{result.scenario_id}_cross_validation.json"

        data = {
            "scenario_id": result.scenario_id,
            "edge_agreement_rate": result.edge_agreement_rate,
            "root_cause_agreement": result.root_cause_agreement,
            "path_similarity": result.path_similarity,
            "cohens_kappa": result.cohens_kappa,
            "gpt4_labels": asdict(result.gpt4_labels),
            "claude_labels": asdict(result.claude_labels) if result.claude_labels else None
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        return filepath


async def label_all_traces(
    scenarios_dir: Path = Path("data/scenarios"),
    traces_dir: Path = Path("data/traces"),
    labels_dir: Path = Path("data/labels"),
    cross_validate: bool = True,
    max_concurrent: int = 3
) -> dict[str, list[CrossValidationResult]]:
    """Label all traces in the benchmark."""

    from .scenario_generator import ScenarioDomain, Scenario
    from .trace_collector import TraceResult

    labeler = GroundTruthLabeler(output_dir=labels_dir)
    results_by_domain = {}

    for domain in ScenarioDomain:
        print(f"\n{'='*50}")
        print(f"Labeling traces for {domain.value}...")
        print('='*50)

        # Load scenarios and traces
        domain_dir = scenarios_dir / domain.value
        if not domain_dir.exists():
            continue

        results = []
        semaphore = asyncio.Semaphore(max_concurrent)

        async def process_scenario(scenario_file: Path):
            async with semaphore:
                # Load scenario
                with open(scenario_file) as f:
                    scenario = Scenario.from_dict(json.load(f))

                # Load trace
                trace_file = traces_dir / f"{scenario.scenario_id}_trace.json"
                if not trace_file.exists():
                    print(f"  No trace found for {scenario.scenario_id}")
                    return None

                with open(trace_file) as f:
                    trace_data = json.load(f)
                trace_result = TraceResult(**trace_data)

                print(f"  Labeling {scenario.scenario_id}...")

                if cross_validate:
                    result = await labeler.label_with_cross_validation(scenario, trace_result)
                    labeler.save_cross_validation(result)
                else:
                    labels = await labeler.label_with_gpt4(scenario, trace_result)
                    labeler.save_labels(labels, "gpt4")
                    result = CrossValidationResult(
                        scenario_id=scenario.scenario_id,
                        gpt4_labels=labels,
                        claude_labels=None,
                        edge_agreement_rate=1.0,
                        root_cause_agreement=True,
                        path_similarity=1.0,
                        cohens_kappa=1.0
                    )

                return result

        tasks = [
            process_scenario(f) for f in domain_dir.glob("*.json")
        ]

        task_results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in task_results:
            if isinstance(result, Exception):
                print(f"  Error: {result}")
            elif result is not None:
                results.append(result)

        results_by_domain[domain.value] = results

        # Print summary
        if results:
            avg_agreement = sum(r.edge_agreement_rate for r in results) / len(results)
            root_cause_matches = sum(1 for r in results if r.root_cause_agreement)
            avg_kappa = sum(r.cohens_kappa for r in results) / len(results)

            print(f"\n  Summary for {domain.value}:")
            print(f"    Labeled: {len(results)} traces")
            print(f"    Avg edge agreement: {avg_agreement:.2%}")
            print(f"    Root cause matches: {root_cause_matches}/{len(results)}")
            print(f"    Avg Cohen's Kappa: {avg_kappa:.3f}")

    return results_by_domain


# CLI interface
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate ground truth labels")
    parser.add_argument("--scenarios-dir", type=str, default="data/scenarios")
    parser.add_argument("--traces-dir", type=str, default="data/traces")
    parser.add_argument("--labels-dir", type=str, default="data/labels")
    parser.add_argument("--no-cross-validate", action="store_true",
                        help="Skip Claude cross-validation")
    parser.add_argument("--max-concurrent", type=int, default=3)

    args = parser.parse_args()

    asyncio.run(label_all_traces(
        scenarios_dir=Path(args.scenarios_dir),
        traces_dir=Path(args.traces_dir),
        labels_dir=Path(args.labels_dir),
        cross_validate=not args.no_cross_validate,
        max_concurrent=args.max_concurrent
    ))
