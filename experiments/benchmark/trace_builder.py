"""
Trace Builder for AgentTrace Benchmark.

This module builds CausalGraph traces directly from scenario definitions,
providing accurate traces for evaluation without requiring LLM simulation.
"""

import json
from pathlib import Path
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from typing import Optional

from agenttrace.core.graph import CausalGraph
from agenttrace.core.node import Node, NodeType
from agenttrace.core.edge import Edge, EdgeType

from .scenario_generator import Scenario, Message, ScenarioDomain


@dataclass
class GroundTruth:
    """Ground truth for evaluation, kept separate from the graph."""
    root_cause_node_id: str
    error_node_id: str
    causal_edges: list[tuple[str, str, str]]  # (src, tgt, type)
    causal_path: list[str]


@dataclass
class BuiltTrace:
    """A trace built from scenario definition."""
    scenario_id: str
    graph: CausalGraph
    node_count: int
    edge_count: int
    root_cause_node_id: str
    error_node_id: str
    causal_path: list[str]


class TraceBuilder:
    """
    Builds CausalGraph traces directly from scenario definitions.

    This provides accurate traces for evaluation without expensive LLM simulation.
    """

    def __init__(self, output_dir: Path = Path("data/traces")):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def build_trace(self, scenario: Scenario) -> BuiltTrace:
        """
        Build a CausalGraph from a scenario definition.

        Creates nodes for each step in the buggy flow and edges based on
        the scenario's causal chain definition.
        """
        graph = CausalGraph(run_id=f"scenario_{scenario.scenario_id}")

        # Base timestamp for the trace
        base_time = datetime.now()

        # Create nodes for each step in the buggy flow
        step_to_node: dict[int, Node] = {}

        for i, msg in enumerate(scenario.buggy_flow):
            # Determine node type based on the message
            if msg.is_bug_point:
                node_type = NodeType.ERROR
            elif "tool" in msg.action.lower() or "call" in msg.action.lower():
                node_type = NodeType.TOOL_CALL
            elif msg.from_agent == "user":
                node_type = NodeType.AGENT_INPUT
            elif msg.to_agent == "user" or msg.to_agent in ["output", "final_output"]:
                node_type = NodeType.AGENT_OUTPUT
            else:
                # Default to decision for inter-agent communication
                node_type = NodeType.DECISION

            # Create node with realistic data
            node = Node(
                type=node_type,
                agent_id=msg.from_agent or "unknown",
                data={
                    "step": msg.step,
                    "action": msg.action,
                    "content": msg.content,
                    "to_agent": msg.to_agent,
                    "is_bug_point": msg.is_bug_point,
                    "bug_description": msg.bug_description
                },
                metadata={
                    "scenario_id": scenario.scenario_id,
                    "domain": scenario.domain.value,
                    "bug_type": scenario.bug_type.value
                }
            )

            # Adjust timestamp to maintain order
            node.timestamp = base_time + timedelta(milliseconds=i * 100)

            # Track parent relationships (previous step)
            if i > 0:
                prev_node = step_to_node.get(scenario.buggy_flow[i-1].step)
                if prev_node:
                    node.parent_ids = [prev_node.id]

            graph.add_node(node)
            step_to_node[msg.step] = node

        # Create edges based on scenario's causal chain
        for causal_edge in scenario.causal_chain:
            source_node = step_to_node.get(causal_edge.source_step)
            target_node = step_to_node.get(causal_edge.target_step)

            if source_node and target_node:
                # Map relationship type
                edge_type_map = {
                    "data_flow": EdgeType.DATA_FLOW,
                    "trigger": EdgeType.TRIGGER_RESPONSE,
                    "state_dependency": EdgeType.STATE_DEPENDENCY,
                    "temporal": EdgeType.TEMPORAL
                }
                edge_type = edge_type_map.get(
                    causal_edge.relationship_type,
                    EdgeType.DATA_FLOW
                )

                edge = Edge(
                    source_id=source_node.id,
                    target_id=target_node.id,
                    type=edge_type,
                    confidence=1.0,  # Ground truth edges
                    metadata={
                        "description": causal_edge.description,
                        "source_step": causal_edge.source_step,
                        "target_step": causal_edge.target_step
                    }
                )
                graph.add_edge(edge)

        # Also add sequential edges for temporal flow
        # Track existing edges to avoid duplicates
        existing_edges = set()
        for causal_edge in scenario.causal_chain:
            source_node = step_to_node.get(causal_edge.source_step)
            target_node = step_to_node.get(causal_edge.target_step)
            if source_node and target_node:
                existing_edges.add((source_node.id, target_node.id))

        prev_node = None
        for step in sorted(step_to_node.keys()):
            node = step_to_node[step]
            if prev_node:
                # Check if edge already exists
                if (prev_node.id, node.id) not in existing_edges:
                    edge = Edge(
                        source_id=prev_node.id,
                        target_id=node.id,
                        type=EdgeType.TEMPORAL,
                        confidence=0.9,
                        metadata={"inferred": True, "type": "sequential"}
                    )
                    graph.add_edge(edge)
                    existing_edges.add((prev_node.id, node.id))
            prev_node = node

        # Identify key nodes
        root_cause_node = step_to_node.get(scenario.root_cause_step)
        error_node = step_to_node.get(scenario.error_manifestation_step)

        # Build causal path
        causal_path = []
        if root_cause_node:
            causal_path.append(root_cause_node.id)
            for edge in scenario.causal_chain:
                target_node = step_to_node.get(edge.target_step)
                if target_node:
                    causal_path.append(target_node.id)

        return BuiltTrace(
            scenario_id=scenario.scenario_id,
            graph=graph,
            node_count=graph.node_count,
            edge_count=graph.edge_count,
            root_cause_node_id=root_cause_node.id if root_cause_node else "",
            error_node_id=error_node.id if error_node else "",
            causal_path=causal_path
        )

    def save_trace(self, built_trace: BuiltTrace) -> Path:
        """Save a built trace to disk."""
        filepath = self.output_dir / f"{built_trace.scenario_id}_trace.json"

        data = {
            "scenario_id": built_trace.scenario_id,
            "trace_json": built_trace.graph.to_json(),
            "node_count": built_trace.node_count,
            "edge_count": built_trace.edge_count,
            "root_cause_node_id": built_trace.root_cause_node_id,
            "error_node_id": built_trace.error_node_id,
            "causal_path": built_trace.causal_path,
            "execution_log": [],  # Not from simulation
            "error_occurred": True,  # Buggy scenarios have errors
            "built_at": datetime.now().isoformat()
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        return filepath

    def load_trace(self, scenario_id: str) -> Optional[BuiltTrace]:
        """Load a built trace from disk."""
        filepath = self.output_dir / f"{scenario_id}_trace.json"
        if not filepath.exists():
            return None

        with open(filepath) as f:
            data = json.load(f)

        graph = CausalGraph.from_json(data["trace_json"])

        return BuiltTrace(
            scenario_id=data["scenario_id"],
            graph=graph,
            node_count=data["node_count"],
            edge_count=data["edge_count"],
            root_cause_node_id=data["root_cause_node_id"],
            error_node_id=data["error_node_id"],
            causal_path=data["causal_path"]
        )


class BlindTraceBuilder:
    """
    Builds CausalGraph traces WITHOUT ground truth causal edges.

    Only sequential/temporal edges are included. Ground truth is returned
    separately so that evaluation can measure inference quality independently
    from ranking quality (two-stage evaluation).
    """

    def __init__(self, output_dir: Path = Path("data/traces_blind")):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def build_blind_trace(self, scenario: Scenario) -> tuple[CausalGraph, GroundTruth]:
        """
        Build a CausalGraph with ONLY temporal/sequential edges.

        Returns:
            (graph, ground_truth) - graph has no causal leakage,
            ground truth is kept separate for evaluation.
        """
        graph = CausalGraph(run_id=f"blind_{scenario.scenario_id}")
        base_time = datetime.now()

        step_to_node: dict[int, Node] = {}

        for i, msg in enumerate(scenario.buggy_flow):
            # Determine node type — do NOT use is_bug_point to set ERROR type
            if "tool" in msg.action.lower() or "call" in msg.action.lower():
                node_type = NodeType.TOOL_CALL
            elif msg.from_agent == "user":
                node_type = NodeType.AGENT_INPUT
            elif msg.to_agent == "user" or msg.to_agent in ["output", "final_output"]:
                node_type = NodeType.AGENT_OUTPUT
            else:
                node_type = NodeType.DECISION

            # Strip ground truth fields from node data
            node = Node(
                type=node_type,
                agent_id=msg.from_agent or "unknown",
                data={
                    "step": msg.step,
                    "action": msg.action,
                    "content": msg.content,
                    "to_agent": msg.to_agent,
                },
                metadata={
                    "scenario_id": scenario.scenario_id,
                    "domain": scenario.domain.value,
                    "bug_type": scenario.bug_type.value
                }
            )

            node.timestamp = base_time + timedelta(milliseconds=i * 100)

            if i > 0:
                prev_node = step_to_node.get(scenario.buggy_flow[i-1].step)
                if prev_node:
                    node.parent_ids = [prev_node.id]

            graph.add_node(node)
            step_to_node[msg.step] = node

        # Add ONLY sequential/temporal edges (no causal chain edges)
        prev_node = None
        existing_edges = set()
        for step in sorted(step_to_node.keys()):
            node = step_to_node[step]
            if prev_node:
                if (prev_node.id, node.id) not in existing_edges:
                    edge = Edge(
                        source_id=prev_node.id,
                        target_id=node.id,
                        type=EdgeType.TEMPORAL,
                        confidence=0.9,
                        metadata={"inferred": True, "type": "sequential"}
                    )
                    graph.add_edge(edge)
                    existing_edges.add((prev_node.id, node.id))
            prev_node = node

        # Build ground truth separately
        root_cause_node = step_to_node.get(scenario.root_cause_step)
        error_node = step_to_node.get(scenario.error_manifestation_step)

        causal_edges = []
        causal_path = []
        if root_cause_node:
            causal_path.append(root_cause_node.id)

        for causal_edge in scenario.causal_chain:
            source_node = step_to_node.get(causal_edge.source_step)
            target_node = step_to_node.get(causal_edge.target_step)
            if source_node and target_node:
                causal_edges.append((
                    source_node.id,
                    target_node.id,
                    causal_edge.relationship_type
                ))
                if target_node.id not in causal_path:
                    causal_path.append(target_node.id)

        ground_truth = GroundTruth(
            root_cause_node_id=root_cause_node.id if root_cause_node else "",
            error_node_id=error_node.id if error_node else "",
            causal_edges=causal_edges,
            causal_path=causal_path
        )

        return graph, ground_truth

    def save_blind_trace(
        self,
        scenario_id: str,
        graph: CausalGraph,
        ground_truth: GroundTruth
    ) -> Path:
        """Save blind trace and ground truth to disk."""
        filepath = self.output_dir / f"{scenario_id}_blind_trace.json"

        data = {
            "scenario_id": scenario_id,
            "trace_json": graph.to_json(),
            "node_count": graph.node_count,
            "edge_count": graph.edge_count,
            "ground_truth": {
                "root_cause_node_id": ground_truth.root_cause_node_id,
                "error_node_id": ground_truth.error_node_id,
                "causal_edges": [
                    {"source": s, "target": t, "type": tp}
                    for s, t, tp in ground_truth.causal_edges
                ],
                "causal_path": ground_truth.causal_path,
            },
            "built_at": datetime.now().isoformat()
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        return filepath


def build_all_traces(
    scenarios_dir: Path = Path("data/scenarios"),
    traces_dir: Path = Path("data/traces")
) -> dict[str, list[BuiltTrace]]:
    """Build traces for all scenarios."""
    builder = TraceBuilder(output_dir=traces_dir)
    results = {}

    for domain in ScenarioDomain:
        domain_dir = scenarios_dir / domain.value
        if not domain_dir.exists():
            continue

        print(f"[{domain.value}] Building traces...")
        domain_traces = []

        for scenario_file in domain_dir.glob("*.json"):
            with open(scenario_file) as f:
                scenario = Scenario.from_dict(json.load(f))

            try:
                built_trace = builder.build_trace(scenario)
                builder.save_trace(built_trace)
                domain_traces.append(built_trace)
            except Exception as e:
                print(f"  Error building trace for {scenario.scenario_id}: {e}")

        results[domain.value] = domain_traces
        print(f"[{domain.value}] ✓ {len(domain_traces)} traces built")

    return results


if __name__ == "__main__":
    results = build_all_traces()
    total = sum(len(v) for v in results.values())
    print(f"\nTotal traces built: {total}")
