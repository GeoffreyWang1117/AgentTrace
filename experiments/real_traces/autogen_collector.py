"""
AutoGen Trace Collector for Real System Validation.

Creates AutoGen multi-agent workflows with controlled failures,
collects traces via OpenTelemetry, and converts to AgentTrace format.
"""

import json
import time
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Any

from agenttrace.core.graph import CausalGraph
from agenttrace.core.node import Node, NodeType
from agenttrace.core.edge import Edge, EdgeType


@dataclass
class RealScenario:
    """A real-system scenario with ground truth annotation."""
    scenario_id: str
    description: str
    failure_type: str  # 'wrong_tool_result', 'lost_context', 'wrong_routing', etc.
    agents: list[str]
    ground_truth: dict  # manual annotation
    trace_spans: list[dict] = field(default_factory=list)


@dataclass
class OTelSpan:
    """Simplified OpenTelemetry span representation."""
    trace_id: str
    span_id: str
    parent_span_id: Optional[str]
    name: str
    start_time: float
    end_time: float
    attributes: dict[str, Any]
    events: list[dict] = field(default_factory=list)


class OTelToAgentTrace:
    """
    Adapter: converts OpenTelemetry spans to AgentTrace Node/Edge objects.
    """

    def __init__(self):
        self._span_to_node_id: dict[str, str] = {}

    def convert(self, spans: list[OTelSpan], run_id: str = "") -> CausalGraph:
        """
        Convert OTel spans to a CausalGraph.

        Mapping:
        - Each span -> Node
        - Parent-child span relationship -> Edge (TRIGGER_RESPONSE)
        - Temporal adjacency within same service -> Edge (TEMPORAL)
        """
        graph = CausalGraph(run_id=run_id or f"otel_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

        # Sort spans by start time
        spans = sorted(spans, key=lambda s: s.start_time)

        # Create nodes
        for span in spans:
            node_type = self._infer_node_type(span)
            agent_id = span.attributes.get("agent.name", span.attributes.get("service.name", "unknown"))

            node = Node(
                type=node_type,
                agent_id=agent_id,
                data={
                    "span_name": span.name,
                    "attributes": span.attributes,
                    "duration_ms": (span.end_time - span.start_time) * 1000,
                    "events": span.events,
                },
                metadata={
                    "otel_span_id": span.span_id,
                    "otel_trace_id": span.trace_id,
                },
            )
            node.timestamp = datetime.fromtimestamp(span.start_time)

            if span.parent_span_id and span.parent_span_id in self._span_to_node_id:
                node.parent_ids = [self._span_to_node_id[span.parent_span_id]]

            graph.add_node(node)
            self._span_to_node_id[span.span_id] = node.id

        # Add parent-child edges
        for span in spans:
            if span.parent_span_id and span.parent_span_id in self._span_to_node_id:
                parent_node_id = self._span_to_node_id[span.parent_span_id]
                child_node_id = self._span_to_node_id[span.span_id]

                if parent_node_id in graph._nodes and child_node_id in graph._nodes:
                    edge = Edge(
                        source_id=parent_node_id,
                        target_id=child_node_id,
                        type=EdgeType.TRIGGER_RESPONSE,
                        confidence=1.0,
                        metadata={"source": "otel_parent_child"},
                    )
                    try:
                        graph.add_edge(edge)
                    except ValueError:
                        pass

        # Add temporal edges between sequential spans from same agent
        agent_spans: dict[str, list[tuple[float, str]]] = {}
        for span in spans:
            agent_id = span.attributes.get("agent.name", span.attributes.get("service.name", "unknown"))
            node_id = self._span_to_node_id[span.span_id]
            agent_spans.setdefault(agent_id, []).append((span.start_time, node_id))

        for agent_id, span_list in agent_spans.items():
            span_list.sort()
            for i in range(len(span_list) - 1):
                src_id = span_list[i][1]
                tgt_id = span_list[i + 1][1]
                if src_id in graph._nodes and tgt_id in graph._nodes:
                    edge = Edge(
                        source_id=src_id,
                        target_id=tgt_id,
                        type=EdgeType.TEMPORAL,
                        confidence=0.8,
                        metadata={"source": "otel_temporal", "agent_id": agent_id},
                    )
                    try:
                        graph.add_edge(edge)
                    except ValueError:
                        pass

        return graph

    def _infer_node_type(self, span: OTelSpan) -> NodeType:
        """Infer AgentTrace NodeType from OTel span attributes."""
        name = span.name.lower()
        attrs = span.attributes

        if "tool" in name or attrs.get("openai.tool_call"):
            return NodeType.TOOL_CALL
        if "tool_result" in name or "tool_response" in name:
            return NodeType.TOOL_RESULT
        if "input" in name or "receive" in name:
            return NodeType.AGENT_INPUT
        if "output" in name or "send" in name or "response" in name:
            return NodeType.AGENT_OUTPUT
        if "decision" in name or "route" in name:
            return NodeType.DECISION
        if "error" in name or "exception" in name:
            return NodeType.ERROR
        if "state" in name:
            if "read" in name:
                return NodeType.STATE_READ
            return NodeType.STATE_WRITE

        return NodeType.DECISION


# ============================================================
# Scenario definitions for AutoGen real-system validation
# ============================================================

AUTOGEN_SCENARIOS = [
    # GroupChat scenarios
    {
        "id": "autogen_gc_wrong_tool",
        "description": "GroupChat: tool returns wrong data, propagates through discussion",
        "failure_type": "wrong_tool_result",
        "n_agents": 3,
        "pattern": "group_chat",
    },
    {
        "id": "autogen_gc_lost_context",
        "description": "GroupChat: important context lost in long conversation",
        "failure_type": "lost_context",
        "n_agents": 4,
        "pattern": "group_chat",
    },
    {
        "id": "autogen_gc_wrong_routing",
        "description": "GroupChat: message routed to wrong specialist agent",
        "failure_type": "wrong_routing",
        "n_agents": 4,
        "pattern": "group_chat",
    },
    {
        "id": "autogen_gc_conflicting_info",
        "description": "GroupChat: two agents provide conflicting information",
        "failure_type": "state_inconsistency",
        "n_agents": 3,
        "pattern": "group_chat",
    },
    {
        "id": "autogen_gc_cascade",
        "description": "GroupChat: timeout in one agent causes cascade failure",
        "failure_type": "timeout_cascade",
        "n_agents": 3,
        "pattern": "group_chat",
    },
    # Two-agent-chat scenarios
    {
        "id": "autogen_2a_data_corrupt",
        "description": "Two-agent chat: data gets corrupted in transfer",
        "failure_type": "data_corruption",
        "n_agents": 2,
        "pattern": "two_agent",
    },
    {
        "id": "autogen_2a_logic_error",
        "description": "Two-agent chat: incorrect decision logic",
        "failure_type": "logic_error",
        "n_agents": 2,
        "pattern": "two_agent",
    },
    {
        "id": "autogen_2a_invalid_output",
        "description": "Two-agent chat: invalid output format breaks downstream",
        "failure_type": "invalid_output",
        "n_agents": 2,
        "pattern": "two_agent",
    },
    {
        "id": "autogen_2a_missing_context",
        "description": "Two-agent chat: critical context not passed",
        "failure_type": "missing_context",
        "n_agents": 2,
        "pattern": "two_agent",
    },
    {
        "id": "autogen_2a_race_condition",
        "description": "Two-agent chat: concurrent state access causes inconsistency",
        "failure_type": "race_condition",
        "n_agents": 2,
        "pattern": "two_agent",
    },
]


def save_ground_truth(
    scenario_id: str,
    root_cause_description: str,
    root_cause_span_id: str,
    error_span_id: str,
    causal_chain: list[str],
    output_dir: Path = Path("experiments/real_traces/ground_truth"),
):
    """Save manual ground truth annotation for a real scenario."""
    output_dir.mkdir(parents=True, exist_ok=True)

    annotation = {
        "scenario_id": scenario_id,
        "root_cause_description": root_cause_description,
        "root_cause_span_id": root_cause_span_id,
        "error_span_id": error_span_id,
        "causal_chain": causal_chain,
        "annotated_at": datetime.now().isoformat(),
    }

    filepath = output_dir / f"{scenario_id}_gt.json"
    with open(filepath, 'w') as f:
        json.dump(annotation, f, indent=2)

    return filepath


def collect_autogen_trace(scenario_config: dict) -> Optional[list[OTelSpan]]:
    """
    Run an AutoGen scenario and collect OTel spans.

    Returns None if pyautogen is not installed.
    """
    try:
        import autogen
    except ImportError:
        print("WARNING: pyautogen not installed, cannot collect real traces")
        return None

    # This is a placeholder — actual AutoGen workflow setup would go here.
    # In practice, you would:
    # 1. Set up OpenTelemetry exporter
    # 2. Create AutoGen agents with tracing enabled
    # 3. Run the workflow
    # 4. Collect spans from the exporter
    print(f"  Collecting trace for {scenario_config['id']}...")
    return []


if __name__ == "__main__":
    print("AutoGen Real Trace Collector")
    print("=" * 40)
    print(f"Defined {len(AUTOGEN_SCENARIOS)} scenarios")

    for s in AUTOGEN_SCENARIOS:
        print(f"  {s['id']}: {s['description']}")

    print("\nTo collect traces, ensure pyautogen>=0.4 and opentelemetry are installed.")
