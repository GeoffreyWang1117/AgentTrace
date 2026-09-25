"""
Layer 1: Unified Trace Schema.

All datasets (Who&When, AgentRx, Synthetic, real frameworks) are
normalized to this schema before graph construction and evaluation.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class UnifiedNode:
    """A single agent action/step in a unified trace."""
    node_id: str
    agent_id: str
    step: int
    action: str          # process, submit, tool_call, plan, verify, etc.
    content: str         # text content of the action
    role: str = ""       # user, assistant, tool, system
    node_type: str = ""  # planning, execution, tool_use, communication, aggregation
    is_root_cause: bool = False  # ground truth (kept separate from graph)


@dataclass
class UnifiedEdge:
    """A causal dependency between two nodes."""
    source_id: str
    target_id: str
    edge_type: str  # sequential, communication, data_flow, tool_dependency
    confidence: float = 1.0


@dataclass
class UnifiedTrace:
    """Normalized trace from any source."""
    trace_id: str
    source: str          # who_when, agentrx, synthetic, autogen, langgraph, crewai
    nodes: list[UnifiedNode] = field(default_factory=list)
    edges: list[UnifiedEdge] = field(default_factory=list)

    # Ground truth (separate from graph)
    root_cause_node_id: str = ""
    root_cause_agent: str = ""
    root_cause_step: int = -1
    error_node_id: str = ""
    failure_type: str = ""  # planning, tool_misuse, communication, hallucination, etc.

    # Metadata
    n_agents: int = 0
    n_steps: int = 0
    topology: str = ""  # chain, tree, dag, dense (computed after construction)

    @property
    def has_ground_truth(self):
        return bool(self.root_cause_node_id)
