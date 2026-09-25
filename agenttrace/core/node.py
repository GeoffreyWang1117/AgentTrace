"""
Node definitions for the Causal Trace Graph.

Nodes represent decision points in the multi-agent system:
- Agent outputs (decisions, responses)
- Tool calls (external interactions)
- State changes (internal state mutations)
"""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import Any
from dataclasses import dataclass, field


class NodeType(Enum):
    """Types of nodes in the causal graph."""

    AGENT_INPUT = "agent_input"  # Input received by an agent
    AGENT_OUTPUT = "agent_output"  # Output produced by an agent
    TOOL_CALL = "tool_call"  # Tool invocation
    TOOL_RESULT = "tool_result"  # Tool return value
    STATE_READ = "state_read"  # State variable read
    STATE_WRITE = "state_write"  # State variable write
    DECISION = "decision"  # Explicit decision point
    ERROR = "error"  # Error occurrence
    CHECKPOINT = "checkpoint"  # Manual checkpoint


@dataclass
class Node:
    """
    A node in the causal trace graph representing a decision point.

    Attributes:
        id: Unique identifier for this node
        type: The type of decision point
        agent_id: ID of the agent that created this node
        timestamp: When this node was created
        data: The actual data/value at this decision point
        metadata: Additional context (e.g., function name, parameters)
        parent_ids: IDs of nodes that directly caused this node
        run_id: ID of the execution run this node belongs to
    """

    type: NodeType
    agent_id: str
    data: Any
    metadata: dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)
    parent_ids: list[str] = field(default_factory=list)
    run_id: str = ""

    def __hash__(self) -> int:
        return hash(self.id)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Node):
            return False
        return self.id == other.id

    def to_dict(self) -> dict[str, Any]:
        """Convert node to dictionary for serialization."""
        return {
            "id": self.id,
            "type": self.type.value,
            "agent_id": self.agent_id,
            "timestamp": self.timestamp.isoformat(),
            "data": self._serialize_data(self.data),
            "metadata": self.metadata,
            "parent_ids": self.parent_ids,
            "run_id": self.run_id,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Node:
        """Create node from dictionary."""
        return cls(
            id=d["id"],
            type=NodeType(d["type"]),
            agent_id=d["agent_id"],
            timestamp=datetime.fromisoformat(d["timestamp"]),
            data=d["data"],
            metadata=d.get("metadata", {}),
            parent_ids=d.get("parent_ids", []),
            run_id=d.get("run_id", ""),
        )

    def _serialize_data(self, data: Any) -> Any:
        """Serialize data for JSON storage."""
        if isinstance(data, (str, int, float, bool, type(None))):
            return data
        if isinstance(data, (list, tuple)):
            return [self._serialize_data(item) for item in data]
        if isinstance(data, dict):
            return {k: self._serialize_data(v) for k, v in data.items()}
        # For other types, convert to string representation
        return str(data)

    @property
    def summary(self) -> str:
        """Get a brief summary of this node."""
        data_preview = str(self.data)[:50]
        if len(str(self.data)) > 50:
            data_preview += "..."
        return f"[{self.type.value}] {self.agent_id}: {data_preview}"
