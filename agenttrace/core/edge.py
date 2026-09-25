"""
Edge definitions for the Causal Trace Graph.

Edges represent causal relationships between nodes:
- Input-Output: An input caused an output
- Trigger-Response: An event triggered a response
- Data flow: Data flowed from one point to another
"""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import Any
from dataclasses import dataclass, field


class EdgeType(Enum):
    """Types of causal relationships."""

    INPUT_OUTPUT = "input_output"       # Input directly produced output
    TRIGGER_RESPONSE = "trigger_response"  # Event triggered a response
    DATA_FLOW = "data_flow"             # Data flowed between points
    STATE_DEPENDENCY = "state_dependency"  # Output depended on state
    TEMPORAL = "temporal"               # Temporal ordering (weak causality)
    TOOL_INVOCATION = "tool_invocation"  # Agent invoked a tool
    ERROR_PROPAGATION = "error_propagation"  # Error propagated


@dataclass
class Edge:
    """
    An edge in the causal trace graph representing a causal relationship.

    Attributes:
        source_id: ID of the source node (cause)
        target_id: ID of the target node (effect)
        type: The type of causal relationship
        confidence: Confidence score for inferred relationships (0-1)
        metadata: Additional context about the relationship
    """

    source_id: str
    target_id: str
    type: EdgeType
    confidence: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)

    def __post_init__(self):
        if not 0 <= self.confidence <= 1:
            raise ValueError(f"Confidence must be between 0 and 1, got {self.confidence}")

    def __hash__(self) -> int:
        return hash(self.id)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Edge):
            return False
        return self.id == other.id

    def to_dict(self) -> dict[str, Any]:
        """Convert edge to dictionary for serialization."""
        return {
            "id": self.id,
            "source_id": self.source_id,
            "target_id": self.target_id,
            "type": self.type.value,
            "confidence": self.confidence,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat(),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Edge:
        """Create edge from dictionary."""
        return cls(
            id=d["id"],
            source_id=d["source_id"],
            target_id=d["target_id"],
            type=EdgeType(d["type"]),
            confidence=d.get("confidence", 1.0),
            metadata=d.get("metadata", {}),
            timestamp=datetime.fromisoformat(d["timestamp"]),
        )

    @property
    def is_strong(self) -> bool:
        """Check if this is a strong causal relationship."""
        return self.confidence >= 0.8

    @property
    def summary(self) -> str:
        """Get a brief summary of this edge."""
        return f"{self.source_id[:8]}... --[{self.type.value}]--> {self.target_id[:8]}..."
