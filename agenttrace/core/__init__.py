"""Core data structures for the causal trace graph."""

from agenttrace.core.node import Node, NodeType
from agenttrace.core.edge import Edge, EdgeType
from agenttrace.core.graph import CausalGraph

__all__ = ["Node", "NodeType", "Edge", "EdgeType", "CausalGraph"]
