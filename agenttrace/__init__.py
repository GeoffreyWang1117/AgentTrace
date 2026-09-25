"""
AgentTrace: Multi-Agent Causal Tracing System

A debugging tool for complex multi-agent systems that builds causal trace graphs
to answer "why" questions about system behavior.
"""

from agenttrace.core.graph import CausalGraph
from agenttrace.core.node import Node, NodeType
from agenttrace.core.edge import Edge, EdgeType
from agenttrace.hooks.decorators import trace_agent, trace_tool
from agenttrace.tracer import Tracer

__version__ = "0.1.0"
__all__ = [
    "CausalGraph",
    "Node",
    "NodeType",
    "Edge",
    "EdgeType",
    "Tracer",
    "trace_agent",
    "trace_tool",
]
