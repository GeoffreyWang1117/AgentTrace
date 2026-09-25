"""
Main Tracer class for AgentTrace.

The Tracer is the primary interface for recording and querying
causal traces in multi-agent systems.
"""

from __future__ import annotations

import threading
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Callable, Iterator

from agenttrace.core.graph import CausalGraph
from agenttrace.core.node import Node, NodeType
from agenttrace.core.edge import Edge, EdgeType

# Global tracer instance
_global_tracer: "Tracer | None" = None
_tracer_lock = threading.Lock()


def get_tracer() -> "Tracer | None":
    """Get the global tracer instance."""
    return _global_tracer


def set_tracer(tracer: "Tracer | None") -> None:
    """Set the global tracer instance."""
    global _global_tracer
    _global_tracer = tracer


class Tracer:
    """
    Main tracer class for recording and analyzing causal traces.

    Usage:
        # Create and activate a tracer
        tracer = Tracer()
        tracer.start()

        # Record events
        tracer.record(
            node_type=NodeType.AGENT_INPUT,
            agent_id="my_agent",
            data={"query": "Hello"},
        )

        # Query traces
        causes = tracer.trace_backward(error_node_id)

        # Stop and get results
        tracer.stop()
        graph = tracer.graph
    """

    def __init__(
        self,
        run_id: str | None = None,
        auto_activate: bool = False,
    ):
        """
        Initialize a new tracer.

        Args:
            run_id: Optional run identifier
            auto_activate: If True, automatically set as global tracer
        """
        self._graph = CausalGraph(run_id=run_id)
        self._active = False
        self._current_node_id: str | None = None
        self._context_stack: list[str] = []

        # Thread-local storage for current node ID
        self._local = threading.local()

        if auto_activate:
            self.start()

    @property
    def graph(self) -> CausalGraph:
        """Get the underlying causal graph."""
        return self._graph

    @property
    def is_active(self) -> bool:
        """Check if tracer is active."""
        return self._active

    @property
    def run_id(self) -> str:
        """Get the run ID."""
        return self._graph.run_id

    @property
    def current_node_id(self) -> str | None:
        """Get the current context node ID for this thread."""
        return getattr(self._local, 'current_node_id', None)

    @current_node_id.setter
    def current_node_id(self, value: str | None) -> None:
        """Set the current context node ID for this thread."""
        self._local.current_node_id = value

    def start(self) -> "Tracer":
        """
        Start the tracer and set it as the global instance.

        Returns:
            self for method chaining
        """
        self._active = True
        set_tracer(self)
        return self

    def stop(self) -> "Tracer":
        """
        Stop the tracer and clear the global instance.

        Returns:
            self for method chaining
        """
        self._active = False
        if get_tracer() is self:
            set_tracer(None)
        return self

    def record(
        self,
        node_type: NodeType,
        agent_id: str,
        data: Any,
        metadata: dict[str, Any] | None = None,
        parent_ids: list[str] | None = None,
    ) -> Node:
        """
        Record a new node in the causal graph.

        Args:
            node_type: Type of the node
            agent_id: ID of the agent creating this node
            data: The data/value at this decision point
            metadata: Additional context
            parent_ids: IDs of parent nodes (causes)

        Returns:
            The created node
        """
        if not self._active:
            # Still create the node but don't add to graph
            return Node(
                type=node_type,
                agent_id=agent_id,
                data=data,
                metadata=metadata or {},
                parent_ids=parent_ids or [],
            )

        node = Node(
            type=node_type,
            agent_id=agent_id,
            data=data,
            metadata=metadata or {},
            parent_ids=parent_ids or [],
        )

        self._graph.add_node(node)
        return node

    def link(
        self,
        source_id: str,
        target_id: str,
        edge_type: EdgeType = EdgeType.DATA_FLOW,
        confidence: float = 1.0,
        metadata: dict[str, Any] | None = None,
    ) -> Edge:
        """
        Create a causal link between two nodes.

        Args:
            source_id: Source node ID (cause)
            target_id: Target node ID (effect)
            edge_type: Type of causal relationship
            confidence: Confidence in this relationship (0-1)
            metadata: Additional context

        Returns:
            The created edge
        """
        edge = Edge(
            source_id=source_id,
            target_id=target_id,
            type=edge_type,
            confidence=confidence,
            metadata=metadata or {},
        )
        return self._graph.add_edge(edge)

    def checkpoint(
        self,
        agent_id: str,
        name: str,
        data: Any = None,
    ) -> Node:
        """
        Create a named checkpoint in the trace.

        Checkpoints are useful for marking significant points
        in the execution that you may want to return to.

        Args:
            agent_id: Agent creating the checkpoint
            name: Name of the checkpoint
            data: Optional data to associate with checkpoint

        Returns:
            The checkpoint node
        """
        return self.record(
            node_type=NodeType.CHECKPOINT,
            agent_id=agent_id,
            data=data,
            metadata={"checkpoint_name": name},
            parent_ids=[self.current_node_id] if self.current_node_id else [],
        )

    @contextmanager
    def span(
        self,
        agent_id: str,
        operation: str,
        data: Any = None,
    ) -> Iterator[Node]:
        """
        Create a traced span for a block of code.

        Usage:
            with tracer.span("my_agent", "process_request") as span:
                # code here is traced
                result = do_work()

        Args:
            agent_id: Agent ID for this span
            operation: Name of the operation
            data: Optional input data

        Yields:
            The span's input node
        """
        input_node = self.record(
            node_type=NodeType.AGENT_INPUT,
            agent_id=agent_id,
            data=data,
            metadata={"operation": operation},
            parent_ids=[self.current_node_id] if self.current_node_id else [],
        )

        prev_node_id = self.current_node_id
        self.current_node_id = input_node.id

        error = None
        try:
            yield input_node
        except Exception as e:
            error = e
            # Record error
            error_node = self.record(
                node_type=NodeType.ERROR,
                agent_id=agent_id,
                data={
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                },
                metadata={"operation": operation},
                parent_ids=[input_node.id],
            )
            self.link(
                input_node.id,
                error_node.id,
                EdgeType.ERROR_PROPAGATION,
            )
            raise
        finally:
            self.current_node_id = prev_node_id

    # === Query Methods ===

    def trace_forward(
        self,
        node_id: str,
        max_depth: int = -1,
    ) -> list[Node]:
        """
        Find all nodes affected by a given node.

        Args:
            node_id: Starting node ID
            max_depth: Maximum traversal depth (-1 for unlimited)

        Returns:
            List of affected nodes
        """
        return self._graph.trace_forward(node_id, max_depth)

    def trace_backward(
        self,
        node_id: str,
        max_depth: int = -1,
    ) -> list[Node]:
        """
        Find all nodes that caused a given node.

        Args:
            node_id: Target node ID
            max_depth: Maximum traversal depth (-1 for unlimited)

        Returns:
            List of causal nodes
        """
        return self._graph.trace_backward(node_id, max_depth)

    def find_root_cause(self, node_id: str) -> list[Node]:
        """
        Find the root causes of a node.

        Args:
            node_id: The node to analyze

        Returns:
            List of root cause nodes
        """
        return self._graph.find_root_causes(node_id)

    def get_causal_chain(self, node_id: str) -> list[list[Node]]:
        """
        Get all causal chains leading to a node.

        Args:
            node_id: Target node ID

        Returns:
            List of paths, each path is a list of nodes
        """
        return self._graph.get_causal_chain(node_id)

    def find_errors(self) -> list[Node]:
        """Find all error nodes in the trace."""
        return self._graph.get_nodes_by_type(NodeType.ERROR)

    def analyze_error(self, error_node_id: str) -> dict[str, Any]:
        """
        Analyze an error node to find its causes.

        Args:
            error_node_id: ID of an error node

        Returns:
            Analysis result with root causes and causal chains
        """
        node = self._graph.get_node(error_node_id)
        if node is None:
            return {"error": "Node not found"}

        if node.type != NodeType.ERROR:
            return {"error": "Node is not an error node"}

        root_causes = self.find_root_cause(error_node_id)
        chains = self.get_causal_chain(error_node_id)

        return {
            "error_node": node.to_dict(),
            "root_causes": [n.to_dict() for n in root_causes],
            "causal_chains": [
                [n.to_dict() for n in chain]
                for chain in chains
            ],
            "chain_count": len(chains),
            "depth": max(len(chain) for chain in chains) if chains else 0,
        }

    # === Counterfactual Analysis ===

    def create_counterfactual(
        self,
        node_id: str,
        alternative_data: Any,
    ) -> CausalGraph:
        """
        Create a counterfactual graph with modified node data.

        Args:
            node_id: Node to modify
            alternative_data: Alternative data for the node

        Returns:
            New CausalGraph with the modification
        """
        return self._graph.create_counterfactual(node_id, alternative_data)

    def compare_runs(
        self,
        other_graph: CausalGraph,
    ) -> dict[str, Any]:
        """
        Compare this trace with another (e.g., counterfactual).

        Args:
            other_graph: The graph to compare with

        Returns:
            Comparison results
        """
        return self._graph.compare_with(other_graph)

    # === Serialization ===

    def to_json(self) -> str:
        """Export the trace as JSON."""
        return self._graph.to_json()

    def save(self, path: str) -> None:
        """Save the trace to a file."""
        with open(path, 'w') as f:
            f.write(self.to_json())

    @classmethod
    def load(cls, path: str) -> "Tracer":
        """Load a trace from a file."""
        with open(path, 'r') as f:
            graph = CausalGraph.from_json(f.read())

        tracer = cls(run_id=graph.run_id)
        tracer._graph = graph
        return tracer

    # === Statistics ===

    def get_statistics(self) -> dict[str, Any]:
        """Get statistics about the trace."""
        return self._graph.get_statistics()

    def __enter__(self) -> "Tracer":
        """Context manager entry."""
        return self.start()

    def __exit__(self, *args) -> None:
        """Context manager exit."""
        self.stop()
