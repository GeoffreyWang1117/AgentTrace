"""Base storage interface for causal graphs."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any

from agenttrace.core.graph import CausalGraph
from agenttrace.core.node import Node
from agenttrace.core.edge import Edge


class StorageBackend(ABC):
    """Abstract base class for storage backends."""

    @abstractmethod
    def save_graph(self, graph: CausalGraph) -> str:
        """
        Save a graph to storage.

        Args:
            graph: The graph to save

        Returns:
            The run_id of the saved graph
        """
        pass

    @abstractmethod
    def load_graph(self, run_id: str) -> CausalGraph | None:
        """
        Load a graph from storage.

        Args:
            run_id: The run_id of the graph to load

        Returns:
            The loaded graph, or None if not found
        """
        pass

    @abstractmethod
    def list_runs(
        self,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """
        List available runs.

        Args:
            start_time: Optional start time filter
            end_time: Optional end time filter
            limit: Maximum number of runs to return

        Returns:
            List of run metadata dictionaries
        """
        pass

    @abstractmethod
    def delete_run(self, run_id: str) -> bool:
        """
        Delete a run from storage.

        Args:
            run_id: The run_id to delete

        Returns:
            True if deleted, False if not found
        """
        pass

    @abstractmethod
    def append_node(self, run_id: str, node: Node) -> bool:
        """
        Append a node to an existing run (for streaming).

        Args:
            run_id: The run_id to append to
            node: The node to append

        Returns:
            True if successful
        """
        pass

    @abstractmethod
    def append_edge(self, run_id: str, edge: Edge) -> bool:
        """
        Append an edge to an existing run (for streaming).

        Args:
            run_id: The run_id to append to
            edge: The edge to append

        Returns:
            True if successful
        """
        pass

    def query_nodes_at_time(
        self,
        run_id: str,
        timestamp: datetime,
    ) -> list[Node]:
        """
        Query nodes that existed at a specific point in time.

        Args:
            run_id: The run_id to query
            timestamp: The point in time to query

        Returns:
            List of nodes that existed at that time
        """
        graph = self.load_graph(run_id)
        if graph is None:
            return []

        return [n for n in graph if n.timestamp <= timestamp]

    def get_statistics(self, run_id: str) -> dict[str, Any] | None:
        """
        Get statistics for a run.

        Args:
            run_id: The run_id to analyze

        Returns:
            Statistics dictionary, or None if not found
        """
        graph = self.load_graph(run_id)
        if graph is None:
            return None
        return graph.get_statistics()
