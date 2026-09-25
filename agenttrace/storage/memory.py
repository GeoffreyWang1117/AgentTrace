"""In-memory storage backend for causal graphs."""

from __future__ import annotations

from datetime import datetime
from typing import Any
from collections import OrderedDict
import threading

from agenttrace.core.graph import CausalGraph
from agenttrace.core.node import Node
from agenttrace.core.edge import Edge
from agenttrace.storage.base import StorageBackend


class MemoryStorage(StorageBackend):
    """
    In-memory storage for causal graphs.

    Useful for development, testing, and short-lived traces.
    Data is lost when the process exits.
    """

    def __init__(self, max_runs: int = 100):
        """
        Initialize memory storage.

        Args:
            max_runs: Maximum number of runs to keep in memory
        """
        self._graphs: OrderedDict[str, CausalGraph] = OrderedDict()
        self._metadata: dict[str, dict[str, Any]] = {}
        self._max_runs = max_runs
        self._lock = threading.Lock()

    def save_graph(self, graph: CausalGraph) -> str:
        """Save a graph to memory."""
        with self._lock:
            run_id = graph.run_id

            # Enforce max runs limit
            while len(self._graphs) >= self._max_runs:
                oldest = next(iter(self._graphs))
                del self._graphs[oldest]
                del self._metadata[oldest]

            self._graphs[run_id] = graph
            self._metadata[run_id] = {
                "run_id": run_id,
                "created_at": datetime.now().isoformat(),
                "node_count": graph.node_count,
                "edge_count": graph.edge_count,
            }

            return run_id

    def load_graph(self, run_id: str) -> CausalGraph | None:
        """Load a graph from memory."""
        return self._graphs.get(run_id)

    def list_runs(
        self,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """List available runs."""
        results = []

        for run_id, meta in self._metadata.items():
            created = datetime.fromisoformat(meta["created_at"])

            if start_time and created < start_time:
                continue
            if end_time and created > end_time:
                continue

            results.append(meta.copy())

            if len(results) >= limit:
                break

        return results

    def delete_run(self, run_id: str) -> bool:
        """Delete a run from memory."""
        with self._lock:
            if run_id in self._graphs:
                del self._graphs[run_id]
                del self._metadata[run_id]
                return True
            return False

    def append_node(self, run_id: str, node: Node) -> bool:
        """Append a node to an existing run."""
        with self._lock:
            if run_id not in self._graphs:
                return False

            self._graphs[run_id].add_node(node)
            self._metadata[run_id]["node_count"] += 1
            return True

    def append_edge(self, run_id: str, edge: Edge) -> bool:
        """Append an edge to an existing run."""
        with self._lock:
            if run_id not in self._graphs:
                return False

            try:
                self._graphs[run_id].add_edge(edge)
                self._metadata[run_id]["edge_count"] += 1
                return True
            except ValueError:
                return False

    def clear(self) -> None:
        """Clear all stored data."""
        with self._lock:
            self._graphs.clear()
            self._metadata.clear()

    def get_run_count(self) -> int:
        """Get the number of stored runs."""
        return len(self._graphs)
