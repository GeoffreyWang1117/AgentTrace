"""
Temporal storage backend for causal graphs.

Provides time-point queries and efficient storage for large traces.
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any
import threading

from agenttrace.core.graph import CausalGraph
from agenttrace.core.node import Node
from agenttrace.core.edge import Edge
from agenttrace.storage.base import StorageBackend


class TemporalStorage(StorageBackend):
    """
    File-based temporal storage for causal graphs.

    Supports:
    - Time-point queries (get state at specific timestamp)
    - Efficient append-only storage for streaming traces
    - Incremental snapshots for fast loading
    """

    def __init__(self, base_path: str | Path, snapshot_interval: int = 100):
        """
        Initialize temporal storage.

        Args:
            base_path: Base directory for storage
            snapshot_interval: Create a snapshot every N nodes
        """
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.snapshot_interval = snapshot_interval
        self._lock = threading.Lock()

        # In-memory index
        self._index: dict[str, dict[str, Any]] = {}
        self._load_index()

    def _load_index(self) -> None:
        """Load the run index from disk."""
        index_path = self.base_path / "index.json"
        if index_path.exists():
            with open(index_path, "r") as f:
                self._index = json.load(f)

    def _save_index(self) -> None:
        """Save the run index to disk."""
        index_path = self.base_path / "index.json"
        with open(index_path, "w") as f:
            json.dump(self._index, f, indent=2)

    def _get_run_path(self, run_id: str) -> Path:
        """Get the directory path for a run."""
        return self.base_path / run_id

    def save_graph(self, graph: CausalGraph) -> str:
        """Save a graph to storage."""
        with self._lock:
            run_id = graph.run_id
            run_path = self._get_run_path(run_id)
            run_path.mkdir(parents=True, exist_ok=True)

            # Save full graph
            graph_path = run_path / "graph.json"
            with open(graph_path, "w") as f:
                f.write(graph.to_json())

            # Create time index
            self._create_time_index(graph, run_path)

            # Update index
            self._index[run_id] = {
                "run_id": run_id,
                "created_at": datetime.now().isoformat(),
                "node_count": graph.node_count,
                "edge_count": graph.edge_count,
                "path": str(run_path),
            }
            self._save_index()

            return run_id

    def _create_time_index(self, graph: CausalGraph, run_path: Path) -> None:
        """Create a time-based index for efficient temporal queries."""
        time_index: list[dict[str, Any]] = []

        for node in sorted(graph, key=lambda n: n.timestamp):
            time_index.append(
                {
                    "node_id": node.id,
                    "timestamp": node.timestamp.isoformat(),
                    "type": node.type.value,
                    "agent_id": node.agent_id,
                }
            )

        index_path = run_path / "time_index.json"
        with open(index_path, "w") as f:
            json.dump(time_index, f, indent=2)

    def load_graph(self, run_id: str) -> CausalGraph | None:
        """Load a graph from storage."""
        run_path = self._get_run_path(run_id)
        graph_path = run_path / "graph.json"

        if not graph_path.exists():
            return None

        with open(graph_path, "r") as f:
            return CausalGraph.from_json(f.read())

    def list_runs(
        self,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """List available runs."""
        results = []

        for run_id, meta in self._index.items():
            created = datetime.fromisoformat(meta["created_at"])

            if start_time and created < start_time:
                continue
            if end_time and created > end_time:
                continue

            results.append(meta.copy())

            if len(results) >= limit:
                break

        # Sort by creation time, newest first
        results.sort(
            key=lambda x: datetime.fromisoformat(x["created_at"]),
            reverse=True,
        )

        return results

    def delete_run(self, run_id: str) -> bool:
        """Delete a run from storage."""
        with self._lock:
            if run_id not in self._index:
                return False

            run_path = self._get_run_path(run_id)
            if run_path.exists():
                shutil.rmtree(run_path)

            del self._index[run_id]
            self._save_index()

            return True

    def append_node(self, run_id: str, node: Node) -> bool:
        """Append a node to an existing run (streaming mode)."""
        with self._lock:
            run_path = self._get_run_path(run_id)

            # Initialize run if needed
            if run_id not in self._index:
                run_path.mkdir(parents=True, exist_ok=True)
                self._index[run_id] = {
                    "run_id": run_id,
                    "created_at": datetime.now().isoformat(),
                    "node_count": 0,
                    "edge_count": 0,
                    "path": str(run_path),
                }

            # Append to nodes log
            nodes_log = run_path / "nodes.jsonl"
            with open(nodes_log, "a") as f:
                f.write(json.dumps(node.to_dict(), default=str) + "\n")

            # Update count
            self._index[run_id]["node_count"] += 1

            # Create snapshot if needed
            if self._index[run_id]["node_count"] % self.snapshot_interval == 0:
                self._create_snapshot(run_id)

            self._save_index()
            return True

    def append_edge(self, run_id: str, edge: Edge) -> bool:
        """Append an edge to an existing run (streaming mode)."""
        with self._lock:
            if run_id not in self._index:
                return False

            run_path = self._get_run_path(run_id)

            # Append to edges log
            edges_log = run_path / "edges.jsonl"
            with open(edges_log, "a") as f:
                f.write(json.dumps(edge.to_dict(), default=str) + "\n")

            self._index[run_id]["edge_count"] += 1
            self._save_index()
            return True

    def _create_snapshot(self, run_id: str) -> None:
        """Create a snapshot of the current state."""
        run_path = self._get_run_path(run_id)
        snapshot_count = self._index[run_id]["node_count"] // self.snapshot_interval

        # Load from log files
        graph = self._load_from_logs(run_path, run_id)
        if graph:
            snapshot_path = run_path / f"snapshot_{snapshot_count:04d}.json"
            with open(snapshot_path, "w") as f:
                f.write(graph.to_json())

    def _load_from_logs(self, run_path: Path, run_id: str) -> CausalGraph | None:
        """Load a graph from log files."""
        graph = CausalGraph(run_id=run_id)

        # Load nodes
        nodes_log = run_path / "nodes.jsonl"
        if nodes_log.exists():
            with open(nodes_log, "r") as f:
                for line in f:
                    if line.strip():
                        node_dict = json.loads(line)
                        node = Node.from_dict(node_dict)
                        graph._nodes[node.id] = node
                        graph._graph.add_node(node.id, data=node)
                        graph._nodes_by_agent[node.agent_id].append(node.id)
                        graph._nodes_by_type[node.type].append(node.id)

        # Load edges
        edges_log = run_path / "edges.jsonl"
        if edges_log.exists():
            with open(edges_log, "r") as f:
                for line in f:
                    if line.strip():
                        edge_dict = json.loads(line)
                        edge = Edge.from_dict(edge_dict)
                        try:
                            graph._edges[edge.id] = edge
                            graph._graph.add_edge(
                                edge.source_id,
                                edge.target_id,
                                id=edge.id,
                                data=edge,
                            )
                        except Exception:
                            pass  # Skip invalid edges

        return graph

    def query_nodes_at_time(
        self,
        run_id: str,
        timestamp: datetime,
    ) -> list[Node]:
        """Query nodes that existed at a specific point in time."""
        run_path = self._get_run_path(run_id)
        time_index_path = run_path / "time_index.json"

        # If we have a time index, use it for efficiency
        if time_index_path.exists():
            with open(time_index_path, "r") as f:
                time_index = json.load(f)

            # Find nodes up to timestamp
            valid_node_ids = set()
            for entry in time_index:
                node_time = datetime.fromisoformat(entry["timestamp"])
                if node_time <= timestamp:
                    valid_node_ids.add(entry["node_id"])
                else:
                    break

            # Load full graph and filter
            graph = self.load_graph(run_id)
            if graph:
                return [n for n in graph if n.id in valid_node_ids]

        # Fall back to full load and filter
        return super().query_nodes_at_time(run_id, timestamp)

    def get_time_range(self, run_id: str) -> tuple[datetime, datetime] | None:
        """Get the time range covered by a run."""
        run_path = self._get_run_path(run_id)
        time_index_path = run_path / "time_index.json"

        if not time_index_path.exists():
            return None

        with open(time_index_path, "r") as f:
            time_index = json.load(f)

        if not time_index:
            return None

        start = datetime.fromisoformat(time_index[0]["timestamp"])
        end = datetime.fromisoformat(time_index[-1]["timestamp"])

        return (start, end)

    def cleanup_old_runs(self, max_age_days: int = 30) -> int:
        """
        Clean up runs older than max_age_days.

        Returns:
            Number of runs deleted
        """
        cutoff = datetime.now().timestamp() - (max_age_days * 24 * 60 * 60)
        deleted = 0

        with self._lock:
            to_delete = []
            for run_id, meta in self._index.items():
                created = datetime.fromisoformat(meta["created_at"])
                if created.timestamp() < cutoff:
                    to_delete.append(run_id)

            for run_id in to_delete:
                self.delete_run(run_id)
                deleted += 1

        return deleted
