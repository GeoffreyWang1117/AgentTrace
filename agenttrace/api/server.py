"""
FastAPI server for AgentTrace debugging UI.

Provides REST API and WebSocket endpoints for interactive debugging.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel

from agenttrace.core.node import Node, NodeType
from agenttrace.core.edge import Edge
from agenttrace.storage.memory import MemoryStorage
from agenttrace.storage.temporal_db import TemporalStorage
from agenttrace.inference.engine import InferenceEngine


# Request/Response models
class TraceForwardRequest(BaseModel):
    node_id: str
    max_depth: int = -1


class TraceBackwardRequest(BaseModel):
    node_id: str
    max_depth: int = -1


class CounterfactualRequest(BaseModel):
    node_id: str
    alternative_data: Any


class NodeResponse(BaseModel):
    id: str
    type: str
    agent_id: str
    timestamp: str
    data: Any
    metadata: dict
    parent_ids: list[str]


class EdgeResponse(BaseModel):
    id: str
    source_id: str
    target_id: str
    type: str
    confidence: float
    metadata: dict


class GraphResponse(BaseModel):
    run_id: str
    nodes: list[NodeResponse]
    edges: list[EdgeResponse]
    statistics: dict


class RunListResponse(BaseModel):
    runs: list[dict]
    total: int


# Global state
_storage: MemoryStorage | TemporalStorage | None = None
_connected_clients: set[WebSocket] = set()


def get_storage():
    """Get the storage backend."""
    global _storage
    if _storage is None:
        _storage = MemoryStorage()
    return _storage


def set_storage(storage: MemoryStorage | TemporalStorage):
    """Set the storage backend."""
    global _storage
    _storage = storage


def create_app(
    storage: MemoryStorage | TemporalStorage | None = None,
    static_dir: str | Path | None = None,
) -> FastAPI:
    """
    Create the FastAPI application.

    Args:
        storage: Optional storage backend to use
        static_dir: Optional directory for static files (UI)

    Returns:
        The FastAPI application
    """
    app = FastAPI(
        title="AgentTrace",
        description="Multi-Agent Causal Tracing System",
        version="0.1.0",
    )

    # CORS middleware for development
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    if storage:
        set_storage(storage)

    # Mount static files if provided
    if static_dir:
        static_path = Path(static_dir)
        if static_path.exists():
            app.mount("/static", StaticFiles(directory=str(static_path)), name="static")

    inference_engine = InferenceEngine()

    # === Health Check ===

    @app.get("/health")
    async def health_check():
        return {"status": "healthy", "timestamp": datetime.now().isoformat()}

    # === Run Management ===

    @app.get("/api/runs", response_model=RunListResponse)
    async def list_runs(
        limit: int = Query(default=50, le=200),
        offset: int = Query(default=0, ge=0),
    ):
        """List all available trace runs."""
        storage = get_storage()
        runs = storage.list_runs(limit=limit + offset)
        return {
            "runs": runs[offset : offset + limit],
            "total": len(runs),
        }

    @app.get("/api/runs/{run_id}")
    async def get_run(run_id: str):
        """Get details of a specific run."""
        storage = get_storage()
        graph = storage.load_graph(run_id)
        if graph is None:
            raise HTTPException(status_code=404, detail="Run not found")

        return {
            "run_id": graph.run_id,
            "statistics": graph.get_statistics(),
        }

    @app.delete("/api/runs/{run_id}")
    async def delete_run(run_id: str):
        """Delete a run."""
        storage = get_storage()
        if not storage.delete_run(run_id):
            raise HTTPException(status_code=404, detail="Run not found")
        return {"status": "deleted", "run_id": run_id}

    # === Graph Operations ===

    @app.get("/api/runs/{run_id}/graph", response_model=GraphResponse)
    async def get_graph(run_id: str):
        """Get the full causal graph for a run."""
        storage = get_storage()
        graph = storage.load_graph(run_id)
        if graph is None:
            raise HTTPException(status_code=404, detail="Run not found")

        return {
            "run_id": graph.run_id,
            "nodes": [_node_to_response(n) for n in graph],
            "edges": [_edge_to_response(e) for e in graph._edges.values()],
            "statistics": graph.get_statistics(),
        }

    @app.get("/api/runs/{run_id}/nodes/{node_id}")
    async def get_node(run_id: str, node_id: str):
        """Get a specific node."""
        storage = get_storage()
        graph = storage.load_graph(run_id)
        if graph is None:
            raise HTTPException(status_code=404, detail="Run not found")

        node = graph.get_node(node_id)
        if node is None:
            raise HTTPException(status_code=404, detail="Node not found")

        return _node_to_response(node)

    # === Query Operations ===

    @app.post("/api/runs/{run_id}/trace/forward")
    async def trace_forward(run_id: str, request: TraceForwardRequest):
        """Forward trace: Find all nodes affected by a given node."""
        storage = get_storage()
        graph = storage.load_graph(run_id)
        if graph is None:
            raise HTTPException(status_code=404, detail="Run not found")

        affected = graph.trace_forward(request.node_id, request.max_depth)
        return {
            "source_node_id": request.node_id,
            "affected_nodes": [_node_to_response(n) for n in affected],
            "count": len(affected),
        }

    @app.post("/api/runs/{run_id}/trace/backward")
    async def trace_backward(run_id: str, request: TraceBackwardRequest):
        """Backward trace: Find all nodes that caused a given node."""
        storage = get_storage()
        graph = storage.load_graph(run_id)
        if graph is None:
            raise HTTPException(status_code=404, detail="Run not found")

        causes = graph.trace_backward(request.node_id, request.max_depth)
        return {
            "target_node_id": request.node_id,
            "cause_nodes": [_node_to_response(n) for n in causes],
            "count": len(causes),
        }

    @app.get("/api/runs/{run_id}/root-causes/{node_id}")
    async def find_root_causes(run_id: str, node_id: str):
        """Find root causes of a node."""
        storage = get_storage()
        graph = storage.load_graph(run_id)
        if graph is None:
            raise HTTPException(status_code=404, detail="Run not found")

        root_causes = graph.find_root_causes(node_id)
        return {
            "target_node_id": node_id,
            "root_causes": [_node_to_response(n) for n in root_causes],
            "count": len(root_causes),
        }

    @app.get("/api/runs/{run_id}/causal-chains/{node_id}")
    async def get_causal_chains(run_id: str, node_id: str):
        """Get all causal chains leading to a node."""
        storage = get_storage()
        graph = storage.load_graph(run_id)
        if graph is None:
            raise HTTPException(status_code=404, detail="Run not found")

        chains = graph.get_causal_chain(node_id)
        return {
            "target_node_id": node_id,
            "chains": [[_node_to_response(n) for n in chain] for chain in chains],
            "chain_count": len(chains),
        }

    @app.get("/api/runs/{run_id}/path")
    async def find_path(
        run_id: str,
        source_id: str = Query(...),
        target_id: str = Query(...),
    ):
        """Find the causal path between two nodes."""
        storage = get_storage()
        graph = storage.load_graph(run_id)
        if graph is None:
            raise HTTPException(status_code=404, detail="Run not found")

        path = graph.find_path(source_id, target_id)
        if path is None:
            return {"path": None, "exists": False}

        return {
            "path": [_node_to_response(n) for n in path],
            "exists": True,
            "length": len(path),
        }

    # === Error Analysis ===

    @app.get("/api/runs/{run_id}/errors")
    async def get_errors(run_id: str):
        """Get all error nodes in a run."""
        storage = get_storage()
        graph = storage.load_graph(run_id)
        if graph is None:
            raise HTTPException(status_code=404, detail="Run not found")

        errors = graph.get_nodes_by_type(NodeType.ERROR)
        return {
            "errors": [_node_to_response(n) for n in errors],
            "count": len(errors),
        }

    @app.get("/api/runs/{run_id}/analyze-error/{error_node_id}")
    async def analyze_error(run_id: str, error_node_id: str):
        """Analyze an error node to find its causes."""
        storage = get_storage()
        graph = storage.load_graph(run_id)
        if graph is None:
            raise HTTPException(status_code=404, detail="Run not found")

        node = graph.get_node(error_node_id)
        if node is None:
            raise HTTPException(status_code=404, detail="Error node not found")
        if node.type != NodeType.ERROR:
            raise HTTPException(status_code=400, detail="Node is not an error node")

        # Get analysis
        root_causes = graph.find_root_causes(error_node_id)
        chains = graph.get_causal_chain(error_node_id)

        # Get suggestions from inference engine
        suggestions = inference_engine.suggest_likely_causes(graph, error_node_id)

        return {
            "error_node": _node_to_response(node),
            "root_causes": [_node_to_response(n) for n in root_causes],
            "causal_chains": [
                [_node_to_response(n) for n in chain]
                for chain in chains[:5]  # Limit chains
            ],
            "likely_causes": suggestions,
        }

    # === Counterfactual Analysis ===

    @app.post("/api/runs/{run_id}/counterfactual")
    async def create_counterfactual(run_id: str, request: CounterfactualRequest):
        """Create a counterfactual analysis."""
        storage = get_storage()
        graph = storage.load_graph(run_id)
        if graph is None:
            raise HTTPException(status_code=404, detail="Run not found")

        try:
            cf_graph = graph.create_counterfactual(
                request.node_id,
                request.alternative_data,
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

        # Save counterfactual graph
        cf_run_id = storage.save_graph(cf_graph)

        # Compare
        comparison = graph.compare_with(cf_graph)

        return {
            "original_run_id": run_id,
            "counterfactual_run_id": cf_run_id,
            "modified_node_id": request.node_id,
            "comparison": comparison,
        }

    # === Inference ===

    @app.post("/api/runs/{run_id}/infer")
    async def infer_relationships(run_id: str):
        """Run causal inference on a graph."""
        storage = get_storage()
        graph = storage.load_graph(run_id)
        if graph is None:
            raise HTTPException(status_code=404, detail="Run not found")

        inferred = inference_engine.infer_relationships(graph, apply_to_graph=True)

        # Save updated graph
        storage.save_graph(graph)

        return {
            "inferred_edges": [_edge_to_response(e) for e in inferred],
            "count": len(inferred),
        }

    # === Time Travel ===

    @app.get("/api/runs/{run_id}/at-time")
    async def get_state_at_time(
        run_id: str,
        timestamp: str = Query(..., description="ISO format timestamp"),
    ):
        """Get the state of the graph at a specific point in time."""
        storage = get_storage()

        try:
            ts = datetime.fromisoformat(timestamp)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid timestamp format")

        nodes = storage.query_nodes_at_time(run_id, ts)

        return {
            "timestamp": timestamp,
            "nodes": [_node_to_response(n) for n in nodes],
            "count": len(nodes),
        }

    # === WebSocket for real-time updates ===

    @app.websocket("/ws/{run_id}")
    async def websocket_endpoint(websocket: WebSocket, run_id: str):
        """WebSocket endpoint for real-time trace updates."""
        await websocket.accept()
        _connected_clients.add(websocket)

        try:
            while True:
                # Keep connection alive and handle commands
                data = await websocket.receive_json()

                if data.get("type") == "subscribe":
                    # Client wants updates for this run
                    await websocket.send_json(
                        {
                            "type": "subscribed",
                            "run_id": run_id,
                        }
                    )

                elif data.get("type") == "ping":
                    await websocket.send_json({"type": "pong"})

        except WebSocketDisconnect:
            _connected_clients.discard(websocket)

    # === UI Routes ===

    @app.get("/", response_class=HTMLResponse)
    async def root():
        """Serve the main UI."""
        ui_path = Path(__file__).parent.parent.parent / "ui" / "index.html"
        if ui_path.exists():
            return FileResponse(ui_path)
        return HTMLResponse("""
        <html>
            <head><title>AgentTrace</title></head>
            <body>
                <h1>AgentTrace API</h1>
                <p>API is running. Visit <a href="/docs">/docs</a> for API documentation.</p>
            </body>
        </html>
        """)

    return app


def _node_to_response(node: Node) -> dict:
    """Convert a Node to a response dictionary."""
    return {
        "id": node.id,
        "type": node.type.value,
        "agent_id": node.agent_id,
        "timestamp": node.timestamp.isoformat(),
        "data": node.data,
        "metadata": node.metadata,
        "parent_ids": node.parent_ids,
    }


def _edge_to_response(edge: Edge) -> dict:
    """Convert an Edge to a response dictionary."""
    return {
        "id": edge.id,
        "source_id": edge.source_id,
        "target_id": edge.target_id,
        "type": edge.type.value,
        "confidence": edge.confidence,
        "metadata": edge.metadata,
    }


async def broadcast_update(run_id: str, update: dict):
    """Broadcast an update to all connected WebSocket clients."""
    for client in _connected_clients.copy():
        try:
            await client.send_json(
                {
                    "type": "update",
                    "run_id": run_id,
                    **update,
                }
            )
        except Exception:
            _connected_clients.discard(client)


def run_server(
    host: str = "127.0.0.1",
    port: int = 8000,
    storage: MemoryStorage | TemporalStorage | None = None,
):
    """
    Run the AgentTrace server.

    Args:
        host: Host to bind to
        port: Port to bind to
        storage: Optional storage backend
    """
    import uvicorn

    app = create_app(storage=storage)
    uvicorn.run(app, host=host, port=port)
