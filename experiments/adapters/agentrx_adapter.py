"""
Adapter: AgentRx benchmark (Magentic-One split) → AgentTrace CausalGraph format.

AgentRx (Microsoft Research, 2026) annotates every failure in each failed trajectory and marks one
of them as the root cause. This adapter loads the Magentic-One trajectories that have ground truth:
`data/ground_truth/magentic_one_ground_truth.json` entries matched by `trajectory_id` to
`data/magentic_dataset/<trajectory_id>.json`. In the release we checked, 44 traces have ground truth
and 43 of them have a resolvable root cause.

Key format details:
  - Trajectories are lists of {"content": ..., "role": ...}; role carries the agent name
    (e.g. "WebSurfer", "Orchestrator (-> WebSurfer)").
  - GT `step_number` is 1-based: failed_agent matches the role at index step_number-1 in 293/295
    annotated failures.
  - The ground-truth step is the failure named by `root_cause.failure_id`. Traces whose root cause
    cannot be resolved are skipped, never filled in.
  - An explicit outcome node (agent "__outcome__") is appended and used as the error node.

Correction (2026-10): earlier versions of this adapter (a) paired the category-named example
trajectories in `trajectories/magentic-one/` with ground truth through a fallback that always
returned the first GT entry, so all of them carried one label; (b) fabricated placeholder traces for
the τ-bench ground truth, whose trajectories are not in the release, with the root-cause text on the
root-cause node only; (c) used the earliest failure rather than the root cause, as a 0-based index.
Results computed with those versions are not AgentRx results.
"""

import json
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from agenttrace.core.edge import Edge, EdgeType
from agenttrace.core.graph import CausalGraph
from agenttrace.core.node import Node, NodeType


def normalize_agent_name(role: str) -> str:
    """"Orchestrator (-> WebSurfer)" → "Orchestrator"."""
    m = re.match(r"^(\w+)", role.strip())
    return m.group(1) if m else role.strip()


def root_cause_failure(gt_entry: dict):
    """The failure record named by root_cause.failure_id, or None."""
    rid = (gt_entry.get("root_cause") or {}).get("failure_id")
    hits = [f for f in gt_entry.get("failures", []) if f.get("failure_id") == rid]
    return hits[0] if hits else None


def convert_trajectory(trajectory: list, gt_entry: dict, trace_id: str):
    """Convert one AgentRx trajectory + its GT entry. Returns None if the root cause is unresolved.

    Returns dict with graph, root_cause_node_id, error_node_id (the outcome node), gt_agent, gt_step
    (0-based), n_nodes (real steps), n_agents.
    """
    if not trajectory or len(trajectory) < 2:
        return None
    rc = root_cause_failure(gt_entry)
    if rc is None:
        return None
    gt_step = int(rc["step_number"]) - 1
    if not 0 <= gt_step < len(trajectory):
        return None

    graph = CausalGraph(run_id=f"agentrx_{trace_id}")
    nodes = []
    t0 = datetime(2026, 1, 1)
    for i, entry in enumerate(trajectory):
        role = str(entry.get("role", f"agent_{i}"))
        node = Node(type=NodeType.DECISION, agent_id=normalize_agent_name(role),
                    data={"step": i, "action": "respond", "content": str(entry.get("content", ""))[:500],
                          "role": role})
        node.timestamp = t0 + timedelta(seconds=i)
        if nodes:
            node.parent_ids = [nodes[-1].id]
        graph.add_node(node)
        nodes.append(node)

    for a, b in zip(nodes, nodes[1:]):
        if a.agent_id != b.agent_id:
            try:
                graph.add_edge(Edge(source_id=a.id, target_id=b.id, type=EdgeType.TRIGGER_RESPONSE,
                                    confidence=0.85))
            except ValueError:
                pass

    outcome = Node(type=NodeType.AGENT_OUTPUT, agent_id="__outcome__",
                   data={"step": len(nodes), "action": "outcome", "content": "", "role": "outcome"})
    outcome.timestamp = t0 + timedelta(seconds=len(nodes))
    outcome.parent_ids = [nodes[-1].id]
    graph.add_node(outcome)

    return {
        "graph": graph,
        "root_cause_node_id": nodes[gt_step].id,
        "error_node_id": outcome.id,
        "gt_agent": rc.get("failed_agent", ""),
        "gt_step": gt_step,
        "n_nodes": len(nodes),
        "n_agents": len({n.agent_id for n in nodes}),
    }


def load_all_agentrx(data_dir: str = None):
    """Load the AgentRx Magentic-One traces that have ground truth.

    Returns list of (trace_id, trace_data_dict, gt_dict); gt_dict['mistake_step'] is 0-based.
    """
    if data_dir is None:
        data_dir = str(Path(__file__).parent.parent.parent / "data/external/agentrx/repo")
    base = Path(data_dir) / "data"
    gt_file = base / "ground_truth" / "magentic_one_ground_truth.json"
    traj_dir = base / "magentic_dataset"
    if not gt_file.exists() or not traj_dir.exists():
        return []

    results = []
    for entry in json.load(open(gt_file, encoding="utf-8")):
        tid = str(entry.get("trajectory_id", ""))
        tf = traj_dir / f"{tid}.json"
        if not tf.exists():
            continue
        converted = convert_trajectory(json.load(open(tf, encoding="utf-8")), entry, tid)
        if converted is None:
            continue
        sid = f"mag_{tid}"
        td = {
            "scenario_id": sid,
            "trace_json": converted["graph"].to_json(),
            "root_cause_node_id": converted["root_cause_node_id"],
            "error_node_id": converted["error_node_id"],
        }
        gt = {"mistake_agent": converted["gt_agent"], "mistake_step": str(converted["gt_step"])}
        results.append((sid, td, gt))
    return results


if __name__ == "__main__":
    traces = load_all_agentrx()
    print(f"Loaded {len(traces)} AgentRx traces")
    for sid, td, gt in traces[:5]:
        trace = json.loads(td["trace_json"])
        print(f"  {sid}: {len(trace['nodes'])} nodes, gt_agent={gt['mistake_agent']}, gt_step={gt['mistake_step']}")
