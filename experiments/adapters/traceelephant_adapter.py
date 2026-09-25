"""
Adapter: TraceElephant ("Seeing the Whole Elephant", ACL 2026) -> AgentTrace.

TraceElephant ships 220 annotated multi-agent failure traces across five
subsets, each run directory holding two files:

    trace_metadata.json : task_id, task_instruction, system_name,
                          agent_configuration, mistake_agent, mistake_step,
                          mistake_reason, ground_truth
    step_records.json   : [{step_id, agent_id, agent_name, input, output,
                            tool_logs}, ...]

Ground truth is Who&When-shaped (mistake_agent + mistake_step), so this adapter
mirrors who_and_when_adapter.py. Two deliberate differences:

1. ORCHESTRATOR-AWARE EDGES. Captain-Agent and SWE-agent runs interleave a
   dispatching agent with several experts. Building only i -> i+1 sequential
   edges flattens that into a chain, which is exactly the artefact that made
   chain_frac saturate at 1.0 on our earlier adapters. When a single agent
   accounts for >= `hub_min_share` of the steps AND at least two other agents
   are present, we additionally connect each dispatch to its matching return,
   producing a genuine DAG. Set `topology='sequential'` to disable and get the
   old chain behaviour for comparison.

2. EXPLICIT GT PROVENANCE. `gt_source` records how mistake_step was mapped to a
   node ('exact' / 'clamped' / 'agent_fallback'), and `gt_is_exact` flags it, so
   a silent fall-through can never masquerade as an annotation. This is the
   defect class found in trail_adapter on 2026-08-05.

License: CC BY 4.0.
"""

import json
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from agenttrace.core.graph import CausalGraph
from agenttrace.core.node import Node, NodeType
from agenttrace.core.edge import Edge, EdgeType

SUBSETS = [
    "captain-runs-gaia",
    "captain-runs-assistantbench",
    "magentic-runs-gaia",
    "magentic-runs-assistant-bench",
    "swe-agent-runs-swe-bench",
]

DEFAULT_ROOT = "data/external/traceelephant/extracted/data"


def _infer_node_type(rec: dict) -> NodeType:
    name = str(rec.get("agent_name", "")).lower()
    out = str(rec.get("output", ""))[:200].lower()
    if rec.get("tool_logs"):
        return NodeType.TOOL_CALL
    if "orchestrat" in name or "manager" in name:
        return NodeType.AGENT_INPUT
    if any(k in out for k in ("error", "fail", "cannot", "unable", "traceback")):
        return NodeType.AGENT_OUTPUT
    return NodeType.DECISION


def _content_of(rec: dict) -> str:
    parts = []
    for k in ("input", "output"):
        v = rec.get(k)
        if v is None:
            continue
        if isinstance(v, (dict, list)):
            v = json.dumps(v, ensure_ascii=False)
        v = str(v).strip()
        if v:
            parts.append(f"{k}: {v}")
    tl = rec.get("tool_logs")
    if tl:
        s = tl if isinstance(tl, str) else json.dumps(tl, ensure_ascii=False)
        parts.append(f"tool_logs: {str(s)[:2000]}")
    return " | ".join(parts)


def convert_run(meta: dict, steps: list, run_id: str,
                topology: str = "hub", hub_min_share: float = 0.35) -> Optional[dict]:
    """Convert one TraceElephant run directory to AgentTrace format."""
    if not steps or len(steps) < 2:
        return None
    mistake_step = meta.get("mistake_step")
    mistake_agent = str(meta.get("mistake_agent", "") or "")
    if mistake_step is None or str(mistake_step) == "":
        return None
    try:
        mistake_step = int(mistake_step)
    except (TypeError, ValueError):
        return None

    graph = CausalGraph(run_id=f"telephant_{run_id}")
    base_time = datetime.now()
    nodes = []

    for i, rec in enumerate(steps):
        agent_id = str(rec.get("agent_name") or rec.get("agent_id") or f"agent_{i}")
        node = Node(
            type=_infer_node_type(rec),
            agent_id=agent_id,
            data={
                "step": i + 1,
                "content": _content_of(rec),
                "action": "respond",
                "step_id": rec.get("step_id"),
            },
            metadata={"source": "traceelephant", "run_id": run_id},
        )
        node.timestamp = base_time + timedelta(milliseconds=i * 100)
        if i > 0:
            node.parent_ids = [nodes[i - 1].id]
        graph.add_node(node)
        nodes.append(node)

    existing = {(e.source_id, e.target_id) for e in graph._edges.values()}

    def _add(src, tgt, etype, conf, meta_d):
        if (src, tgt) in existing or src == tgt:
            return
        try:
            graph.add_edge(Edge(source_id=src, target_id=tgt, type=etype,
                                confidence=conf, metadata=meta_d))
            existing.add((src, tgt))
        except ValueError:
            pass

    # agent-handoff edges (same as who_and_when)
    for i in range(len(nodes) - 1):
        if nodes[i].agent_id != nodes[i + 1].agent_id:
            _add(nodes[i].id, nodes[i + 1].id, EdgeType.TRIGGER_RESPONSE, 0.85,
                 {"type": "agent_handoff"})

    # orchestrator-aware dispatch/return edges -> breaks the linear backbone
    agent_counts = Counter(n.agent_id for n in nodes)
    hub_agent = None
    if topology == "hub" and len(agent_counts) >= 3:
        cand, cnt = agent_counts.most_common(1)[0]
        if cnt / len(nodes) >= hub_min_share:
            hub_agent = cand

    n_dispatch = 0
    if hub_agent is not None:
        # every hub step that precedes a worker step dispatches to it; the
        # worker's last consecutive step returns to the next hub step.
        i = 0
        while i < len(nodes) - 1:
            if nodes[i].agent_id == hub_agent and nodes[i + 1].agent_id != hub_agent:
                j = i + 1
                while j + 1 < len(nodes) and nodes[j + 1].agent_id == nodes[i + 1].agent_id:
                    j += 1
                _add(nodes[i].id, nodes[j].id, EdgeType.TRIGGER_RESPONSE, 0.8,
                     {"type": "dispatch"})
                n_dispatch += 1
                k = j + 1
                while k < len(nodes) and nodes[k].agent_id != hub_agent:
                    k += 1
                if k < len(nodes):
                    _add(nodes[j].id, nodes[k].id, EdgeType.TRIGGER_RESPONSE, 0.8,
                         {"type": "return"})
                    n_dispatch += 1
                i = j
            i += 1

    # ── ground truth, with explicit provenance ──────────────────────────
    rc_idx, gt_source = None, "unresolved"
    if 1 <= mistake_step <= len(nodes):
        rc_idx, gt_source = mistake_step - 1, "exact"
    elif mistake_agent:
        for i, n in enumerate(nodes):
            if str(n.agent_id).lower() == mistake_agent.lower():
                rc_idx, gt_source = i, "agent_fallback"
                break
    if rc_idx is None:
        rc_idx = min(max(mistake_step - 1, 0), len(nodes) - 1)
        gt_source = "clamped"

    error_idx = len(nodes) - 1
    if rc_idx == error_idx and len(nodes) > 1:
        # keep the annotation; move the manifestation, rather than silently
        # rewriting the root cause the way trail_adapter used to.
        gt_source += "+rc_is_last_step"

    return {
        "graph": graph,
        "root_cause_node_id": nodes[rc_idx].id,
        "error_node_id": nodes[error_idx].id,
        "gt_source": gt_source,
        "gt_is_exact": gt_source.startswith("exact"),
        "mistake_agent": mistake_agent,
        "mistake_step": mistake_step,
        "mistake_reason": meta.get("mistake_reason", ""),
        "system_name": meta.get("system_name", ""),
        "n_nodes": len(nodes),
        "n_agents": len(agent_counts),
        "hub_agent": hub_agent,
        "n_hub_edges": n_dispatch,
    }


def load_all_runs(data_dir: str = DEFAULT_ROOT, subsets=None,
                  topology: str = "hub", verbose: bool = True):
    """Return list of (run_id, converted, raw_meta)."""
    base = Path(data_dir)
    out = []
    for subset in (subsets or SUBSETS):
        d = base / subset
        if not d.exists():
            continue
        for run in sorted(p for p in d.iterdir() if p.is_dir()):
            try:
                meta = json.load(open(run / "trace_metadata.json"))
                steps = json.load(open(run / "step_records.json"))
            except Exception as e:
                if verbose:
                    print(f"Warning: skipped {run.name}: {e}")
                continue
            conv = convert_run(meta, steps, run.name, topology=topology)
            if conv:
                conv["subset"] = subset
                out.append((f"{subset}/{run.name}", conv, meta))

    if verbose and out:
        prov = Counter(c["gt_source"] for _, c, _ in out)
        exact = sum(1 for _, c, _ in out if c["gt_is_exact"])
        hub = sum(1 for _, c, _ in out if c["hub_agent"])
        print(f"[traceelephant] topology={topology} runs={len(out)} "
              f"exact GT={exact} ({exact/len(out)*100:.1f}%) hub-structured={hub} "
              f"provenance={dict(prov)}")
    return out


if __name__ == "__main__":
    runs = load_all_runs()
    print(f"Loaded {len(runs)} TraceElephant runs")
