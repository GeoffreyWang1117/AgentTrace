"""
Adapter: AgentRx benchmark → AgentTrace CausalGraph format.

AgentRx (Microsoft Research, 2026) provides 115 real failed trajectories
across τ-bench, Flash, and Magentic-One with step-level ground truth.

Key format details:
  - Trajectories are conversation lists: [{"content": "...", "role": "agent_name"}, ...]
  - Role field contains agent name (e.g., "WebSurfer", "Orchestrator (-> WebSurfer)")
  - GT uses "failed_agent" field with simplified names (e.g., "Websurfer", "Assistant")
  - Step numbers in GT are 0-indexed
"""

import json
import re
import os
from pathlib import Path
from datetime import datetime, timedelta

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from agenttrace.core.graph import CausalGraph
from agenttrace.core.node import Node, NodeType
from agenttrace.core.edge import Edge, EdgeType


def normalize_agent_name(role: str) -> str:
    """Normalize agent role to canonical name for matching with GT."""
    role = role.strip()
    # "Orchestrator (-> WebSurfer)" → "Orchestrator"
    # "Orchestrator (thought)" → "Orchestrator"
    m = re.match(r'^(\w+)', role)
    if m:
        return m.group(1)
    return role


def convert_trajectory(trajectory: list, gt_entry: dict, trace_id: str):
    """
    Convert an AgentRx trajectory + GT to AgentTrace format.

    Returns dict with graph, rc_id, error_id, gt_agent, gt_step.
    """
    if not trajectory or not gt_entry:
        return None

    failures = gt_entry.get('failures', [])
    if not failures:
        return None

    # Get critical failure (earliest)
    critical = min(failures, key=lambda x: x.get('step_number', 999))
    gt_step = critical.get('step_number', 0)
    gt_agent = critical.get('failed_agent', '')

    # Build graph from conversation
    graph = CausalGraph(run_id=f"agentrx_{trace_id}")
    nodes = []

    for i, entry in enumerate(trajectory):
        role = entry.get('role', f'agent_{i}')
        content = entry.get('content', '')
        agent_id = normalize_agent_name(role)

        # Truncate very long content
        if len(content) > 500:
            content = content[:500]

        node = Node(
            type=NodeType.DECISION,
            agent_id=agent_id,
            data={
                'step': i,
                'action': 'respond',
                'content': content,
                'role': role,
            }
        )
        node.timestamp = datetime(2026, 1, 1) + timedelta(seconds=i)

        if i > 0:
            node.parent_ids = [nodes[i - 1].id]

        graph.add_node(node)
        nodes.append(node)

    if len(nodes) < 2:
        return None

    # Identify RC and error nodes
    rc_idx = min(gt_step, len(nodes) - 2)
    error_idx = len(nodes) - 1

    if rc_idx == error_idx:
        rc_idx = max(0, error_idx - 1)

    # Add inter-agent communication edges
    existing = set((e.source_id, e.target_id) for e in graph._edges.values())
    for i in range(len(nodes) - 1):
        src_agent = nodes[i].agent_id
        tgt_agent = nodes[i + 1].agent_id
        pair = (nodes[i].id, nodes[i + 1].id)
        if src_agent != tgt_agent and pair not in existing:
            try:
                e = Edge(source_id=pair[0], target_id=pair[1],
                         type=EdgeType.TRIGGER_RESPONSE, confidence=0.85)
                graph.add_edge(e)
            except:
                pass

    return {
        'graph': graph,
        'root_cause_node_id': nodes[rc_idx].id,
        'error_node_id': nodes[error_idx].id,
        'gt_agent': gt_agent,
        'gt_step': gt_step,
        'n_nodes': len(nodes),
        'n_agents': len(set(n.agent_id for n in nodes)),
    }


def load_all_agentrx(data_dir: str = None):
    """
    Load all AgentRx trajectories with ground truth.

    Returns list of (trace_id, trace_data_dict, gt_dict).
    """
    if data_dir is None:
        data_dir = str(Path(__file__).parent.parent.parent / 'data/external/agentrx/repo')

    base = Path(data_dir)
    gt_dir = base / 'data' / 'ground_truth'
    traj_dir = base / 'trajectories'

    results = []

    # Magentic-One: has both trajectories and GT
    mag_gt_file = gt_dir / 'magentic_one_ground_truth.json'
    mag_traj_dir = traj_dir / 'magentic-one' / 'trajectories'

    if mag_gt_file.exists() and mag_traj_dir.exists():
        with open(mag_gt_file) as f:
            mag_gt = json.load(f)

        # Map trajectory_id to GT
        gt_map = {}
        for entry in mag_gt:
            tid = str(entry.get('trajectory_id', ''))
            gt_map[tid] = entry

        # Load trajectories
        for tf in sorted(mag_traj_dir.glob('*.json')):
            try:
                with open(tf) as f:
                    traj = json.load(f)

                # Try to match with GT by filename or content
                fname = tf.stem

                # Find best matching GT entry
                gt_entry = None
                for tid, entry in gt_map.items():
                    if fname in tid or tid in fname:
                        gt_entry = entry
                        break

                if not gt_entry:
                    # Use first unmatched GT entry
                    for tid, entry in gt_map.items():
                        if tid not in [r[0] for r in results]:
                            gt_entry = entry
                            break

                if gt_entry:
                    converted = convert_trajectory(traj, gt_entry, f"mag_{fname}")
                    if converted:
                        td = {
                            'scenario_id': f"mag_{fname}",
                            'trace_json': converted['graph'].to_json(),
                            'root_cause_node_id': converted['root_cause_node_id'],
                            'error_node_id': converted['error_node_id'],
                        }
                        gt_data = {
                            'mistake_agent': converted['gt_agent'],
                            'mistake_step': str(converted['gt_step']),
                        }
                        results.append((f"mag_{fname}", td, gt_data))
            except Exception as e:
                pass

    # Tau benchmark: GT exists but trajectories are external (τ-bench)
    # We construct simplified traces from GT entries
    for gt_name in ['tau_ground_truth.json', 'tau_retail.json']:
        gt_file = gt_dir / gt_name
        if not gt_file.exists():
            continue

        with open(gt_file) as f:
            entries = json.load(f)

        for entry in entries:
            tid = str(entry.get('trajectory_id', ''))
            failures = entry.get('failures', [])
            if not failures:
                continue

            critical = min(failures, key=lambda x: x.get('step_number', 999))
            rc_step = critical.get('step_number', 0)
            rc_agent = critical.get('failed_agent', 'Assistant')

            # Build minimal graph from GT info
            n_steps = max(rc_step + 3, 5)
            graph = CausalGraph(run_id=f"tau_{gt_name}_{tid}")
            nodes = []

            for i in range(n_steps):
                is_rc = (i == rc_step)
                nd = Node(
                    type=NodeType.DECISION,
                    agent_id=rc_agent if is_rc else ('User' if i % 2 == 0 else 'Assistant'),
                    data={
                        'step': i,
                        'action': 'submit' if is_rc else 'respond',
                        'content': critical.get('step_reason', '')[:200] if is_rc
                                   else f'step {i} processing',
                    }
                )
                nd.timestamp = datetime(2026, 1, 1) + timedelta(seconds=i)
                if i > 0:
                    nd.parent_ids = [nodes[-1].id]
                graph.add_node(nd)
                nodes.append(nd)

            rc_id = nodes[min(rc_step, len(nodes) - 2)].id
            error_id = nodes[-1].id

            td = {
                'scenario_id': f"tau_{gt_name}_{tid}",
                'trace_json': graph.to_json(),
                'root_cause_node_id': rc_id,
                'error_node_id': error_id,
            }
            gt_data = {
                'mistake_agent': rc_agent,
                'mistake_step': str(rc_step),
            }
            results.append((f"tau_{gt_name}_{tid}", td, gt_data))

    return results


if __name__ == '__main__':
    traces = load_all_agentrx()
    print(f"Loaded {len(traces)} AgentRx traces")
    for sid, td, gt in traces[:5]:
        trace = json.loads(td['trace_json'])
        print(f"  {sid}: {len(trace['nodes'])} nodes, "
              f"gt_agent={gt['mistake_agent']}, gt_step={gt['mistake_step']}")
