"""
Adapter: Who&When benchmark → AgentTrace CausalGraph format.

Who&When (EMNLP 2025 / NeurIPS 2025) provides 184 multi-agent failure scenarios
with ground truth: which agent caused the failure and at which step.

Data format:
{
    "history": [{"content": "...", "role": "...", "name": "agent_name"}, ...],
    "mistake_agent": "AgentName",
    "mistake_step": 2,  # 1-indexed step in history where error originates
    "mistake_reason": "...",
    "ground_truth": "correct answer",
    "is_correct": false,
    ...
}

Conversion:
- Each history entry → Node in CausalGraph
- Sequential edges between consecutive entries
- Agent communication edges between different agents
- Ground truth: mistake_step → root cause node, last step → error node
"""

import json
import glob
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from agenttrace.core.graph import CausalGraph
from agenttrace.core.node import Node, NodeType
from agenttrace.core.edge import Edge, EdgeType


def infer_node_type(entry: dict) -> NodeType:
    """Infer NodeType from history entry."""
    role = entry.get('role', '').lower()
    name = (entry.get('name', '') or '').lower()
    content = (entry.get('content', '') or '').lower()

    if role == 'user':
        return NodeType.AGENT_INPUT
    if 'tool' in role or 'function' in role:
        return NodeType.TOOL_RESULT
    if any(kw in content[:100] for kw in ['error', 'fail', 'cannot', 'unable']):
        return NodeType.AGENT_OUTPUT
    return NodeType.DECISION


def convert_scenario(scenario: dict, scenario_id: str) -> Optional[dict]:
    """
    Convert a Who&When scenario to AgentTrace format.

    Returns dict with:
        graph: CausalGraph (blind, no GT leakage)
        root_cause_node_id: str
        error_node_id: str
        mistake_agent: str
        mistake_step: int
        mistake_reason: str
    """
    history = scenario.get('history', [])
    if not history or len(history) < 2:
        return None

    mistake_step = scenario.get('mistake_step')
    mistake_agent = scenario.get('mistake_agent', '')
    if mistake_step is None:
        return None
    mistake_step = int(mistake_step)

    # Build graph
    graph = CausalGraph(run_id=f"whowhen_{scenario_id}")
    base_time = datetime.now()
    nodes = []

    for i, entry in enumerate(history):
        agent_id = entry.get('name', entry.get('role', f'agent_{i}')) or f'agent_{i}'
        node_type = infer_node_type(entry)

        node = Node(
            type=node_type,
            agent_id=agent_id,
            data={
                'step': i + 1,
                'role': entry.get('role', 'unknown'),
                'content': entry.get('content', ''),
                'action': 'respond',
            },
            metadata={
                'source': 'who_and_when',
                'scenario_id': scenario_id,
            }
        )
        node.timestamp = base_time + timedelta(milliseconds=i * 100)

        if i > 0:
            node.parent_ids = [nodes[i-1].id]

        graph.add_node(node)
        nodes.append(node)

    # Add agent communication edges (when agent changes between steps)
    existing_edges = set()
    for e in graph._edges.values():
        existing_edges.add((e.source_id, e.target_id))

    for i in range(len(nodes) - 1):
        src_agent = nodes[i].agent_id
        tgt_agent = nodes[i+1].agent_id
        pair = (nodes[i].id, nodes[i+1].id)
        if src_agent != tgt_agent and pair not in existing_edges:
            edge = Edge(
                source_id=nodes[i].id,
                target_id=nodes[i+1].id,
                type=EdgeType.TRIGGER_RESPONSE,
                confidence=0.85,
                metadata={'type': 'agent_handoff'}
            )
            try:
                graph.add_edge(edge)
            except ValueError:
                pass

    # Identify root cause and error nodes
    # mistake_step is 1-indexed
    rc_idx = min(mistake_step - 1, len(nodes) - 1)
    error_idx = len(nodes) - 1  # last step is where error manifests

    # Don't let root cause == error node
    if rc_idx == error_idx and rc_idx > 0:
        error_idx = len(nodes) - 1

    return {
        'graph': graph,
        'root_cause_node_id': nodes[rc_idx].id,
        'error_node_id': nodes[error_idx].id,
        'mistake_agent': mistake_agent,
        'mistake_step': mistake_step,
        'mistake_reason': scenario.get('mistake_reason', ''),
        'n_nodes': len(nodes),
        'n_agents': len(set(n.agent_id for n in nodes)),
    }


def load_all_scenarios(data_dir: str = "data/external/who_and_when/repo/Who&When"
                       ) -> list[tuple[str, dict, dict]]:
    """
    Load all Who&When scenarios and convert to AgentTrace format.

    Returns list of (scenario_id, converted_data, raw_scenario).
    Only includes failed scenarios (is_correct=False) with ground truth.
    """
    results = []
    data_path = Path(data_dir)

    for subset in ['Algorithm-Generated', 'Hand-Crafted']:
        subset_dir = data_path / subset
        if not subset_dir.exists():
            continue

        for f in sorted(subset_dir.glob('*.json')):
            try:
                with open(f) as fh:
                    scenario = json.load(fh)

                # Only use scenarios with ground truth annotation
                if scenario.get('is_correct', False) is True:
                    continue  # skip correct scenarios
                if not scenario.get('mistake_step'):
                    continue

                sid = f"{subset[:3].lower()}_{f.stem}"
                converted = convert_scenario(scenario, sid)
                if converted:
                    results.append((sid, converted, scenario))
            except Exception as e:
                pass

    return results


def save_converted_traces(
    converted: list[tuple[str, dict, dict]],
    traces_dir: str = "data/external/who_and_when/traces",
    gt_dir: str = "data/external/who_and_when/ground_truth",
):
    """Save converted traces and ground truth."""
    traces_path = Path(traces_dir)
    gt_path = Path(gt_dir)
    traces_path.mkdir(parents=True, exist_ok=True)
    gt_path.mkdir(parents=True, exist_ok=True)

    for sid, data, raw in converted:
        # Save trace (blind)
        trace_file = traces_path / f"{sid}_trace.json"
        trace_data = {
            'scenario_id': sid,
            'trace_json': data['graph'].to_json(),
            'node_count': data['n_nodes'],
            'edge_count': data['graph'].edge_count,
            'root_cause_node_id': data['root_cause_node_id'],
            'error_node_id': data['error_node_id'],
            'causal_path': [],
            'error_occurred': True,
            'built_at': 'who_and_when_adapter',
        }
        with open(trace_file, 'w') as f:
            json.dump(trace_data, f, indent=2)

        # Save ground truth
        gt_file = gt_path / f"{sid}_gt.json"
        gt_data = {
            'scenario_id': sid,
            'root_cause_node_id': data['root_cause_node_id'],
            'error_node_id': data['error_node_id'],
            'mistake_agent': data['mistake_agent'],
            'mistake_step': data['mistake_step'],
            'mistake_reason': data['mistake_reason'],
        }
        with open(gt_file, 'w') as f:
            json.dump(gt_data, f, indent=2)

    return len(converted)


if __name__ == '__main__':
    print("Loading Who&When scenarios...")
    converted = load_all_scenarios()
    print(f"Converted {len(converted)} scenarios")

    if converted:
        print(f"Sample: {converted[0][0]}")
        d = converted[0][1]
        print(f"  Nodes: {d['n_nodes']}, Agents: {d['n_agents']}")
        print(f"  Mistake agent: {d['mistake_agent']}, step: {d['mistake_step']}")
        print(f"  Root cause: {d['root_cause_node_id'][:20]}...")
        print(f"  Error node: {d['error_node_id'][:20]}...")

    n = save_converted_traces(converted)
    print(f"Saved {n} traces to data/external/who_and_when/traces/")
