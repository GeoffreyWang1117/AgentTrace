"""
Adapter: MAST benchmark → AgentTrace CausalGraph format.

MAST (Multi-Agent System failure Taxonomy) from UC Berkeley provides 1,642+
annotated traces from 7 MAS frameworks with 14 failure modes.

Supported subsets:
- AG2: 7,184 traces (math + coding tasks via AutoGen/AG2)
- HyperAgent: 223 traces (SWE-bench tasks)
- MagenticOne_GAIA: ~150 traces (GAIA tasks)
- programdev: 92 traces (software dev tasks)

Data format (per trace JSON):
{
    "instance_id": "...",
    "problem_statement": ["..."],
    "trajectory": [
        {"content": ["..."], "role": "assistant|user", "name": "agent_name"},
        ...
    ],
    "note": {
        "text": ["annotation text"],
        "options": {
            "Fail to detect ambiguities/contradictions": "yes|no",
            "Proceed with incorrect assumptions": "yes|no",
            ...  # 22 failure mode flags
        }
    }
}

Conversion:
- Each trajectory entry → Node in CausalGraph
- Sequential edges between entries; communication edges on agent handoff
- Failure modes from note.options provide failure category
- Root cause: heuristic based on failure mode (first agent error step)
"""

import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from agenttrace.core.graph import CausalGraph
from agenttrace.core.node import Node, NodeType
from agenttrace.core.edge import Edge, EdgeType


# Failure modes that indicate agent-level errors (vs system issues)
AGENT_ERROR_MODES = {
    'Proceed with incorrect assumptions',
    'Fail to detect ambiguities/contradictions',
    'Derailing from task objectives',
    'Invented content',
    'Discontinued reasoning',
    'Ignoring good suggestions from other agent',
    'Misalignment between internal thoughts and response message',
    'Claiming that a task is done while it is not true.',
    'Poor adherence to specified constraints',
}

VERIFICATION_ERROR_MODES = {
    'No attempt to verify outcome',
    'Evaluator agent fails to be critical',
}

COORDINATION_ERROR_MODES = {
    'Withholding relevant information',
    'Waiting on agents to discover known insights for increased confidence',
    'Redundant conversation turns for iterative tasks rather than batching',
    'Unaware of stopping conditions',
    'Difficulty in agreeing with agents',
    'Blurring roles',
}


def _convert_log_trajectory(log_lines: list[str]) -> list[dict]:
    """Convert HyperAgent-style log lines to chat-message trajectory format."""
    import re
    entries = []
    for line in log_lines:
        if not isinstance(line, str) or not line.strip():
            continue
        # Parse "AgentName_taskid - INFO - Content"
        m = re.match(r'^(\w+?)(?:_\S+)?\s*-\s*\w+\s*-\s*(.+)$', line.strip())
        if m:
            agent_name = m.group(1)
            content = m.group(2)
        else:
            agent_name = 'Agent'
            content = line.strip()
        entries.append({
            'content': [content],
            'role': 'assistant',
            'name': agent_name,
        })
    # Merge consecutive entries from same agent (reduce noise)
    if not entries:
        return entries
    merged = [entries[0]]
    for e in entries[1:]:
        if e['name'] == merged[-1]['name']:
            merged[-1]['content'].extend(e['content'])
        else:
            merged.append(e)
    return merged


def _infer_node_type(entry: dict) -> NodeType:
    """Infer NodeType from trajectory entry."""
    role = entry.get('role', '').lower()
    content = ''
    if isinstance(entry.get('content'), list):
        content = ' '.join(str(c) for c in entry['content']).lower()[:200]
    elif isinstance(entry.get('content'), str):
        content = entry['content'].lower()[:200]

    if role == 'tool' or 'function' in role:
        return NodeType.TOOL_RESULT
    if any(kw in content for kw in ['error', 'fail', 'exception', 'traceback']):
        return NodeType.AGENT_OUTPUT
    if role == 'user':
        return NodeType.AGENT_INPUT
    return NodeType.DECISION


def _get_content_str(entry: dict) -> str:
    """Extract content string from trajectory entry."""
    content = entry.get('content', '')
    if isinstance(content, list):
        return '\n'.join(str(c) for c in content)
    return str(content)


def _get_active_failure_modes(note: dict) -> list[str]:
    """Extract active (yes) failure modes from annotation."""
    options = note.get('options', {})
    return [k for k, v in options.items()
            if isinstance(v, str) and v.lower() == 'yes']


def _estimate_root_cause_step(trajectory: list, failure_modes: list[str],
                              note: dict) -> int:
    """
    Estimate which step is the root cause based on failure modes and content.

    Heuristics:
    1. If agent error modes active → find first step by the failing agent
       where content shows incorrect reasoning
    2. If coordination errors → find first agent handoff that goes wrong
    3. Default → middle of trajectory (conservative estimate)
    """
    n = len(trajectory)
    if n <= 2:
        return 0

    # Check annotation text for clues
    annot_text = ' '.join(note.get('text', [])).lower()

    # Look for step references in annotation
    for i in range(n):
        step_ref = f'step {i + 1}'
        if step_ref in annot_text:
            return i

    # Agent error modes: find first substantive agent step
    if any(m in AGENT_ERROR_MODES for m in failure_modes):
        for i, entry in enumerate(trajectory):
            if entry.get('role') == 'assistant':
                content = _get_content_str(entry).lower()
                # Skip instruction/setup steps
                if len(content) > 50 and i > 0:
                    return i

    # Verification errors: typically near the end
    if any(m in VERIFICATION_ERROR_MODES for m in failure_modes):
        return max(0, n - 2)

    # Coordination errors: find first agent switch
    if any(m in COORDINATION_ERROR_MODES for m in failure_modes):
        for i in range(1, n):
            prev_agent = trajectory[i - 1].get('name', '')
            curr_agent = trajectory[i].get('name', '')
            if prev_agent and curr_agent and prev_agent != curr_agent:
                return i

    # Default: ~1/3 into the trajectory
    return max(1, n // 3)


def convert_trace(trace: dict, trace_id: str,
                  subset: str) -> Optional[dict]:
    """
    Convert a MAST trace to AgentTrace format.

    Returns dict with:
        graph: CausalGraph
        root_cause_node_id: str
        error_node_id: str
        failure_modes: list of active failure categories
        subset: str (AG2, HyperAgent, etc.)
    """
    trajectory = trace.get('trajectory', [])
    if not trajectory or len(trajectory) < 2:
        return None

    note = trace.get('note', {})
    failure_modes = _get_active_failure_modes(note)

    # Build graph
    graph = CausalGraph(run_id=f"mast_{subset}_{trace_id}")
    base_time = datetime.now()
    nodes = []

    for i, entry in enumerate(trajectory):
        agent_id = entry.get('name', entry.get('role', f'agent_{i}')) or f'agent_{i}'
        node_type = _infer_node_type(entry)
        content = _get_content_str(entry)

        node = Node(
            type=node_type,
            agent_id=agent_id,
            data={
                'step': i + 1,
                'role': entry.get('role', 'unknown'),
                'content': content[:2000],  # truncate very long content
                'action': 'respond',
            },
            metadata={
                'source': 'mast',
                'subset': subset,
                'instance_id': trace.get('instance_id', ''),
            }
        )
        node.timestamp = base_time + timedelta(milliseconds=i * 100)

        if i > 0:
            node.parent_ids = [nodes[i - 1].id]

        graph.add_node(node)
        nodes.append(node)

    # Add communication edges on agent handoff
    existing_edges = {(e.source_id, e.target_id) for e in graph._edges.values()}
    for i in range(len(nodes) - 1):
        if nodes[i].agent_id != nodes[i + 1].agent_id:
            pair = (nodes[i].id, nodes[i + 1].id)
            if pair not in existing_edges:
                try:
                    graph.add_edge(Edge(
                        source_id=nodes[i].id,
                        target_id=nodes[i + 1].id,
                        type=EdgeType.TRIGGER_RESPONSE,
                        confidence=0.85,
                        metadata={'type': 'agent_handoff'}
                    ))
                except ValueError:
                    pass

    # Determine root cause and error nodes
    rc_step = _estimate_root_cause_step(trajectory, failure_modes, note)
    rc_idx = min(rc_step, len(nodes) - 1)
    error_idx = len(nodes) - 1

    if rc_idx == error_idx and rc_idx > 0:
        rc_idx = max(0, error_idx - 1)

    # Categorize failure
    failure_category = 'unknown'
    if any(m in AGENT_ERROR_MODES for m in failure_modes):
        failure_category = 'agent_error'
    elif any(m in VERIFICATION_ERROR_MODES for m in failure_modes):
        failure_category = 'verification_error'
    elif any(m in COORDINATION_ERROR_MODES for m in failure_modes):
        failure_category = 'coordination_error'

    return {
        'graph': graph,
        'root_cause_node_id': nodes[rc_idx].id,
        'error_node_id': nodes[error_idx].id,
        'failure_modes': failure_modes,
        'failure_category': failure_category,
        'n_nodes': len(nodes),
        'n_agents': len(set(n.agent_id for n in nodes)),
        'subset': subset,
        'instance_id': trace.get('instance_id', ''),
        'annotation_text': ' '.join(note.get('text', [])),
    }


def load_all_traces(
    data_dir: str = "data/external/mast/repo/traces",
    subsets: Optional[list[str]] = None,
    max_per_subset: Optional[int] = None,
    max_nodes: int = 100,
) -> list[tuple[str, dict, dict]]:
    """
    Load MAST traces and convert to AgentTrace format.

    Args:
        data_dir: path to MAST traces directory
        subsets: list of subsets to load (default: all with JSON traces)
        max_per_subset: max traces per subset (for quick testing)

    Returns list of (trace_id, converted_data, raw_trace).
    """
    base = Path(data_dir)
    available_subsets = ['AG2', 'HyperAgent', 'programdev']

    if subsets:
        available_subsets = [s for s in subsets if s in available_subsets]

    results = []

    for subset in available_subsets:
        subset_dir = base / subset
        if not subset_dir.exists():
            continue

        json_files = sorted(subset_dir.glob('*.json'))
        if max_per_subset:
            json_files = json_files[:max_per_subset]

        loaded = 0
        for f in json_files:
            try:
                trace = json.load(open(f))

                # Skip traces without trajectory
                if not trace.get('trajectory'):
                    continue

                # Convert log-format trajectories (HyperAgent)
                traj = trace['trajectory']
                if traj and isinstance(traj[0], str):
                    trace['trajectory'] = _convert_log_trajectory(traj)
                    if len(trace['trajectory']) < 2:
                        continue

                # Skip traces without failure annotation
                if not trace.get('note', {}).get('options'):
                    continue
                # Skip traces where no failure mode is active
                modes = _get_active_failure_modes(trace.get('note', {}))
                if not modes:
                    continue

                tid = f"{subset}_{f.stem}"
                converted = convert_trace(trace, tid, subset)
                if converted:
                    if converted['n_nodes'] > max_nodes:
                        continue  # skip very large traces (betweenness too slow)
                    results.append((tid, converted, trace))
                    loaded += 1

            except Exception as e:
                continue

        print(f"  {subset}: {loaded} traces loaded from {len(json_files)} files")

    return results


# --- MagenticOne GAIA subset (different format: .log files) ---

def load_magneticone_traces(
    data_dir: str = "data/external/mast/repo/traces/MagenticOne_GAIA",
    max_traces: Optional[int] = None,
) -> list[tuple[str, dict, dict]]:
    """
    Load MagenticOne GAIA traces (log-based format).
    These are stored as directories with .log files per agent.
    """
    base = Path(data_dir)
    results = []

    # Each subdirectory is a trace
    trace_dirs = sorted([d for d in base.iterdir() if d.is_dir()])
    if max_traces:
        trace_dirs = trace_dirs[:max_traces]

    for tdir in trace_dirs:
        log_files = sorted(tdir.glob('*.log'))
        if not log_files:
            continue

        # Build a simple trajectory from log files
        trajectory = []
        for lf in log_files:
            try:
                content = lf.read_text(errors='replace')[:5000]
                agent_name = lf.stem.split('_')[0] if '_' in lf.stem else lf.stem
                trajectory.append({
                    'content': [content],
                    'role': 'assistant',
                    'name': agent_name,
                })
            except Exception:
                continue

        if len(trajectory) < 2:
            continue

        trace = {
            'instance_id': tdir.name,
            'trajectory': trajectory,
            'note': {'text': [], 'options': {
                'Unknown failure': 'yes'
            }},
        }

        tid = f"magneticone_{tdir.name}"
        converted = convert_trace(trace, tid, 'MagenticOne')
        if converted:
            results.append((tid, converted, trace))

    print(f"  MagenticOne: {len(results)} traces loaded")
    return results


if __name__ == '__main__':
    print("Loading MAST traces...")
    traces = load_all_traces()
    print(f"\nTotal: {len(traces)} annotated traces")

    # Per-subset stats
    by_subset = {}
    for tid, t, _ in traces:
        s = t['subset']
        by_subset.setdefault(s, []).append(t)

    for s, ts in sorted(by_subset.items()):
        nodes = sum(t['n_nodes'] for t in ts)
        agents = sum(t['n_agents'] for t in ts)
        print(f"  {s}: {len(ts)} traces, {nodes} total nodes, "
              f"avg {nodes/len(ts):.1f} nodes/trace, "
              f"avg {agents/len(ts):.1f} agents/trace")

    # Failure mode distribution
    all_modes = {}
    for _, t, _ in traces:
        for m in t['failure_modes']:
            all_modes[m] = all_modes.get(m, 0) + 1

    print(f"\nFailure mode distribution (top 10):")
    for mode, count in sorted(all_modes.items(), key=lambda x: -x[1])[:10]:
        print(f"  {mode}: {count} ({100*count/len(traces):.1f}%)")
