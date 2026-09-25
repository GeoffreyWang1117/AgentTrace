"""
Adapter: LLM CTF (SaTML'24) dataset → AgentTrace CausalGraph format.

137,063 attack interactions in a secret extraction game.
Attacker tries to get a defended LLM to reveal a secret string.

Data format (per sample):
{
    "secret": {"value": "rs20ix"},
    "model": "openai/gpt-3.5-turbo-1106",
    "defense": {"team": "PPMLHuskies"},
    "history": [
        {
            "role": "user|assistant",
            "content": "message text",
            "timestamp": {"$date": "ISO 8601"},
            "filter_steps": [
                {"filter_type": null|"python", "content": "filtered output"}
            ]
        },
        ...
    ],
    "is_attack": true,
    "was_successful_secret_extraction": true|false
}

Conversion (3-agent model):
- Attacker agent: user messages (strategic prompts)
- Target LLM: assistant raw responses (before filtering)
- Defense Filter: filter_steps that modify/block responses
- Sequential edges + adversarial edges + filter edges
- Root cause: first attacker message that bypasses defense
- Error node: first assistant response leaking secret (or last response if failed)

Security forensics angle:
- Each filter_step is a defense checkpoint — track where defenses hold vs fail
- Secret leakage detection: check if assistant content contains secret value
- Multi-layered causality: attacker → model → filter → output
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


def _check_secret_in_text(text: str, secret: str) -> bool:
    """Check if the secret value appears in text (case-insensitive)."""
    if not secret or not text:
        return False
    return secret.lower() in text.lower()


def _find_leakage_turn(history: list, secret: str) -> Optional[int]:
    """
    Find the first conversation turn where the model leaks the secret.
    Checks both raw response and post-filter output.
    """
    for i, turn in enumerate(history):
        if turn.get('role') != 'assistant':
            continue
        # Check raw response
        if _check_secret_in_text(turn.get('content', ''), secret):
            return i
        # Check post-filter outputs
        for fs in (turn.get('filter_steps') or []):
            if _check_secret_in_text(fs.get('content', ''), secret):
                return i
    return None


def _find_bypass_turn(history: list, leakage_idx: Optional[int]) -> int:
    """
    Find the attacker message that caused the defense bypass.
    = last user message before the leakage turn.
    If no leakage, return the last user message (best attempt).
    """
    target_idx = leakage_idx if leakage_idx is not None else len(history)
    last_user = 0
    for i in range(target_idx):
        if history[i].get('role') == 'user':
            last_user = i
    return last_user


def convert_interaction(sample: dict, sample_id: str) -> Optional[dict]:
    """
    Convert an LLM CTF interaction to AgentTrace format.

    Models 3 agent types:
    - Attacker: user messages
    - TargetLLM: assistant raw responses
    - DefenseFilter: filter processing steps (inserted as intermediate nodes)
    """
    history = sample.get('history', [])
    if not history or len(history) < 2:
        return None

    secret_info = sample.get('secret', {})
    secret_value = secret_info.get('value', '') if isinstance(secret_info, dict) else ''
    is_successful = sample.get('was_successful_secret_extraction', False)
    model = sample.get('model', 'unknown')
    defense_team = ''
    defense_info = sample.get('defense', {})
    if isinstance(defense_info, dict):
        defense_team = defense_info.get('team', '')

    # Build graph with 3 agent types
    graph = CausalGraph(run_id=f"ctf_{sample_id}")
    base_time = datetime.now()
    nodes = []
    node_idx = 0

    for i, turn in enumerate(history):
        role = turn.get('role', 'unknown')
        content = turn.get('content', '')
        filter_steps = turn.get('filter_steps') or []

        if role == 'user':
            # Attacker node
            node = Node(
                type=NodeType.DECISION,
                agent_id='Attacker',
                data={
                    'step': node_idx + 1,
                    'role': 'attacker',
                    'content': content[:2000],
                    'action': 'probe',
                    'turn': i,
                },
                metadata={
                    'source': 'llm_ctf',
                    'model': model,
                    'defense': defense_team,
                }
            )
            node.timestamp = base_time + timedelta(milliseconds=node_idx * 100)
            if nodes:
                node.parent_ids = [nodes[-1].id]
            graph.add_node(node)
            nodes.append(node)
            node_idx += 1

        elif role == 'assistant':
            # Target LLM raw response node
            llm_node = Node(
                type=NodeType.AGENT_OUTPUT,
                agent_id='TargetLLM',
                data={
                    'step': node_idx + 1,
                    'role': 'model_response',
                    'content': content[:2000],
                    'action': 'respond',
                    'turn': i,
                    'contains_secret': _check_secret_in_text(content, secret_value),
                },
                metadata={
                    'source': 'llm_ctf',
                    'model': model,
                }
            )
            llm_node.timestamp = base_time + timedelta(milliseconds=node_idx * 100)
            if nodes:
                llm_node.parent_ids = [nodes[-1].id]
            graph.add_node(llm_node)
            nodes.append(llm_node)
            node_idx += 1

            # Add defense filter nodes (if any non-trivial filters)
            for fi, fs in enumerate(filter_steps):
                filter_type = fs.get('filter_type')
                filter_content = fs.get('content', '')

                if filter_type is None and filter_content == content:
                    continue  # pass-through filter, skip

                filter_node = Node(
                    type=NodeType.TOOL_CALL,
                    agent_id=f'Filter_{filter_type or "passthrough"}',
                    data={
                        'step': node_idx + 1,
                        'role': 'defense_filter',
                        'content': filter_content[:2000],
                        'action': f'filter_{filter_type or "check"}',
                        'turn': i,
                        'filter_type': filter_type,
                        'modified': filter_content != content,
                        'contains_secret': _check_secret_in_text(filter_content, secret_value),
                    },
                    metadata={
                        'source': 'llm_ctf',
                        'defense': defense_team,
                    }
                )
                filter_node.timestamp = base_time + timedelta(milliseconds=node_idx * 100)
                filter_node.parent_ids = [llm_node.id]
                graph.add_node(filter_node)
                nodes.append(filter_node)
                node_idx += 1

                # Add filter edge
                try:
                    graph.add_edge(Edge(
                        source_id=llm_node.id,
                        target_id=filter_node.id,
                        type=EdgeType.DATA_FLOW,
                        confidence=1.0,
                        metadata={'type': 'defense_filter', 'filter_type': filter_type}
                    ))
                except ValueError:
                    pass

    if len(nodes) < 2:
        return None

    # Add adversarial edges (attacker → next model response)
    existing_edges = {(e.source_id, e.target_id) for e in graph._edges.values()}
    for i in range(len(nodes) - 1):
        src_agent = nodes[i].agent_id
        tgt_agent = nodes[i + 1].agent_id
        if src_agent == 'Attacker' and tgt_agent == 'TargetLLM':
            pair = (nodes[i].id, nodes[i + 1].id)
            if pair not in existing_edges:
                try:
                    graph.add_edge(Edge(
                        source_id=nodes[i].id,
                        target_id=nodes[i + 1].id,
                        type=EdgeType.TRIGGER_RESPONSE,
                        confidence=0.95,
                        metadata={'type': 'adversarial_prompt'}
                    ))
                except ValueError:
                    pass

    # Determine root cause and error nodes
    leakage_idx = None
    if is_successful and secret_value:
        # Find which node leaks the secret
        for ni, n in enumerate(nodes):
            d = n.data if isinstance(n.data, dict) else {}
            if d.get('contains_secret'):
                leakage_idx = ni
                break

    if leakage_idx is not None:
        error_node_id = nodes[leakage_idx].id
        # Root cause: last attacker node before leakage
        rc_idx = 0
        for ni in range(leakage_idx):
            if nodes[ni].agent_id == 'Attacker':
                rc_idx = ni
        root_cause_node_id = nodes[rc_idx].id
    else:
        # Failed attack: last model response = "error" (from attacker's perspective)
        error_node_id = nodes[-1].id
        # Root cause: last attacker message
        rc_idx = 0
        for ni, n in enumerate(nodes):
            if n.agent_id == 'Attacker':
                rc_idx = ni
        root_cause_node_id = nodes[rc_idx].id

    # Ensure different
    if root_cause_node_id == error_node_id and len(nodes) > 1:
        root_cause_node_id = nodes[0].id

    return {
        'graph': graph,
        'root_cause_node_id': root_cause_node_id,
        'error_node_id': error_node_id,
        'attack_success': is_successful,
        'secret_value': secret_value,
        'model': model,
        'defense_team': defense_team,
        'n_nodes': len(nodes),
        'n_turns': len(history),
        'n_filter_nodes': sum(1 for n in nodes if n.agent_id.startswith('Filter')),
        'leakage_detected': leakage_idx is not None,
    }


def load_all_interactions(
    data_dir: str = "data/external/llm_ctf/dataset_interaction_chats",
    attacks_only: bool = True,
    successful_only: bool = False,
    max_traces: Optional[int] = None,
    models: Optional[list[str]] = None,
) -> list[tuple[str, dict, dict]]:
    """
    Load LLM CTF interactions and convert to AgentTrace format.

    Args:
        data_dir: path to HuggingFace dataset
        attacks_only: only load attack interactions (not evaluations)
        successful_only: only load successful extractions
        max_traces: max traces to load
        models: filter by model name
    """
    from datasets import load_from_disk

    ds = load_from_disk(data_dir)['attack']
    results = []

    for i, sample in enumerate(ds):
        if max_traces and len(results) >= max_traces:
            break

        if attacks_only and not sample.get('is_attack', False):
            continue
        if successful_only and not sample.get('was_successful_secret_extraction', False):
            continue
        if models:
            if sample.get('model', '') not in models:
                continue

        sid = f"ctf_{i:06d}"
        converted = convert_interaction(sample, sid)
        if converted:
            results.append((sid, converted, sample))

    return results


if __name__ == '__main__':
    print("Loading LLM CTF interactions...")

    # Load a balanced sample: some successful + some failed
    print("\n--- Successful extractions ---")
    success = load_all_interactions(successful_only=True, max_traces=200)
    print(f"Loaded {len(success)} successful attacks")

    print("\n--- Failed attacks (sample) ---")
    failed = load_all_interactions(successful_only=False, max_traces=200)
    failed = [(s, t, r) for s, t, r in failed if not t['attack_success']][:100]
    print(f"Loaded {len(failed)} failed attacks")

    all_traces = success + failed
    print(f"\nTotal: {len(all_traces)} traces")

    # Stats
    total_nodes = sum(t['n_nodes'] for _, t, _ in all_traces)
    total_filters = sum(t['n_filter_nodes'] for _, t, _ in all_traces)
    leakages = sum(1 for _, t, _ in all_traces if t['leakage_detected'])
    successes = sum(1 for _, t, _ in all_traces if t['attack_success'])

    print(f"  Total nodes: {total_nodes}, avg {total_nodes/max(len(all_traces),1):.1f}/trace")
    print(f"  Filter nodes: {total_filters}")
    print(f"  Leakage detected in trace: {leakages}")
    print(f"  Successful attacks: {successes}")

    from collections import Counter
    models = Counter(t['model'] for _, t, _ in all_traces)
    print(f"\n  Models:")
    for m, c in models.most_common():
        print(f"    {m}: {c}")
