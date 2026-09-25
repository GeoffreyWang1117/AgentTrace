"""
Adapter: DEF CON 34 AI Village Red Team dataset → AgentTrace CausalGraph format.

17,309 adversarial conversations where red teamers attempt to break LLM guardrails.
Categories: Societal Harm, Prompt Injections, Information Integrity, Internal Consistency, Security.

Data format (per sample):
{
    "category_name": "Prompt Injections",
    "challenges_name": "Bad Math",
    "conversation": [
        {"role": "system_1", "body": "system prompt..."},
        {"role": "user_2", "body": "attacker message"},
        {"role": "assistant_3", "body": "model response"},
        ...
    ],
    "submission_grade": "accepted|rejected|unsubmitted",
    "conversation_length": 5
}

Conversion:
- Each conversation entry → Node (system=setup, user=attacker, assistant=target)
- Sequential edges between turns; adversarial edges from attacker→model
- Root cause: first attacker message that primes successful attack
- Error node: first model response violating safety (for accepted attacks)
- For rejected/failed: root cause = attacker's best attempt, error node = last turn

Security forensics angle:
- Track how attacker messages escalate through conversation
- Identify attack technique from category_name
- Model this as adversarial causal chain: attacker intent → prompt → model failure
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


def _parse_role(role_str: str) -> tuple[str, str]:
    """Parse role string like 'user_2' → ('user', 'Attacker'), 'assistant_3' → ('assistant', 'TargetLLM')."""
    role_lower = role_str.lower().split('_')[0] if '_' in role_str else role_str.lower()
    if role_lower == 'system':
        return 'system', 'System'
    elif role_lower == 'user':
        return 'user', 'Attacker'
    elif role_lower == 'assistant':
        return 'assistant', 'TargetLLM'
    return role_lower, role_str


def _infer_node_type(role: str) -> NodeType:
    if role == 'system':
        return NodeType.AGENT_INPUT
    elif role == 'user':
        return NodeType.DECISION  # attacker choosing strategy
    elif role == 'assistant':
        return NodeType.AGENT_OUTPUT  # model responding
    return NodeType.DECISION


def _find_attack_pivot(conversation: list, is_successful: bool) -> int:
    """
    Find the conversation turn where the attack pivots.
    For successful attacks: first attacker message after which model starts complying.
    For failed attacks: last attacker message (best attempt).
    """
    if not conversation:
        return 0

    user_turns = [(i, c) for i, c in enumerate(conversation)
                  if c.get('role', '').startswith('user')]

    if not user_turns:
        return 0

    if is_successful:
        # For successful attacks, find the first user turn followed by a non-refusal response
        for i, (idx, _) in enumerate(user_turns):
            # Check if next assistant response is compliant (doesn't refuse)
            for j in range(idx + 1, len(conversation)):
                resp = conversation[j]
                if resp.get('role', '').startswith('assistant'):
                    body = resp.get('body', '').lower()
                    refusal_phrases = ['i cannot', "i can't", 'i apologize', 'i\'m sorry',
                                       'not appropriate', 'against my', 'i must decline']
                    if not any(p in body for p in refusal_phrases):
                        return idx  # this attacker message worked
                    break
        # Fallback: last user turn
        return user_turns[-1][0]
    else:
        # For failed attacks: last user turn
        return user_turns[-1][0]


def _find_error_node(conversation: list, is_successful: bool, pivot_idx: int) -> int:
    """
    Find the error manifestation node.
    For successful attacks: first model response after pivot that violates safety.
    For failed attacks: last model response.
    """
    if is_successful:
        for i in range(pivot_idx + 1, len(conversation)):
            if conversation[i].get('role', '').startswith('assistant'):
                return i
    # Fallback: last turn
    return len(conversation) - 1


def convert_conversation(sample: dict, sample_id: str) -> Optional[dict]:
    """
    Convert a DEF CON red team conversation to AgentTrace format.

    Returns dict with:
        graph: CausalGraph
        root_cause_node_id: str (attacker pivot message)
        error_node_id: str (model compliance/failure)
        attack_category: str
        attack_success: bool
    """
    conversation = sample.get('conversation', [])
    if not conversation or len(conversation) < 2:
        return None

    grade = sample.get('submission_grade', '')
    is_successful = grade == 'accepted'
    category = sample.get('category_name', 'unknown')
    challenge = sample.get('challenges_name', 'unknown')

    # Build graph
    graph = CausalGraph(run_id=f"defcon_{sample_id}")
    base_time = datetime.now()
    nodes = []

    for i, turn in enumerate(conversation):
        role_raw = turn.get('role', f'unknown_{i}')
        role, agent_id = _parse_role(role_raw)
        node_type = _infer_node_type(role)
        body = turn.get('body', '')

        node = Node(
            type=node_type,
            agent_id=agent_id,
            data={
                'step': i + 1,
                'role': role,
                'content': body[:3000],  # cap very long messages
                'action': 'attack' if role == 'user' else 'respond' if role == 'assistant' else 'setup',
            },
            metadata={
                'source': 'defcon_redteam',
                'category': category,
                'challenge': challenge,
                'grade': grade,
                'original_role': role_raw,
            }
        )
        node.timestamp = base_time + timedelta(seconds=i)

        if i > 0:
            node.parent_ids = [nodes[i - 1].id]

        graph.add_node(node)
        nodes.append(node)

    # Add adversarial edges: every attacker→model response is an adversarial influence
    existing_edges = {(e.source_id, e.target_id) for e in graph._edges.values()}
    for i in range(len(nodes) - 1):
        src_role = conversation[i].get('role', '').split('_')[0]
        tgt_role = conversation[i + 1].get('role', '').split('_')[0]
        if src_role == 'user' and tgt_role == 'assistant':
            pair = (nodes[i].id, nodes[i + 1].id)
            if pair not in existing_edges:
                try:
                    graph.add_edge(Edge(
                        source_id=nodes[i].id,
                        target_id=nodes[i + 1].id,
                        type=EdgeType.TRIGGER_RESPONSE,
                        confidence=0.95,
                        metadata={'type': 'adversarial_prompt', 'category': category}
                    ))
                except ValueError:
                    pass

    # Determine root cause and error nodes
    pivot_idx = _find_attack_pivot(conversation, is_successful)
    error_idx = _find_error_node(conversation, is_successful, pivot_idx)

    # Ensure they're different
    if pivot_idx == error_idx:
        if pivot_idx > 0:
            pivot_idx = max(0, pivot_idx - 1)
        elif error_idx < len(nodes) - 1:
            error_idx = min(len(nodes) - 1, error_idx + 1)

    return {
        'graph': graph,
        'root_cause_node_id': nodes[pivot_idx].id,
        'error_node_id': nodes[error_idx].id,
        'attack_category': category,
        'attack_challenge': challenge,
        'attack_success': is_successful,
        'grade': grade,
        'n_nodes': len(nodes),
        'n_turns': len(conversation),
    }


def load_all_conversations(
    data_dir: str = "data/external/defcon_redteam/dataset",
    grades: Optional[list[str]] = None,
    categories: Optional[list[str]] = None,
    max_traces: Optional[int] = None,
) -> list[tuple[str, dict, dict]]:
    """
    Load DEF CON red team conversations and convert to AgentTrace format.

    Args:
        data_dir: path to HuggingFace dataset on disk
        grades: filter by grade (default: ['accepted', 'rejected'] — skip unsubmitted)
        categories: filter by category name
        max_traces: max traces to load
    """
    from datasets import load_from_disk

    ds = load_from_disk(data_dir)['train']

    if grades is None:
        grades = ['accepted', 'rejected']  # skip unsubmitted (no evaluation)

    results = []
    for i, sample in enumerate(ds):
        if max_traces and len(results) >= max_traces:
            break

        grade = sample.get('submission_grade', '')
        if grade not in grades:
            continue

        if categories:
            cat = sample.get('category_name', '')
            if cat not in categories:
                continue

        sid = f"dc_{i:05d}"
        converted = convert_conversation(sample, sid)
        if converted:
            results.append((sid, converted, sample))

    return results


if __name__ == '__main__':
    print("Loading DEF CON Red Team conversations...")
    traces = load_all_conversations(max_traces=500)
    print(f"Loaded {len(traces)} traces")

    # Stats
    from collections import Counter
    cats = Counter(t['attack_category'] for _, t, _ in traces)
    success = sum(1 for _, t, _ in traces if t['attack_success'])
    total_nodes = sum(t['n_nodes'] for _, t, _ in traces)

    print(f"  Successful attacks: {success}/{len(traces)} ({100*success/max(len(traces),1):.1f}%)")
    print(f"  Total nodes: {total_nodes}, avg {total_nodes/max(len(traces),1):.1f}/trace")
    print(f"\n  Categories:")
    for cat, count in cats.most_common():
        print(f"    {cat}: {count}")
