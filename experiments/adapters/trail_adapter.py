"""
Adapter: TRAIL benchmark → AgentTrace CausalGraph format.

TRAIL (Trace Reasoning and Agentic Issue Localization) provides 148 OpenTelemetry
traces with 841 annotated errors across GAIA and SWE-Bench tasks.

Data format (per trace JSON):
{
    "trace_id": "...",
    "spans": [{
        "span_id": "...",
        "parent_span_id": "..." or null,
        "span_name": "Step 1" | "CodeAgent.run" | "LiteLLMModel.__call__" | ...,
        "span_kind": "Internal",
        "span_attributes": {...},
        "duration": "PT1M0.872089S",
        "events": [...],
        "child_spans": [...]  # recursive tree
    }]
}

Annotations (per trace JSON in processed_annotations_*/):
{
    "errors": [{"category": "...", "location": "<span_id>", "evidence": "...",
                "description": "...", "impact": "HIGH|MEDIUM|LOW"}, ...],
    "scores": [{"overall": ..., ...}]
}

Conversion:
- Flatten span tree via DFS, filter to meaningful spans (Steps, Agent.run, Tool calls)
- Each span → Node in CausalGraph
- Parent-child span relationships → edges
- Error annotations map span_id → root cause candidates
- Earliest HIGH/MEDIUM error span → root cause; last span → error node
"""

import json
import os
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional
from dateutil import parser as dtparser

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from agenttrace.core.graph import CausalGraph
from agenttrace.core.node import Node, NodeType
from agenttrace.core.edge import Edge, EdgeType


# Span names that represent meaningful agent steps (not internal LLM calls)
MEANINGFUL_SPAN_PATTERNS = [
    'Step ',           # "Step 1", "Step 2", etc.
    'Agent.run',       # "CodeAgent.run", "ManagedAgent.run"
    'Tool',            # "FinalAnswerTool", "WebSearchTool", etc.
    'answer_single',   # top-level task span
    'create_agent',    # agent creation
    'execute',         # execution spans
    'ManagedAgent',    # managed agent calls
]

# Skip these internal spans (LLM calls, SDK internals)
SKIP_SPAN_PATTERNS = [
    'LiteLLMModel.__call__',
    'litellm',
    'openai',
    'anthropic.',
    'get_examples',
]


def _is_meaningful_span(span: dict) -> bool:
    """Check if a span represents a meaningful agent step vs internal call."""
    name = span.get('span_name', '')
    if any(skip in name for skip in SKIP_SPAN_PATTERNS):
        return False
    if any(pat in name for pat in MEANINGFUL_SPAN_PATTERNS):
        return True
    # Include spans with children (structural spans)
    if span.get('child_spans'):
        return True
    # Include spans with events (tool executions, etc.)
    if span.get('events'):
        return True
    return False


def _flatten_spans(span: dict, depth: int = 0) -> list[dict]:
    """Flatten span tree via DFS, keeping meaningful spans with depth info."""
    result = []
    span['_depth'] = depth
    if _is_meaningful_span(span):
        result.append(span)
    for child in span.get('child_spans', []):
        child['_parent_span_id'] = span.get('span_id')
        result.extend(_flatten_spans(child, depth + 1))
    return result


def _infer_node_type(span: dict) -> NodeType:
    """Infer NodeType from span name and attributes."""
    name = span.get('span_name', '').lower()
    if 'tool' in name:
        return NodeType.TOOL_CALL
    if 'agent' in name and 'run' in name:
        return NodeType.DECISION
    if 'step' in name:
        return NodeType.DECISION
    if 'create' in name:
        return NodeType.AGENT_INPUT
    return NodeType.AGENT_OUTPUT


def _parse_duration_iso(duration_str: str) -> float:
    """Parse ISO 8601 duration like 'PT1M0.872089S' to seconds."""
    if not duration_str or not duration_str.startswith('PT'):
        return 0.0
    s = duration_str[2:]  # strip 'PT'
    total = 0.0
    if 'H' in s:
        h, s = s.split('H')
        total += float(h) * 3600
    if 'M' in s:
        m, s = s.split('M')
        total += float(m) * 60
    if 'S' in s:
        total += float(s.rstrip('S'))
    return total


def _extract_agent_id(span: dict) -> str:
    """Extract agent identifier from span."""
    name = span.get('span_name', '')
    # "CodeAgent.run" → "CodeAgent"
    if '.run' in name:
        return name.split('.run')[0]
    # "Step N" → use parent agent or service name
    if name.startswith('Step '):
        return span.get('service_name', '').split('/')[-1] or 'Agent'
    # Tool names
    if 'Tool' in name:
        return name
    return name or 'unknown'


def _build_parent_map(span: dict, parent_id: Optional[str] = None,
                      out: Optional[dict] = None) -> dict:
    """span_id -> parent span_id over the FULL tree, including skipped spans.

    _flatten_spans only records parents it walks past; we need the complete
    chain so an annotation landing on a skipped span (e.g. LiteLLMModel.__call__)
    can be lifted to its nearest retained ancestor.
    """
    if out is None:
        out = {}
    sid = span.get('span_id')
    if sid:
        out[sid] = parent_id
    for child in span.get('child_spans') or []:
        _build_parent_map(child, sid, out)
    return out


def _resolve_root_cause(errors: list, span_id_to_node_id: dict,
                        parent_map: dict, mode: str) -> tuple[Optional[str], str]:
    """Map error annotations to a retained node.

    Returns (node_id or None, provenance tag). Provenance is recorded so a
    caller can tell a genuine annotation from the fall-through default --
    the distinction the previous implementation silently erased.

    mode='legacy'  : exact pre-2026-08-05 behaviour (direct span match only).
    mode='ancestor': if the annotated span was filtered out, attribute the
                     error to the nearest retained ANCESTOR span (the enclosing
                     step/agent call that contains the annotated LLM call).
    """
    if not errors:
        return None, "no_annotations"

    def direct(only_high_med: bool):
        for err in errors:
            loc = err.get('location', '')
            imp = err.get('impact', '').upper()
            if only_high_med and imp not in ('HIGH', 'MEDIUM'):
                continue
            if loc in span_id_to_node_id:
                return span_id_to_node_id[loc]
        return None

    def via_ancestor(only_high_med: bool):
        for err in errors:
            loc = err.get('location', '')
            imp = err.get('impact', '').upper()
            if only_high_med and imp not in ('HIGH', 'MEDIUM'):
                continue
            cur = parent_map.get(loc)
            seen = set()
            while cur is not None and cur not in span_id_to_node_id:
                if cur in seen:
                    cur = None
                    break
                seen.add(cur)
                cur = parent_map.get(cur)
            if cur is not None and cur in span_id_to_node_id:
                return span_id_to_node_id[cur]
        return None

    nid = direct(True)
    if nid is not None:
        return nid, "direct_high_med"
    if mode == "ancestor":
        nid = via_ancestor(True)
        if nid is not None:
            return nid, "ancestor_high_med"
    nid = direct(False)
    if nid is not None:
        return nid, "direct_any"
    if mode == "ancestor":
        nid = via_ancestor(False)
        if nid is not None:
            return nid, "ancestor_any"
    return None, "unresolved"


def convert_trace(trace: dict, annotations: Optional[dict],
                  trace_id: str, gt_mode: str = None) -> Optional[dict]:
    """
    Convert a TRAIL trace + annotations to AgentTrace format.

    Returns dict with:
        graph: CausalGraph (blind, no GT leakage)
        root_cause_node_id: str (earliest HIGH/MEDIUM error span)
        error_node_id: str (last span)
        errors: list of error annotations
        n_errors: int
    """
    if not trace.get('spans'):
        return None

    # Flatten span tree
    root_span = trace['spans'][0]
    flat_spans = _flatten_spans(root_span)

    if len(flat_spans) < 2:
        return None

    # Build graph
    graph = CausalGraph(run_id=f"trail_{trace_id}")

    try:
        base_time = dtparser.parse(root_span.get('timestamp', ''))
    except Exception:
        base_time = datetime.now()

    nodes = []
    span_id_to_node_id = {}

    for i, span in enumerate(flat_spans):
        agent_id = _extract_agent_id(span)
        node_type = _infer_node_type(span)
        duration = _parse_duration_iso(span.get('duration', ''))

        # Extract content from events or span attributes
        content_parts = []
        for evt in span.get('events', []):
            if isinstance(evt, dict):
                content_parts.append(evt.get('name', ''))
        if span.get('status_message'):
            content_parts.append(span['status_message'])

        node = Node(
            type=node_type,
            agent_id=agent_id,
            data={
                'step': i + 1,
                'span_name': span.get('span_name', ''),
                'content': ' | '.join(content_parts) if content_parts else span.get('span_name', ''),
                'action': span.get('span_kind', 'Internal'),
                'duration_s': duration,
            },
            metadata={
                'source': 'trail',
                'trace_id': trace_id,
                'span_id': span.get('span_id', ''),
                'depth': span.get('_depth', 0),
            }
        )
        node.timestamp = base_time + timedelta(milliseconds=i * 100)

        # Link to parent span if exists
        parent_sid = span.get('_parent_span_id') or span.get('parent_span_id')
        if parent_sid and parent_sid in span_id_to_node_id:
            node.parent_ids = [span_id_to_node_id[parent_sid]]
        elif i > 0:
            node.parent_ids = [nodes[i - 1].id]

        graph.add_node(node)
        nodes.append(node)
        span_id_to_node_id[span.get('span_id', '')] = node.id

    # Add communication edges where agent changes
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

    # Determine root cause from annotations
    error_node_id = nodes[-1].id  # last span = error manifestation

    mode = gt_mode or os.environ.get("TRAIL_GT_MODE", "ancestor")
    errors = annotations.get('errors', []) if annotations else []
    parent_map = _build_parent_map(root_span)

    resolved_id, gt_source = _resolve_root_cause(
        errors, span_id_to_node_id, parent_map, mode
    )
    if resolved_id is None:
        root_cause_node_id = nodes[0].id  # documented fall-through default
        if gt_source not in ("no_annotations",):
            gt_source = "default_node0"
    else:
        root_cause_node_id = resolved_id

    # Don't let root cause == error node
    if root_cause_node_id == error_node_id and len(nodes) > 1:
        root_cause_node_id = nodes[0].id
        gt_source = f"{gt_source}+collapsed_to_node0"

    return {
        'graph': graph,
        'root_cause_node_id': root_cause_node_id,
        'error_node_id': error_node_id,
        'gt_source': gt_source,
        'gt_mode': mode,
        'gt_is_annotated': gt_source.startswith(('direct_', 'ancestor_')),
        'errors': errors,
        'n_errors': len(errors),
        'n_nodes': len(nodes),
        'n_agents': len(set(n.agent_id for n in nodes)),
        'source_task': 'gaia' if 'gaia' in trace_id.lower() else 'swe_bench',
    }


def load_all_traces(
    data_dir: str = "data/external/trail/repo/benchmarking",
    gt_mode: str = None,
    verbose: bool = True,
) -> list[tuple[str, dict, dict]]:
    """
    Load all TRAIL traces and convert to AgentTrace format.

    Returns list of (trace_id, converted_data, raw_trace).

    gt_mode: 'ancestor' (default) lifts an annotation that landed on a filtered
    span to its nearest retained ancestor; 'legacy' reproduces the pre-2026-08-05
    behaviour, in which such annotations were dropped and the root cause silently
    fell through to the first node. Override globally with TRAIL_GT_MODE=legacy.
    """
    results = []
    base = Path(data_dir)

    for subset, annot_dir in [
        ('GAIA', 'processed_annotations_gaia'),
        ('SWE Bench', 'processed_annotations_swe_bench'),
    ]:
        trace_dir = base / 'data' / subset
        annotation_dir = base / annot_dir

        if not trace_dir.exists():
            continue

        for f in sorted(trace_dir.glob('*.json')):
            try:
                trace = json.load(open(f))
                trace_id = f.stem

                # Load matching annotation
                annot_file = annotation_dir / f'{trace_id}.json'
                annotations = None
                if annot_file.exists():
                    annotations = json.load(open(annot_file))

                converted = convert_trace(trace, annotations, trace_id,
                                          gt_mode=gt_mode)
                if converted:
                    results.append((trace_id, converted, trace))

            except Exception as e:
                print(f"Warning: Failed to convert {f.name}: {e}")
                continue

    if verbose and results:
        from collections import Counter
        census = Counter(c.get('gt_source', '?') for _, c, _ in results)
        annotated = sum(1 for _, c, _ in results if c.get('gt_is_annotated'))
        mode = results[0][1].get('gt_mode')
        print(f"[trail_adapter] gt_mode={mode}  traces={len(results)}  "
              f"annotated GT={annotated} ({annotated/len(results)*100:.1f}%)  "
              f"provenance={dict(census)}")

    return results


if __name__ == '__main__':
    traces = load_all_traces()
    print(f"Loaded {len(traces)} TRAIL traces")

    # Stats
    gaia = [t for _, t, _ in traces if t['source_task'] == 'gaia']
    swe = [t for _, t, _ in traces if t['source_task'] == 'swe_bench']
    print(f"  GAIA: {len(gaia)} traces")
    print(f"  SWE-Bench: {len(swe)} traces")

    total_errors = sum(t['n_errors'] for _, t, _ in traces)
    total_nodes = sum(t['n_nodes'] for _, t, _ in traces)
    print(f"  Total errors: {total_errors}")
    print(f"  Total nodes: {total_nodes}")
    print(f"  Avg nodes/trace: {total_nodes / len(traces):.1f}")
    print(f"  Avg errors/trace: {total_errors / len(traces):.1f}")
