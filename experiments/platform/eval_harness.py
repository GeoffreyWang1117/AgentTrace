"""
Layer 4: Unified Evaluation Harness.

Evaluates any method on any dataset/graph with consistent metrics and slicing.
"""

import time
import numpy as np
import networkx as nx
from collections import defaultdict
from agenttrace.core.graph import CausalGraph
from agenttrace.causal.scm import HPActualCause


def evaluate_batch(graphs, rc_ids, error_ids, method_fn, agent_gts=None):
    """
    Evaluate a method on a batch of graphs.

    method_fn(graph, error_id) -> (predicted_node_id, predicted_agent_id) or None

    Returns dict of metrics.
    """
    step_hits = agent_hits = total = 0
    mrrs = []
    times = []

    for i in range(len(graphs)):
        try:
            t0 = time.time()
            result = method_fn(graphs[i], error_ids[i])
            elapsed = time.time() - t0
            times.append(elapsed)

            if result is None:
                continue

            pid, pa = result

            if pid == rc_ids[i]:
                step_hits += 1
            mrrs.append(1.0 if pid == rc_ids[i] else 0.0)

            if agent_gts and agent_gts[i]:
                if pa and agent_gts[i] in pa:
                    agent_hits += 1

            total += 1
        except:
            pass

    n = max(total, 1)
    return {
        'step_acc': step_hits / n * 100,
        'agent_acc': agent_hits / n * 100 if agent_gts else None,
        'hit_at_1': step_hits / n * 100,
        'mrr': np.mean(mrrs) if mrrs else 0,
        'avg_ms': np.mean(times) * 1000 if times else 0,
        'n': total,
    }


def analyze_hp_discrimination(graph, rc_id, error_id):
    """Analyze HP responsibility properties for a single graph."""
    nx_g = graph._graph
    try:
        ancestors = nx.ancestors(nx_g, error_id)
    except:
        return {'discriminates': False, 'unique_values': 0, 'rc_resp': 0}

    candidates = [nid for nid in ancestors if nid != error_id]
    if not candidates or rc_id not in candidates:
        return {'discriminates': False, 'unique_values': 0, 'rc_resp': 0}

    sources = {n for n in ancestors if nx_g.in_degree(n) == 0}
    sub = nx_g.subgraph(ancestors | {error_id})
    hp = HPActualCause()

    resps = {}
    for nid in candidates:
        resps[nid] = hp.responsibility(sub, nid, sources, error_id)

    rc_resp = resps.get(rc_id, 0)
    others = [v for k, v in resps.items() if k != rc_id]
    unique_vals = len(set(round(v, 3) for v in resps.values()))
    discriminates = bool(others and rc_resp > max(others) + 0.01)

    return {
        'discriminates': discriminates,
        'unique_values': unique_vals,
        'rc_resp': rc_resp,
        'max_other': max(others) if others else 0,
        'n_candidates': len(candidates),
    }


def compute_topology_stats(graph):
    """Compute structural statistics for a graph."""
    g = graph._graph
    n = g.number_of_nodes()
    e = g.number_of_edges()
    if n <= 1:
        return {'n_nodes': n, 'n_edges': e, 'topology': 'trivial'}

    max_out = max(g.out_degree(v) for v in g.nodes())
    max_in = max(g.in_degree(v) for v in g.nodes())
    avg_out = np.mean([g.out_degree(v) for v in g.nodes()])
    density = e / max(n * (n - 1) / 2, 1)

    if max_out <= 1 and max_in <= 1:
        topo = 'chain'
    elif max_in <= 1:
        topo = 'tree'
    elif density < 0.2:
        topo = 'sparse_dag'
    elif density < 0.5:
        topo = 'dag'
    else:
        topo = 'dense'

    return {
        'n_nodes': n,
        'n_edges': e,
        'max_out_degree': max_out,
        'max_in_degree': max_in,
        'avg_out_degree': avg_out,
        'density': density,
        'topology': topo,
    }
