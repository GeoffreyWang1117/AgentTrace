"""
Layer 2: Parameterized Topology Generator.

Generates graph families with continuously variable structural parameters:
  - branching_factor: 1 (chain) → 5 (wide tree)
  - path_redundancy: 1 (single path) → 4 (4 parallel paths)
  - symmetry: symmetric (all paths identical) / asymmetric (only one carries error)
  - n_nodes: 8 → 30
  - error_depth: early/middle/late

Each generated graph has:
  - Realistic node content (with error propagation semantics)
  - Known ground truth root cause
  - Typed edges (sequential, data_flow, communication)
"""

import random
import uuid
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Optional

from agenttrace.core.graph import CausalGraph
from agenttrace.core.node import Node, NodeType
from agenttrace.core.edge import Edge, EdgeType


@dataclass
class TopologyConfig:
    n_nodes: int = 10
    branching_factor: int = 1       # 1=chain, 2=binary tree, 3+=wide tree
    path_redundancy: int = 1        # number of parallel paths to error
    symmetric: bool = False         # if True, all paths carry similar content
    error_depth: str = "middle"     # early/middle/late
    n_agents: int = 4


CLEAN_ACTIONS = ['process', 'compute', 'analyze', 'verify', 'aggregate', 'forward']
ERROR_ACTIONS = ['submit', 'write', 'modify', 'transform', 'execute']
CLEAN_CONTENTS = [
    'successfully processed input data',
    'computed result matches expected output',
    'analysis completed without anomalies',
    'verification passed all checks',
    'aggregated results from upstream agents',
    'forwarded validated data to next stage',
]
ERROR_CONTENTS = [
    'error: received corrupted data from upstream processing',
    'warning: computation produced unexpected intermediate result',
    'error: failed validation check on processed output',
    'anomaly detected: output format does not match specification',
    'error: downstream agent received inconsistent state',
]
RC_CONTENTS = [
    'submitted incorrect calculation to downstream',
    'wrote malformed output that corrupted subsequent processing',
    'modified shared state with wrong parameters',
    'transformed input using wrong conversion formula',
    'executed operation with incorrect configuration',
]


def _pick(lst):
    return random.choice(lst)


def generate_graph(config: TopologyConfig, seed: int = None) -> tuple:
    """
    Generate a CausalGraph with specified topology.

    Returns (graph, root_cause_id, error_id, topology_label).
    """
    if seed is not None:
        random.seed(seed)

    n = config.n_nodes
    bf = config.branching_factor
    pr = config.path_redundancy
    agents = [f"agent_{i}" for i in range(config.n_agents)]

    # Determine root cause position
    if config.error_depth == 'early':
        rc_pos = random.randint(1, max(1, n // 4))
    elif config.error_depth == 'late':
        rc_pos = random.randint(max(1, 3 * n // 4), n - 2)
    else:
        rc_pos = random.randint(max(1, n // 4), max(2, 3 * n // 4))

    # Create nodes
    g = CausalGraph(run_id=f"topo_bf{bf}_pr{pr}")
    nodes = []
    for i in range(n):
        is_rc = (i == rc_pos)
        # Downstream of RC gets error content (only on primary path if asymmetric)
        on_error_path = is_rc or (i > rc_pos and (config.symmetric or i < rc_pos + n // 2))

        if is_rc:
            action = _pick(ERROR_ACTIONS)
            content = _pick(RC_CONTENTS)
        elif on_error_path and i > rc_pos:
            action = 'process'
            content = _pick(ERROR_CONTENTS)
        else:
            action = _pick(CLEAN_ACTIONS)
            content = _pick(CLEAN_CONTENTS)

        nd = Node(
            type=NodeType.DECISION,
            agent_id=agents[i % len(agents)],
            data={'step': i, 'action': action, 'content': content}
        )
        nd.timestamp = datetime(2026, 1, 1) + timedelta(seconds=i)
        g.add_node(nd)
        nodes.append(nd)

    # Determine topology label
    if bf == 1 and pr == 1:
        topo = 'chain'
    elif bf >= 2 and pr == 1:
        topo = 'tree'
    elif pr >= 2 and bf == 1:
        topo = 'multi_path'
    elif bf >= 2 and pr >= 2:
        topo = 'dense_dag'
    else:
        topo = 'custom'

    # Build edges based on topology
    edges_added = set()

    def add_edge(src_idx, tgt_idx, etype=EdgeType.DATA_FLOW, conf=0.9):
        if src_idx < n and tgt_idx < n and src_idx != tgt_idx:
            pair = (nodes[src_idx].id, nodes[tgt_idx].id)
            if pair not in edges_added:
                try:
                    e = Edge(source_id=pair[0], target_id=pair[1],
                             type=etype, confidence=conf)
                    g.add_edge(e)
                    edges_added.add(pair)
                except:
                    pass

    if bf == 1 and pr == 1:
        # Pure chain
        for i in range(n - 1):
            add_edge(i, i + 1)

    elif bf >= 2 and pr == 1:
        # Tree: each node branches into bf children
        for i in range(n):
            for b in range(bf):
                child = bf * i + b + 1
                if child < n:
                    add_edge(i, child)
        # Connect all leaves to error node
        for i in range(n - 1):
            has_child = any(bf * i + b + 1 < n for b in range(bf))
            if not has_child:
                add_edge(i, n - 1)

    elif pr >= 2:
        # Multi-path: pr parallel paths from source to error
        path_len = max(2, (n - 2) // pr)
        # Source node = 0, error node = n-1
        for p in range(pr):
            path_start = 1 + p * path_len
            add_edge(0, path_start)
            for j in range(path_len - 1):
                idx = path_start + j
                if idx + 1 < n - 1:
                    add_edge(idx, idx + 1)
            # Connect path end to error
            path_end = min(path_start + path_len - 1, n - 2)
            add_edge(path_end, n - 1)

        # Add cross-path edges for dense DAGs
        if bf >= 2:
            for p1 in range(pr):
                for p2 in range(p1 + 1, pr):
                    cross_point = 1 + p1 * path_len + path_len // 2
                    cross_target = 1 + p2 * path_len + path_len // 2
                    if cross_point < n - 1 and cross_target < n - 1:
                        add_edge(cross_point, cross_target)

    # Ensure all nodes are connected (add sequential fallback)
    for i in range(n - 1):
        preds = list(g._graph.predecessors(nodes[i + 1].id))
        if not preds:
            add_edge(i, i + 1, EdgeType.TEMPORAL, 0.7)

    rc_id = nodes[rc_pos].id
    error_id = nodes[-1].id

    return g, rc_id, error_id, topo


def generate_graph_family(
    branching_factors=(1, 2, 3),
    path_redundancies=(1, 2, 3),
    n_nodes_range=(8, 12, 16),
    depths=('early', 'middle', 'late'),
    symmetric_opts=(True, False),
    trials_per_config=200,
) -> list:
    """
    Generate a complete parameterized graph family.

    Returns list of (config_dict, graph, rc_id, error_id, topology).
    """
    results = []
    seed = 0

    for bf in branching_factors:
        for pr in path_redundancies:
            for nn in n_nodes_range:
                for depth in depths:
                    for sym in symmetric_opts:
                        config = TopologyConfig(
                            n_nodes=nn,
                            branching_factor=bf,
                            path_redundancy=pr,
                            symmetric=sym,
                            error_depth=depth,
                        )
                        for trial in range(trials_per_config):
                            seed += 1
                            g, rc_id, eid, topo = generate_graph(config, seed=seed)
                            results.append({
                                'config': {
                                    'bf': bf, 'pr': pr, 'n': nn,
                                    'depth': depth, 'sym': sym,
                                },
                                'graph': g,
                                'rc_id': rc_id,
                                'error_id': eid,
                                'topology': topo,
                                'seed': seed,
                            })

    return results
