"""
Tree-based Ranker for root cause identification.

Uses LightGBM with rich node features for non-linear ranking.
Achieves high Hit@1 while maintaining sub-second inference.
"""

import numpy as np
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional

from agenttrace.core.graph import CausalGraph
from agenttrace.core.node import Node, NodeType


NODE_TYPES = list(NodeType)
N_NODE_TYPES = len(NODE_TYPES)


def extract_rich_features(
    graph: CausalGraph,
    node: Node,
    error_node_id: str,
    all_depths: Dict[str, int],
    vocab: List[str],
    precomputed_bc: Optional[Dict[str, float]] = None,
) -> List[float]:
    """
    Extract a rich feature vector for a node (similar to GNN features but flat).

    Features:
    - One-hot NodeType (N_NODE_TYPES dims)
    - Normalized position (1 dim)
    - In/out degree (2 dims)
    - Depth from error (1 dim)
    - Depth from root (1 dim)
    - Descendants count (1 dim)
    - Betweenness centrality (1 dim)
    - Is divergence point (1 dim)
    - Downstream error keywords (1 dim)
    - Edge confidence stats (2 dims: mean, min)
    - Agent role features (1 dim: num distinct agents in predecessors)
    - TF features over vocabulary (len(vocab) dims)
    """
    nodes_sorted = sorted(graph._nodes.values(), key=lambda n: n.timestamp)
    total_nodes = len(nodes_sorted)
    node_idx = next((i for i, n in enumerate(nodes_sorted) if n.id == node.id), 0)

    # One-hot node type
    type_vec = [0.0] * N_NODE_TYPES
    try:
        idx = NODE_TYPES.index(node.type)
        type_vec[idx] = 1.0
    except ValueError:
        pass

    # Position
    rel_pos = node_idx / max(total_nodes - 1, 1)

    # Degree
    in_deg = len(list(graph._graph.predecessors(node.id)))
    out_deg = len(list(graph._graph.successors(node.id)))

    # Depth from error
    depth_from_error = all_depths.get(node.id, 999)
    max_depth = max(all_depths.values()) if all_depths else 1
    norm_depth = depth_from_error / max(max_depth, 1)

    # Depth from root (BFS upward)
    root_depth = 0
    visited = {node.id}
    queue = [(node.id, 0)]
    while queue:
        cid, d = queue.pop(0)
        preds = list(graph._graph.predecessors(cid))
        if not preds:
            root_depth = max(root_depth, d)
        for pid in preds:
            if pid not in visited:
                visited.add(pid)
                queue.append((pid, d + 1))

    # Descendants
    descendants = graph.trace_forward(node.id)
    desc_count = len(descendants) / max(total_nodes, 1)

    # Betweenness centrality (use precomputed if available)
    if precomputed_bc is not None:
        bc = precomputed_bc.get(node.id, 0.0)
    else:
        try:
            import networkx as nx

            bc = nx.betweenness_centrality(graph._graph).get(node.id, 0)
        except Exception:
            bc = 0.0

    # Is divergence point
    is_div = 1.0 if out_deg > 1 else 0.0

    # Downstream error keywords
    error_kws = {
        "error",
        "fail",
        "failed",
        "failure",
        "invalid",
        "wrong",
        "incorrect",
        "missing",
        "corrupt",
        "timeout",
        "exception",
    }
    downstream_errors = sum(1 for d in descendants if any(kw in str(d.data).lower() for kw in error_kws)) / max(
        len(descendants), 1
    )

    # Edge confidence stats
    incoming_confs = []
    for _, _, e_data in graph._graph.in_edges(node.id, data=True):
        eid = e_data.get("id")
        if eid and eid in graph._edges:
            incoming_confs.append(graph._edges[eid].confidence)
    mean_conf = np.mean(incoming_confs) if incoming_confs else 0.5
    min_conf = min(incoming_confs) if incoming_confs else 0.5

    # Agent diversity in predecessors
    pred_agents = set()
    for pid in graph._graph.predecessors(node.id):
        pred_node = graph.get_node(pid)
        if pred_node:
            pred_agents.add(pred_node.agent_id)
    n_pred_agents = len(pred_agents) / max(total_nodes, 1)

    # TF features
    content_str = str(node.data).lower()
    words = content_str.split()
    word_counter = Counter(words)
    tf_vec = [word_counter.get(w, 0) / max(len(words), 1) for w in vocab]

    features = (
        type_vec
        + [
            rel_pos,
            in_deg / 10.0,
            out_deg / 10.0,
            norm_depth,
            root_depth / max(total_nodes, 1),
            desc_count,
            bc,
            is_div,
            downstream_errors,
            mean_conf,
            min_conf,
            n_pred_agents,
        ]
        + tf_vec
    )
    return features


def build_vocab(graphs: List[CausalGraph], max_words: int = 50) -> List[str]:
    """Build global TF vocabulary from all graphs."""
    word_counts = Counter()
    for graph in graphs:
        for node in graph._nodes.values():
            words = str(node.data).lower().split()
            word_counts.update(words)
    return [w for w, _ in word_counts.most_common(max_words)]


@dataclass
class TreeRankerResult:
    """Result from tree-based ranking."""

    ranked_node_ids: List[str]
    scores: List[float]
    top_prediction: str


class TreeRanker:
    """
    LightGBM-based ranker for root cause identification.

    Uses rich node features with cross-validated LambdaMART.
    Maintains sub-second inference while leveraging non-linear feature interactions.
    """

    def __init__(self, n_folds: int = 5, num_boost_round: int = 100, num_leaves: int = 15, learning_rate: float = 0.1):
        self.n_folds = n_folds
        self.num_boost_round = num_boost_round
        self.num_leaves = num_leaves
        self.learning_rate = learning_rate
        self.models = []
        self.vocab = []

    def train_and_predict(
        self,
        graphs: List[CausalGraph],
        error_node_ids: List[str],
        ground_truth_ids: List[str],
    ) -> List[List[str]]:
        """
        Train with cross-validation and return ranked predictions for each graph.
        """
        try:
            import lightgbm as lgb
        except ImportError:
            print("WARNING: lightgbm not installed")
            return [[] for _ in graphs]

        # Build global vocab
        self.vocab = build_vocab(graphs)

        # Extract features for all graphs
        all_X = []
        all_y = []
        all_groups = []
        all_node_ids = []

        for graph, error_id, gt_id in zip(graphs, error_node_ids, ground_truth_ids):
            nodes = sorted(graph._nodes.values(), key=lambda n: n.timestamp)

            # Compute depths from error
            depths = {error_id: 0}
            visited = {error_id}
            queue = [(error_id, 0)]
            while queue:
                cid, d = queue.pop(0)
                for pid in graph._graph.predecessors(cid):
                    if pid not in visited:
                        visited.add(pid)
                        depths[pid] = d + 1
                        queue.append((pid, d + 1))

            # Precompute betweenness once per graph
            import networkx as nx

            try:
                bc = nx.betweenness_centrality(graph._graph)
            except Exception:
                bc = {}

            group_X = []
            group_y = []
            group_ids = []
            for node in nodes:
                if node.id == error_id:
                    continue
                feat = extract_rich_features(graph, node, error_id, depths, self.vocab, precomputed_bc=bc)
                group_X.append(feat)
                group_y.append(1.0 if node.id == gt_id else 0.0)
                group_ids.append(node.id)

            if group_X:
                all_X.extend(group_X)
                all_y.extend(group_y)
                all_groups.append(len(group_X))
                all_node_ids.append(group_ids)

        X = np.array(all_X)
        y = np.array(all_y)

        # Cross-validation
        n_scenarios = len(all_groups)
        fold_size = max(1, n_scenarios // self.n_folds)
        all_predictions = [None] * n_scenarios

        for fold in range(self.n_folds):
            test_start = fold * fold_size
            test_end = min(test_start + fold_size, n_scenarios)

            # Build train/test splits
            train_idx = []
            test_idx = []
            train_groups = []
            test_groups = []
            offset = 0
            for s_idx, g_size in enumerate(all_groups):
                indices = list(range(offset, offset + g_size))
                if test_start <= s_idx < test_end:
                    test_idx.extend(indices)
                    test_groups.append(g_size)
                else:
                    train_idx.extend(indices)
                    train_groups.append(g_size)
                offset += g_size

            if not train_idx or not test_idx:
                continue

            train_ds = lgb.Dataset(X[train_idx], label=y[train_idx], group=train_groups)
            params = {
                "objective": "lambdarank",
                "metric": "ndcg",
                "ndcg_eval_at": [1, 3, 5],
                "num_leaves": self.num_leaves,
                "learning_rate": self.learning_rate,
                "verbose": -1,
            }
            model = lgb.train(params, train_ds, num_boost_round=self.num_boost_round)

            # Predict on test
            preds = model.predict(X[test_idx])
            test_offset = 0
            for i, g_size in enumerate(test_groups):
                group_preds = preds[test_offset : test_offset + g_size]
                group_ids = all_node_ids[test_start + i]
                ranked_indices = np.argsort(-group_preds)
                ranked_ids = [group_ids[j] for j in ranked_indices]
                all_predictions[test_start + i] = ranked_ids
                test_offset += g_size

        # Fill any None predictions
        for i in range(len(all_predictions)):
            if all_predictions[i] is None:
                all_predictions[i] = all_node_ids[i] if i < len(all_node_ids) else []

        return all_predictions

    def train_full(
        self,
        graphs: List[CausalGraph],
        error_node_ids: List[str],
        ground_truth_ids: List[str],
    ):
        """Train on all data (for deployment, not evaluation)."""
        try:
            import lightgbm as lgb
        except ImportError:
            return

        self.vocab = build_vocab(graphs)
        all_X, all_y, all_groups = [], [], []

        for graph, error_id, gt_id in zip(graphs, error_node_ids, ground_truth_ids):
            nodes = sorted(graph._nodes.values(), key=lambda n: n.timestamp)
            depths = self._compute_depths(graph, error_id)
            import networkx as nx

            try:
                bc = nx.betweenness_centrality(graph._graph)
            except Exception:
                bc = {}

            group_X, group_y = [], []
            for node in nodes:
                if node.id == error_id:
                    continue
                feat = extract_rich_features(graph, node, error_id, depths, self.vocab, precomputed_bc=bc)
                group_X.append(feat)
                group_y.append(1.0 if node.id == gt_id else 0.0)

            if group_X:
                all_X.extend(group_X)
                all_y.extend(group_y)
                all_groups.append(len(group_X))

        X = np.array(all_X)
        y = np.array(all_y)

        train_ds = lgb.Dataset(X, label=y, group=all_groups)
        params = {
            "objective": "lambdarank",
            "metric": "ndcg",
            "ndcg_eval_at": [1, 3, 5],
            "num_leaves": self.num_leaves,
            "learning_rate": self.learning_rate,
            "verbose": -1,
        }
        self.models = [lgb.train(params, train_ds, num_boost_round=self.num_boost_round)]

    def predict(self, graph: CausalGraph, error_node_id: str) -> List[str]:
        """Predict ranked root causes for a single graph using trained model."""
        if not self.models:
            return []

        depths = self._compute_depths(graph, error_node_id)
        nodes = sorted(graph._nodes.values(), key=lambda n: n.timestamp)
        import networkx as nx

        try:
            bc = nx.betweenness_centrality(graph._graph)
        except Exception:
            bc = {}

        X, node_ids = [], []
        for node in nodes:
            if node.id == error_node_id:
                continue
            feat = extract_rich_features(graph, node, error_node_id, depths, self.vocab, precomputed_bc=bc)
            X.append(feat)
            node_ids.append(node.id)

        if not X:
            return []

        X = np.array(X)
        preds = self.models[0].predict(X)
        ranked_indices = np.argsort(-preds)
        return [node_ids[j] for j in ranked_indices]

    def save(self, path: str):
        """Save model and vocab."""
        import pickle

        data = {"vocab": self.vocab, "models": self.models}
        with open(path, "wb") as f:
            pickle.dump(data, f)

    def load(self, path: str):
        """Load model and vocab."""
        import pickle

        with open(path, "rb") as f:
            data = pickle.load(f)
        self.vocab = data["vocab"]
        self.models = data["models"]

    @staticmethod
    def _compute_depths(graph: CausalGraph, error_node_id: str) -> Dict[str, int]:
        depths = {error_node_id: 0}
        visited = {error_node_id}
        queue = [(error_node_id, 0)]
        while queue:
            cid, d = queue.pop(0)
            for pid in graph._graph.predecessors(cid):
                if pid not in visited:
                    visited.add(pid)
                    depths[pid] = d + 1
                    queue.append((pid, d + 1))
        return depths
