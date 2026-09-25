"""
GNN Baseline for AgentTrace Benchmark.

Uses PyTorch Geometric with a 2-layer GCN/GAT for binary classification:
is this node the root cause?

Features: one-hot NodeType (9 dim) + normalized position + in/out degree
+ content TF-IDF (50 dim) = ~62 dimensions.
"""

import json
from typing import Optional
from collections import Counter

from agenttrace.core.graph import CausalGraph
from agenttrace.core.node import NodeType


# Node type list for one-hot encoding
NODE_TYPES = list(NodeType)
N_NODE_TYPES = len(NODE_TYPES)


def _check_torch_geometric():
    """Check if torch_geometric is available."""
    try:
        import torch
        import torch_geometric
        return True
    except ImportError:
        return False


class GNNBaseline:
    """
    GNN-based root cause identification.

    Uses a 2-layer GCN to classify each node as root cause (1) or not (0).
    Implements standard identify_root_cause interface.
    """

    def __init__(self, hidden_dim: int = 64, epochs: int = 100, lr: float = 0.01):
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.lr = lr
        self.model = None
        self._tfidf_vocab = None

    def _extract_node_features(self, graph: CausalGraph) -> list[list[float]]:
        """Extract feature vectors for all nodes."""
        nodes = sorted(graph._nodes.values(), key=lambda n: n.timestamp)
        total = len(nodes)

        # Build TF-IDF vocabulary from all node content
        all_words = []
        node_words_list = []
        for node in nodes:
            content_str = str(node.data).lower()
            words = content_str.split()
            node_words_list.append(words)
            all_words.extend(words)

        # Simple TF-IDF: top 50 most common words
        word_counts = Counter(all_words)
        vocab = [w for w, _ in word_counts.most_common(50)]
        self._tfidf_vocab = vocab

        features = []
        for i, node in enumerate(nodes):
            # One-hot node type
            type_vec = [0.0] * N_NODE_TYPES
            try:
                idx = NODE_TYPES.index(node.type)
                type_vec[idx] = 1.0
            except ValueError:
                pass

            # Normalized position
            rel_pos = i / max(total - 1, 1)

            # In/out degree
            in_deg = len(list(graph._graph.predecessors(node.id)))
            out_deg = len(list(graph._graph.successors(node.id)))

            # Simple TF features
            words = node_words_list[i]
            word_counter = Counter(words)
            tf_vec = [word_counter.get(w, 0) / max(len(words), 1) for w in vocab]

            feature_vec = type_vec + [rel_pos, in_deg / 10.0, out_deg / 10.0] + tf_vec
            features.append(feature_vec)

        return features

    def _build_global_vocab(self, graphs: list[CausalGraph], max_words: int = 50) -> list[str]:
        """Build a global TF vocabulary from all graphs."""
        word_counts = Counter()
        for graph in graphs:
            for node in graph._nodes.values():
                words = str(node.data).lower().split()
                word_counts.update(words)
        return [w for w, _ in word_counts.most_common(max_words)]

    def _extract_features_with_vocab(self, graph: CausalGraph, vocab: list[str]) -> list[list[float]]:
        """Extract features using a fixed vocabulary."""
        nodes = sorted(graph._nodes.values(), key=lambda n: n.timestamp)
        total = len(nodes)
        features = []
        for i, node in enumerate(nodes):
            # One-hot node type
            type_vec = [0.0] * N_NODE_TYPES
            try:
                idx = NODE_TYPES.index(node.type)
                type_vec[idx] = 1.0
            except ValueError:
                pass
            # Position + degree
            rel_pos = i / max(total - 1, 1)
            in_deg = len(list(graph._graph.predecessors(node.id)))
            out_deg = len(list(graph._graph.successors(node.id)))
            # TF with fixed vocab
            words = str(node.data).lower().split()
            word_counter = Counter(words)
            tf_vec = [word_counter.get(w, 0) / max(len(words), 1) for w in vocab]
            features.append(type_vec + [rel_pos, in_deg / 10.0, out_deg / 10.0] + tf_vec)
        return features

    def train_and_predict(
        self,
        graphs: list[CausalGraph],
        error_node_ids: list[str],
        ground_truth_ids: list[str],
        n_folds: int = 5,
    ) -> list[list[str]]:
        """
        Train with cross-validation and return predictions.

        Returns a list of ranked node ID lists, one per graph.
        """
        if not _check_torch_geometric():
            print("WARNING: torch_geometric not installed, returning empty predictions")
            return [[] for _ in graphs]

        import torch
        import torch.nn.functional as F
        from torch_geometric.data import Data
        from torch_geometric.nn import GCNConv

        # Build global vocabulary for consistent feature dimensions
        global_vocab = self._build_global_vocab(graphs)

        # Convert graphs to PyG Data objects
        data_list = []
        for graph, error_id, gt_id in zip(graphs, error_node_ids, ground_truth_ids):
            nodes = sorted(graph._nodes.values(), key=lambda n: n.timestamp)
            node_id_to_idx = {n.id: i for i, n in enumerate(nodes)}

            features = self._extract_features_with_vocab(graph, global_vocab)
            x = torch.tensor(features, dtype=torch.float)

            # Build edge index
            edge_src = []
            edge_tgt = []
            for edge in graph._edges.values():
                if edge.source_id in node_id_to_idx and edge.target_id in node_id_to_idx:
                    edge_src.append(node_id_to_idx[edge.source_id])
                    edge_tgt.append(node_id_to_idx[edge.target_id])
                    # Add reverse edges for undirected message passing
                    edge_src.append(node_id_to_idx[edge.target_id])
                    edge_tgt.append(node_id_to_idx[edge.source_id])

            edge_index = torch.tensor([edge_src, edge_tgt], dtype=torch.long)

            # Labels
            y = torch.zeros(len(nodes), dtype=torch.float)
            if gt_id in node_id_to_idx:
                y[node_id_to_idx[gt_id]] = 1.0

            data = Data(x=x, edge_index=edge_index, y=y)
            data.node_ids = [n.id for n in nodes]
            data_list.append(data)

        if not data_list:
            return []

        # Simple GCN model
        in_dim = data_list[0].x.shape[1]

        hidden = 64

        class GCN(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = GCNConv(in_dim, hidden)
                self.conv2 = GCNConv(hidden, 1)

            def forward(self, data):
                x = F.relu(self.conv1(data.x, data.edge_index))
                x = F.dropout(x, p=0.3, training=self.training)
                x = self.conv2(x, data.edge_index)
                return x.squeeze(-1)

        # Cross-validation
        fold_size = max(1, len(data_list) // n_folds)
        all_predictions = [None] * len(data_list)

        for fold in range(n_folds):
            test_start = fold * fold_size
            test_end = min(test_start + fold_size, len(data_list))

            train_data = data_list[:test_start] + data_list[test_end:]
            test_data = data_list[test_start:test_end]

            if not train_data or not test_data:
                continue

            model = GCN()
            optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)

            # Train
            model.train()
            for epoch in range(self.epochs):
                total_loss = 0
                for data in train_data:
                    optimizer.zero_grad()
                    out = model(data)
                    loss = F.binary_cross_entropy_with_logits(out, data.y)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()

            # Predict on test set
            model.eval()
            with torch.no_grad():
                for i, data in enumerate(test_data):
                    scores = torch.sigmoid(model(data)).numpy()
                    ranked_indices = scores.argsort()[::-1]
                    ranked_ids = [data.node_ids[j] for j in ranked_indices]
                    all_predictions[test_start + i] = ranked_ids

        # Fill any None predictions
        for i in range(len(all_predictions)):
            if all_predictions[i] is None:
                all_predictions[i] = []

        return all_predictions

    def identify_root_cause(
        self,
        graph: CausalGraph,
        error_node_id: str,
    ) -> list[str]:
        """
        Single-graph inference (requires pre-trained model).
        Falls back to position baseline if model not trained.
        """
        from experiments.baselines.position_only_baseline import PositionOnlyBaseline
        return PositionOnlyBaseline().identify_root_cause(graph, error_node_id)
