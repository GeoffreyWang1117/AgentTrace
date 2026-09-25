"""
Semantic Edge Inference for AgentTrace.

Uses sentence embeddings to infer causal data-flow edges between nodes
based on content similarity. Optional LLM-based edge refinement.

Three tiers:
1. Embedding-only (free, CPU, ~50ms/trace): sentence-transformers similarity
2. Embedding + LLM verification (cheap): verify top-k edge candidates with LLM
3. Full LLM edge inference (expensive): LLM reasons about all pairs

Tier 1 alone should boost Who&When accuracy from 25% to ~50%+ by providing
richer causal structure for the ranker to work with.
"""

import json
import math
import numpy as np

from agenttrace.core.graph import CausalGraph
from agenttrace.core.node import Node
from agenttrace.core.edge import Edge, EdgeType


class SemanticEdgeInferrer:
    """
    Infer causal edges from node content using sentence embeddings.

    Key insight: when Agent B's response references or builds upon
    information from Agent A's earlier message, there's likely a
    data-flow dependency — even if no explicit handoff exists.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        similarity_threshold: float = 0.45,
        max_distance: int = 5,  # max steps apart to consider
        decay_lambda: float = 0.3,  # O10 distance-decay rate (sweepable for sensitivity analysis)
    ):
        self.model_name = model_name
        self.similarity_threshold = similarity_threshold
        self.max_distance = max_distance
        self.decay_lambda = decay_lambda
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        return self._model

    def _extract_text(self, node: Node) -> str:
        """Extract text content from a node for embedding."""
        data = node.data if isinstance(node.data, dict) else {}
        parts = []

        # Agent role/name
        if node.agent_id:
            parts.append(f"Agent: {node.agent_id}")

        # Content
        content = data.get("content", "")
        if isinstance(content, dict):
            content = json.dumps(content, ensure_ascii=False)
        elif isinstance(content, list):
            content = " ".join(str(c) for c in content)
        content = str(content)

        # Truncate very long content
        if len(content) > 500:
            content = content[:500]
        parts.append(content)

        # Action
        action = data.get("action", "")
        if action:
            parts.append(f"Action: {action}")

        return " ".join(parts)

    def infer_edges(self, graph: CausalGraph) -> list[Edge]:
        """
        Infer semantic data-flow edges with three optimizations:

        O4: Asymmetric information flow — penalize edges where target
            is already similar to its other predecessors (shared context).
        O9: Adaptive threshold — adjust based on per-trace similarity distribution.
        O10: Distance decay — exponentially reduce confidence with temporal distance.
        """
        nodes = sorted(graph._nodes.values(), key=lambda n: n.timestamp)
        if len(nodes) < 2:
            return []

        texts = [self._extract_text(n) for n in nodes]
        embeddings = self.model.encode(texts, normalize_embeddings=True)

        # O9: Adaptive threshold — based on trace's similarity distribution
        n = len(nodes)
        sims_upper = []
        for i in range(n):
            for j in range(i + 1, min(i + self.max_distance + 1, n)):
                sims_upper.append(float(np.dot(embeddings[i], embeddings[j])))
        # Use base threshold directly — adaptive thresholding (O9) disabled
        # because high-similarity dialogue traces cause over-filtering.
        # Distance decay (O10) handles noise reduction instead.
        adaptive_thresh = self.similarity_threshold

        # Build index: for each node j, what are the existing predecessors' indices?
        node_to_idx = {nd.id: i for i, nd in enumerate(nodes)}
        existing_pred_idx: dict[int, list[int]] = {i: [] for i in range(n)}
        for e in graph._edges.values():
            si = node_to_idx.get(e.source_id)
            ti = node_to_idx.get(e.target_id)
            if si is not None and ti is not None:
                existing_pred_idx[ti].append(si)

        new_edges = []
        existing_pairs = set()
        for e in graph._edges.values():
            existing_pairs.add((e.source_id, e.target_id))

        for i in range(n):
            for j in range(i + 1, min(i + self.max_distance + 1, n)):
                pair = (nodes[i].id, nodes[j].id)
                if pair in existing_pairs:
                    continue

                sim = float(np.dot(embeddings[i], embeddings[j]))
                if sim < adaptive_thresh:
                    continue

                # O10: Distance decay — closer nodes more likely direct causal
                distance = j - i
                decay = math.exp(-self.decay_lambda * (distance - 1))
                confidence = sim * decay

                if confidence < 0.15:
                    continue

                # Determine edge type
                if nodes[i].agent_id != nodes[j].agent_id:
                    edge_type = EdgeType.TRIGGER_RESPONSE
                else:
                    edge_type = EdgeType.DATA_FLOW

                edge = Edge(
                    source_id=nodes[i].id,
                    target_id=nodes[j].id,
                    type=edge_type,
                    confidence=confidence,
                    metadata={
                        "inferred_by": "semantic_embedding",
                        "similarity": round(sim, 4),
                        "distance_decay": round(decay, 4),
                        "model": self.model_name,
                    },
                )
                new_edges.append(edge)
                existing_pairs.add(pair)

        return new_edges

    def enrich_graph(self, graph: CausalGraph) -> int:
        """Add inferred edges to graph in-place. Returns number added."""
        edges = self.infer_edges(graph)
        added = 0
        for edge in edges:
            try:
                graph.add_edge(edge)
                added += 1
            except (ValueError, KeyError):
                pass
        return added


class LLMEdgeRefiner:
    """
    Refine edge candidates using LLM verification.

    Takes top-k embedding-inferred edges and asks LLM:
    "Does node A's output causally influence node B's behavior?"

    Uses Groq (llama-3.3-70b) for cost-effective verification.
    """

    def __init__(
        self,
        api_key: str = "",
        model: str = "llama-3.3-70b-versatile",
        base_url: str = "https://api.groq.com/openai/v1",
        max_verify: int = 20,  # max edges to verify per trace
    ):
        self.api_key = api_key or __import__("os").environ.get("GROQ_API_KEY", "")
        self.model = model
        self.base_url = base_url
        self.max_verify = max_verify
        self._client = None

    @property
    def client(self):
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        return self._client

    def verify_edges(
        self,
        graph: CausalGraph,
        candidate_edges: list[Edge],
    ) -> list[tuple[Edge, bool, float]]:
        """
        Verify edge candidates with LLM.

        Returns list of (edge, is_causal, confidence).
        """
        if not self.api_key:
            return [(e, True, e.confidence) for e in candidate_edges]

        # Sort by confidence, take top-k
        sorted_edges = sorted(candidate_edges, key=lambda e: e.confidence, reverse=True)
        to_verify = sorted_edges[: self.max_verify]

        results = []
        batch_texts = []

        for edge in to_verify:
            src_node = graph.get_node(edge.source_id)
            tgt_node = graph.get_node(edge.target_id)
            if not src_node or not tgt_node:
                results.append((edge, False, 0.0))
                continue

            src_text = self._node_summary(src_node)
            tgt_text = self._node_summary(tgt_node)
            batch_texts.append((edge, src_text, tgt_text))

        if not batch_texts:
            return results

        # Batch verify (one LLM call for all edges)
        edge_descriptions = []
        for i, (edge, src, tgt) in enumerate(batch_texts):
            edge_descriptions.append(f"Edge {i + 1}: [{src}] → [{tgt}]")

        prompt = f"""Analyze these potential causal dependencies in a multi-agent system.
For each edge, determine if the source node's output CAUSALLY INFLUENCES the target node's behavior.

{chr(10).join(edge_descriptions)}

Return JSON: {{"edges": [{{"id": 1, "is_causal": true/false, "confidence": 0.0-1.0}}]}}"""

        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "Analyze causal dependencies. Return valid JSON only."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
                response_format={"type": "json_object"},
                max_tokens=500,
            )
            data = json.loads(resp.choices[0].message.content)
            llm_results = {r["id"]: r for r in data.get("edges", [])}

            for i, (edge, src, tgt) in enumerate(batch_texts):
                r = llm_results.get(i + 1, {})
                is_causal = r.get("is_causal", True)
                conf = r.get("confidence", edge.confidence)
                results.append((edge, is_causal, conf))

        except Exception:
            # Fallback: accept all
            for edge, src, tgt in batch_texts:
                results.append((edge, True, edge.confidence))

        # Add unverified edges (below max_verify threshold)
        for edge in sorted_edges[self.max_verify :]:
            results.append((edge, True, edge.confidence * 0.8))

        return results

    def _node_summary(self, node: Node) -> str:
        data = node.data if isinstance(node.data, dict) else {}
        content = str(data.get("content", ""))[:150]
        return f"{node.agent_id}: {content}"


class HybridEdgeInferrer:
    """
    Combines embedding similarity + optional LLM verification.

    Tier 1: Embedding-only (default)
    Tier 2: Embedding + LLM verification (if API key provided)
    """

    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        similarity_threshold: float = 0.45,
        max_distance: int = 5,
        use_llm: bool = False,
        groq_api_key: str = "",
    ):
        self.semantic = SemanticEdgeInferrer(
            model_name=embedding_model,
            similarity_threshold=similarity_threshold,
            max_distance=max_distance,
        )
        self.use_llm = use_llm
        self.llm_refiner = LLMEdgeRefiner(api_key=groq_api_key) if use_llm else None

    def enrich_graph(self, graph: CausalGraph) -> dict:
        """
        Enrich graph with inferred edges. Returns stats dict.
        """
        # Step 1: Embedding-based inference
        candidate_edges = self.semantic.infer_edges(graph)

        if self.use_llm and self.llm_refiner and candidate_edges:
            # Step 2: LLM verification
            verified = self.llm_refiner.verify_edges(graph, candidate_edges)
            added = 0
            rejected = 0
            for edge, is_causal, conf in verified:
                if is_causal:
                    edge.confidence = conf
                    try:
                        graph.add_edge(edge)
                        added += 1
                    except (ValueError, KeyError):
                        pass
                else:
                    rejected += 1
            return {
                "candidates": len(candidate_edges),
                "added": added,
                "rejected": rejected,
                "method": "embedding+llm",
            }
        else:
            # Embedding-only
            added = 0
            for edge in candidate_edges:
                try:
                    graph.add_edge(edge)
                    added += 1
                except (ValueError, KeyError):
                    pass
            return {
                "candidates": len(candidate_edges),
                "added": added,
                "rejected": 0,
                "method": "embedding_only",
            }
