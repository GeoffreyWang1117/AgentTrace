"""
OTAR-based feature extraction for multi-agent traces.

Parses agent messages into structured Observation-Thought-Action-Result
components and extracts features that indicate potential root causes.

Key insight from CHIEF: structured decomposition of agent responses
reveals error-prone patterns (e.g., unsupported claims, tool misuse,
missing verification steps). We approximate this WITHOUT LLM calls
using pattern matching + embedding similarity.

Features extracted:
  1. OTAR role classification (observation vs action vs result)
  2. Factual assertion density (claims per step)
  3. Verification gap (actions without preceding verification)
  4. Information flow anomaly (content divergence from expected)
  5. Agent turn complexity (multi-step vs single-step contributions)
  6. Error propagation signature (content from upstream with modifications)
"""

import re
import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from agenttrace.core.graph import CausalGraph
from agenttrace.core.node import Node


@dataclass
class OTARFeatures:
    """OTAR-decomposed features for a single node."""
    node_id: str
    agent_id: str

    # OTAR role
    is_observation: bool = False   # receives input / reads environment
    is_thought: bool = False       # reasoning / planning step
    is_action: bool = False        # executes tool / sends message
    is_result: bool = False        # produces output / final answer

    # Content features
    factual_assertion_count: int = 0     # number of factual claims
    question_count: int = 0              # number of questions asked
    list_item_count: int = 0             # structured list items
    code_block_present: bool = False     # contains code
    url_count: int = 0                   # URLs referenced
    number_count: int = 0               # numerical values mentioned

    # Error-indicative features
    hedge_word_count: int = 0            # "might", "could", "approximately"
    negation_count: int = 0              # "not", "no", "don't", "incorrect"
    correction_language: bool = False    # "actually", "instead", "correction"
    confidence_language: bool = False    # "definitely", "certainly", "sure"

    # Flow features
    verification_gap: float = 0.0        # high = action without verification
    content_novelty: float = 0.0         # how much new info vs repeating upstream
    upstream_similarity: float = 0.0     # max similarity to any predecessor
    downstream_divergence: float = 0.0   # how much downstream differs

    # Content divergence (input vs output mismatch)
    content_divergence: float = 0.0      # how much this node's output differs from its input

    # Agent behavior
    agent_consecutive_steps: int = 1     # how many consecutive steps by same agent
    is_first_agent_step: bool = False    # first contribution by this agent
    is_agent_handoff: bool = False       # agent changes after this step


# Pattern sets for classification
OBSERVATION_PATTERNS = [
    r'\b(given|provided|received|input|task|assigned|asked|requested)\b',
    r'\b(search result|output|response|returned|code output|exitcode)\b',
]

THOUGHT_PATTERNS = [
    r'\b(I will|I\'ll|let me|I need to|I should|plan|strategy|approach)\b',
    r'\b(think|consider|analyze|evaluate|check|verify|confirm)\b',
    r'\b(first|then|next|finally|step \d+)\b',
]

ACTION_PATTERNS = [
    r'\b(execute|run|call|invoke|search|fetch|send|submit|write|create)\b',
    r'\b(```|exitcode|code output|function_call)\b',
]

RESULT_PATTERNS = [
    r'\b(result|answer|conclusion|summary|final|output|done|completed)\b',
    r'\b(here is|here are|the answer|in conclusion|therefore)\b',
]

HEDGE_WORDS = [
    'might', 'could', 'possibly', 'approximately', 'roughly', 'about',
    'maybe', 'perhaps', 'likely', 'probably', 'seem', 'appears',
    'uncertain', 'unclear', 'not sure', 'I think',
]

CORRECTION_WORDS = [
    'actually', 'instead', 'correction', 'corrected', 'sorry',
    'mistake', 'wrong', 'incorrect', 'fix', 'revised', 'update',
    'apolog', 'error in my', 'I was wrong',
]

CONFIDENCE_WORDS = [
    'definitely', 'certainly', 'sure', 'confident', 'clearly',
    'without doubt', 'absolutely', 'exactly', 'precisely',
]

FACTUAL_PATTERNS = [
    r'\b\d{4}\b',                    # years
    r'\$[\d,]+',                     # prices
    r'\d+\s*(?:minutes|hours|days|%|percent)', # quantities
    r'(?:is|are|was|were)\s+(?:available|released|published|created)', # claims
]


def _count_patterns(text: str, patterns: list[str]) -> int:
    """Count total matches of regex patterns in text."""
    count = 0
    text_lower = text.lower()
    for pattern in patterns:
        count += len(re.findall(pattern, text_lower, re.IGNORECASE))
    return count


def _classify_otar(text: str) -> dict:
    """Classify text into OTAR roles by pattern matching."""
    scores = {
        'observation': _count_patterns(text, OBSERVATION_PATTERNS),
        'thought': _count_patterns(text, THOUGHT_PATTERNS),
        'action': _count_patterns(text, ACTION_PATTERNS),
        'result': _count_patterns(text, RESULT_PATTERNS),
    }
    return scores


class OTARFeatureExtractor:
    """
    Extract OTAR-based features from a CausalGraph.

    Uses pattern matching + optional embeddings (no LLM calls).
    """

    def __init__(self, use_embeddings: bool = True):
        self.use_embeddings = use_embeddings
        self._model = None

    @property
    def model(self):
        if self._model is None and self.use_embeddings:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer("all-MiniLM-L6-v2")
        return self._model

    def extract_all(self, graph: CausalGraph, error_node_id: str) -> list[OTARFeatures]:
        """Extract OTAR features for all nodes in graph."""
        nodes = sorted(graph._nodes.values(), key=lambda n: n.timestamp)
        if not nodes:
            return []

        # Get text for all nodes
        texts = [self._get_text(n) for n in nodes]

        # Compute embeddings if available
        embeddings = None
        if self.use_embeddings and self.model is not None:
            try:
                embeddings = self.model.encode(texts, normalize_embeddings=True)
            except Exception:
                pass

        features_list = []
        agent_step_count = {}  # track consecutive steps per agent

        for idx, node in enumerate(nodes):
            text = texts[idx]
            text_lower = text.lower()
            data = node.data if isinstance(node.data, dict) else {}

            # OTAR classification
            otar_scores = _classify_otar(text)
            max_role = max(otar_scores, key=otar_scores.get)

            # Content analysis
            factual_count = _count_patterns(text, FACTUAL_PATTERNS)
            question_count = text.count('?')
            list_items = len(re.findall(r'^\s*[\d\-\*\•]\s*', text, re.MULTILINE))
            code_present = '```' in text or 'exitcode' in text_lower
            url_count = len(re.findall(r'https?://', text))
            number_count = len(re.findall(r'\b\d+\.?\d*\b', text))

            # Error indicators
            hedge_count = sum(1 for w in HEDGE_WORDS if w.lower() in text_lower)
            negation_count = len(re.findall(
                r'\b(not|no|don\'t|doesn\'t|isn\'t|aren\'t|wasn\'t|weren\'t|cannot|can\'t|won\'t)\b',
                text_lower
            ))
            has_correction = any(w.lower() in text_lower for w in CORRECTION_WORDS)
            has_confidence = any(w.lower() in text_lower for w in CONFIDENCE_WORDS)

            # Agent behavior
            prev_agent = nodes[idx - 1].agent_id if idx > 0 else None
            next_agent = nodes[idx + 1].agent_id if idx < len(nodes) - 1 else None

            # Count consecutive steps by same agent
            consecutive = 1
            for k in range(idx - 1, -1, -1):
                if nodes[k].agent_id == node.agent_id:
                    consecutive += 1
                else:
                    break

            is_first = node.agent_id not in agent_step_count
            agent_step_count[node.agent_id] = agent_step_count.get(node.agent_id, 0) + 1
            is_handoff = next_agent is not None and next_agent != node.agent_id

            # Embedding-based features
            content_novelty = 0.0
            upstream_sim = 0.0
            downstream_div = 0.0
            verification_gap = 0.0

            if embeddings is not None:
                # Content novelty: 1 - max_similarity_to_predecessors
                pred_sims = []
                for k in range(max(0, idx - 5), idx):
                    sim = float(np.dot(embeddings[idx], embeddings[k]))
                    pred_sims.append(sim)
                if pred_sims:
                    upstream_sim = max(pred_sims)
                    content_novelty = 1.0 - upstream_sim

                # Downstream divergence: how much successors differ
                succ_sims = []
                for k in range(idx + 1, min(idx + 4, len(nodes))):
                    sim = float(np.dot(embeddings[idx], embeddings[k]))
                    succ_sims.append(sim)
                if succ_sims:
                    downstream_div = 1.0 - np.mean(succ_sims)

                # Content divergence: how much this node transforms its input
                # Root cause nodes often receive correct input but produce wrong output
                # Measured as: 1 - sim(predecessor_from_different_agent, this_node)
                content_divergence = 0.0
                if idx > 0:
                    # Find most recent predecessor from a DIFFERENT agent
                    for k in range(idx - 1, max(-1, idx - 6), -1):
                        if nodes[k].agent_id != node.agent_id:
                            cross_agent_sim = float(np.dot(embeddings[idx], embeddings[k]))
                            # High divergence from cross-agent input = suspicious
                            content_divergence = max(0, 1.0 - cross_agent_sim)
                            break

                # Verification gap: action without close verification step
                if otar_scores['action'] > otar_scores['thought']:
                    has_verification = False
                    for k in range(max(0, idx - 3), idx):
                        k_text = texts[k].lower()
                        if any(v in k_text for v in ['verify', 'check', 'confirm', 'validate']):
                            has_verification = True
                            break
                    verification_gap = 0.0 if has_verification else 1.0

            feat = OTARFeatures(
                node_id=node.id,
                agent_id=node.agent_id,
                is_observation=(max_role == 'observation'),
                is_thought=(max_role == 'thought'),
                is_action=(max_role == 'action'),
                is_result=(max_role == 'result'),
                factual_assertion_count=factual_count,
                question_count=question_count,
                list_item_count=list_items,
                code_block_present=code_present,
                url_count=url_count,
                number_count=number_count,
                hedge_word_count=hedge_count,
                negation_count=negation_count,
                correction_language=has_correction,
                confidence_language=has_confidence,
                verification_gap=verification_gap,
                content_novelty=content_novelty,
                upstream_similarity=upstream_sim,
                downstream_divergence=downstream_div,
                content_divergence=content_divergence,
                agent_consecutive_steps=consecutive,
                is_first_agent_step=is_first,
                is_agent_handoff=is_handoff,
            )
            features_list.append(feat)

        return features_list

    def compute_suspiciousness_score(
        self, feat: OTARFeatures, weights: dict[str, float] | None = None
    ) -> float:
        """
        Compute a suspiciousness score from OTAR features.

        Higher = more likely to be root cause.

        Data-driven signal analysis (Who&When, 2026-04-15):
          Strong:  first_step (+0.171), handoff (+0.056), downstream_div (+0.055)
          Reverse: content_div (-0.053) — RC nodes look NORMAL, not abnormal
                   list_items (-0.296) — RC nodes have FEWER lists
          Weak:    factual, hedge, novelty, verif_gap (~0)

        Accepts optional `weights` dict to override defaults for grid search.
        """
        w = weights or self.default_weights
        score = 0.0

        # First agent step — RC 56% vs non-RC 39%
        if feat.is_first_agent_step:
            score += w.get('first_step', 0.12)

        # Agent handoff — error boundary
        if feat.is_agent_handoff:
            score += w.get('handoff', 0.10)

        # Downstream divergence — output diverges from successors
        score += w.get('downstream_div', 0.15) * feat.downstream_divergence

        # Content divergence INVERTED — RC nodes look normal (low divergence)
        # High content_div = looks different from input = LESS suspicious
        score += w.get('content_div_inv', 0.10) * (1.0 - feat.content_divergence)

        # Content novelty
        score += w.get('novelty', 0.08) * feat.content_novelty

        # Verification gap
        score += w.get('verif_gap', 0.05) * feat.verification_gap

        # Correction language
        if feat.correction_language:
            score += w.get('correction', 0.04)

        # Factual density (weak)
        if feat.factual_assertion_count > 3:
            score += w.get('factual', 0.02)

        return min(1.0, score)

    @property
    def default_weights(self) -> dict[str, float]:
        return {
            'first_step': 0.12,
            'handoff': 0.10,
            'downstream_div': 0.15,
            'content_div_inv': 0.10,
            'novelty': 0.08,
            'verif_gap': 0.05,
            'correction': 0.04,
            'factual': 0.02,
        }

    def _get_text(self, node: Node) -> str:
        """Extract text from node."""
        data = node.data if isinstance(node.data, dict) else {}
        content = data.get('content', '')
        if isinstance(content, (list, dict)):
            content = str(content)
        content = str(content)
        if len(content) > 1000:
            content = content[:1000]

        parts = []
        if node.agent_id:
            parts.append(f"[{node.agent_id}]")
        action = data.get('action', '')
        if action:
            parts.append(f"({action})")
        parts.append(content)
        return ' '.join(parts)
