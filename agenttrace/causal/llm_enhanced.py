"""
AgentTrace-LLM: LLM-enhanced causal attribution.

Combines zero-cost structural pre-filtering with selective LLM verification.
4-5 LLM calls per trace (~10K tokens) vs CHIEF's ~15 calls (~20K tokens).

Pipeline:
  Stage 1: Structural pre-filtering (0 LLM calls)
  Stage 2: OTAR structuring (1 LLM call or regex)
  Stage 3: Selective oracle verification (2-3 LLM calls)
  Stage 4: Local vs propagated check (1 LLM call)
"""

import os
import re
from dataclasses import dataclass
from typing import Optional

from agenttrace.core.graph import CausalGraph
from agenttrace.ranking.ranker import ImprovedAgentTrace
from agenttrace.causal.scm import CausalAttributor
from agenttrace.inference.semantic_edges import SemanticEdgeInferrer


@dataclass
class OTARTuple:
    """Observation-Thought-Action-Result for an agent step."""

    observation: str = ""
    thought: str = ""
    action: str = ""
    result: str = ""


@dataclass
class LLMAttributionResult:
    """Result from LLM-enhanced attribution."""

    agent_id: str
    node_id: str
    step: int
    confidence: float
    is_local_cause: bool  # True = error originates here, False = propagated
    reasoning: str
    llm_calls: int
    total_tokens: int


class AgentTraceLLM:
    """
    LLM-enhanced AgentTrace: structural pre-filtering + selective LLM verification.
    """

    def __init__(
        self,
        api_key: str = "",
        model: str = "llama-3.3-70b-versatile",
        base_url: str = "https://api.groq.com/openai/v1",
        use_semantic_edges: bool = True,
        top_k_candidates: int = 5,
        oracle_verify_k: int = 3,
    ):
        self.api_key = api_key or os.environ.get("GROQ_API_KEY", "")
        self.model = model
        self.base_url = base_url
        self.top_k = top_k_candidates
        self.oracle_k = oracle_verify_k
        self.use_semantic = use_semantic_edges

        self._client = None
        self._inferrer = None
        self._ranker = ImprovedAgentTrace()
        self._causal = CausalAttributor()
        self.total_tokens = 0
        self.total_calls = 0

    @property
    def client(self):
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        return self._client

    @property
    def inferrer(self):
        if self._inferrer is None and self.use_semantic:
            self._inferrer = SemanticEdgeInferrer(similarity_threshold=0.55)
            _ = self._inferrer.model
        return self._inferrer

    def _llm_call(self, system: str, user: str, temp: float = 0.3, max_tokens: int = 500) -> str:
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user[:6000]},
                ],
                temperature=temp,
                max_tokens=max_tokens,
            )
            self.total_calls += 1
            self.total_tokens += resp.usage.total_tokens
            return resp.choices[0].message.content
        except Exception:
            self.total_calls += 1
            return ""

    # ─── Stage 1: Structural Pre-filtering ─────────────────────────

    def _stage1_structural(self, graph, error_id):
        """Zero-cost structural ranking. Returns top-k candidates."""
        if self.inferrer:
            self.inferrer.enrich_graph(graph)

        # Get structural ranking
        result = self._ranker.find_root_cause(graph, error_id, top_k=self.top_k)
        if not result.ranked_candidates:
            return []

        # Also get causal scores
        causal_results = self._causal.attribute(graph, error_id)
        causal_map = {r.node_id: r for r in causal_results}

        candidates = []
        for rf in result.ranked_candidates[: self.top_k]:
            node = rf.node
            data = node.data if isinstance(node.data, dict) else {}
            cr = causal_map.get(node.id)
            candidates.append(
                {
                    "node_id": node.id,
                    "agent_id": node.agent_id,
                    "step": data.get("step", 0),
                    "content": str(data.get("content", ""))[:300],
                    "structural_score": rf.score,
                    "responsibility": cr.responsibility if cr else 0,
                    "mediation": cr.mediation_score if cr else 0,
                }
            )

        return candidates

    # ─── Stage 2: OTAR Structuring ─────────────────────────────────

    def _stage2_otar(self, graph, candidates):
        """Extract OTAR tuples for top candidates. 1 LLM call."""
        if not self.api_key or not candidates:
            return self._otar_heuristic(graph, candidates)

        # Build context: top candidates with their content
        context_parts = []
        for c in candidates[: self.oracle_k]:
            context_parts.append(f"Step {c['step']} (Agent: {c['agent_id']}): {c['content']}")
        context = "\n".join(context_parts)

        prompt = (
            f"For each step below, extract the OTAR structure:\n"
            f"- Observation: What input/context did the agent receive?\n"
            f"- Thought: What reasoning did the agent do?\n"
            f"- Action: What concrete action was taken?\n"
            f"- Result: What was the output/outcome?\n\n"
            f"{context}\n\n"
            f"For each step, output one line: Step N: O=... T=... A=... R=..."
        )

        text = self._llm_call("Extract OTAR structure from agent steps.", prompt, max_tokens=600)

        # Parse OTAR from LLM output
        for c in candidates:
            step = c["step"]
            pattern = rf"Step\s*{step}.*?O=(.*?)T=(.*?)A=(.*?)R=(.*?)(?:\n|$)"
            m = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if m:
                c["otar"] = OTARTuple(
                    observation=m.group(1).strip()[:200],
                    thought=m.group(2).strip()[:200],
                    action=m.group(3).strip()[:200],
                    result=m.group(4).strip()[:200],
                )
            else:
                c["otar"] = self._otar_from_content(c)

        return candidates

    def _otar_heuristic(self, graph, candidates):
        """Regex-based OTAR extraction (zero-cost fallback)."""
        for c in candidates:
            c["otar"] = self._otar_from_content(c)
        return candidates

    def _otar_from_content(self, candidate):
        content = candidate["content"].lower()
        return OTARTuple(
            observation=candidate["content"][:100],
            thought="(inferred)" if any(k in content for k in ["think", "plan", "consider", "should"]) else "",
            action=candidate.get("action", "process"),
            result=candidate["content"][-100:] if len(candidate["content"]) > 100 else "",
        )

    # ─── Stage 3: Selective Oracle Verification ────────────────────

    def _stage3_oracle_verify(self, candidates, question="", ground_truth=""):
        """Verify top candidates against synthesized oracles. 2-3 LLM calls."""
        if not self.api_key or not candidates:
            return candidates

        verified = []
        for c in candidates[: self.oracle_k]:
            otar = c.get("otar", OTARTuple())

            prompt = (
                f"A multi-agent system was solving this problem:\n"
                f"Problem: {question[:300]}\n\n"
                f"At step {c['step']}, agent '{c['agent_id']}' did:\n"
                f"  Observation: {otar.observation}\n"
                f"  Action: {otar.action}\n"
                f"  Result: {otar.result}\n\n"
                f"Content: {c['content']}\n\n"
                f"Questions:\n"
                f"1. Did this agent receive CORRECT inputs from upstream? (yes/no)\n"
                f"2. Did this agent produce a CORRECT output? (yes/no)\n"
                f"3. If the output is wrong, is it because of THIS agent's mistake "
                f"or because it received wrong inputs? (local/propagated)\n"
                f"4. Confidence that this is the ROOT CAUSE (0-10)?\n\n"
                f"Answer each question on a separate line."
            )

            text = self._llm_call("You verify agent behavior in multi-agent systems.", prompt, max_tokens=300)

            # Parse verification
            correct_input = "yes" in text.lower().split("\n")[0] if text else True
            correct_output = "yes" in text.lower().split("\n")[1] if len(text.split("\n")) > 1 else True
            is_local = "local" in text.lower() if text else False

            # Extract confidence
            conf_match = re.search(r"(\d+)", text.split("\n")[-1]) if text else None
            confidence = int(conf_match.group(1)) / 10.0 if conf_match else 0.5

            c["oracle_verified"] = True
            c["correct_input"] = correct_input
            c["correct_output"] = correct_output
            c["is_local"] = is_local
            c["oracle_confidence"] = confidence
            verified.append(c)

        # Add unverified candidates
        for c in candidates[self.oracle_k :]:
            c["oracle_verified"] = False
            c["oracle_confidence"] = c["structural_score"] * 0.5
            c["is_local"] = False
            verified.append(c)

        return verified

    # ─── Stage 4: Final Ranking ────────────────────────────────────

    def _stage4_final_rank(self, candidates):
        """Combine structural score + oracle verification into final ranking."""
        for c in candidates:
            struct = c.get("structural_score", 0)
            oracle = c.get("oracle_confidence", 0.5)
            is_local = c.get("is_local", False)
            resp = c.get("responsibility", 0)

            # Combined score: oracle verification dominates when available
            if c.get("oracle_verified"):
                c["final_score"] = oracle * 100.0 + (10.0 if is_local else 0.0) + resp * 5.0 + struct * 1.0
            else:
                c["final_score"] = struct * 10.0 + resp * 5.0

        candidates.sort(key=lambda c: c["final_score"], reverse=True)
        return candidates

    # ─── Main Pipeline ─────────────────────────────────────────────

    def attribute(
        self,
        graph: CausalGraph,
        error_node_id: str,
        question: str = "",
        ground_truth: str = "",
    ) -> Optional[LLMAttributionResult]:
        """
        Full LLM-enhanced attribution pipeline.

        Returns the top-1 attribution result.
        """
        self.total_tokens = 0
        self.total_calls = 0

        # Stage 1: Structural pre-filtering (0 LLM calls)
        candidates = self._stage1_structural(graph, error_node_id)
        if not candidates:
            return None

        # Stage 2: OTAR structuring (1 LLM call)
        candidates = self._stage2_otar(graph, candidates)

        # Stage 3: Oracle verification (2-3 LLM calls)
        candidates = self._stage3_oracle_verify(candidates, question=question, ground_truth=ground_truth)

        # Stage 4: Final ranking
        candidates = self._stage4_final_rank(candidates)

        if not candidates:
            return None

        top = candidates[0]
        return LLMAttributionResult(
            agent_id=top["agent_id"],
            node_id=top["node_id"],
            step=top["step"],
            confidence=top.get("oracle_confidence", top.get("structural_score", 0)),
            is_local_cause=top.get("is_local", False),
            reasoning=f"Oracle verified: {top.get('oracle_verified', False)}, "
            f"correct_input: {top.get('correct_input', '?')}, "
            f"is_local: {top.get('is_local', '?')}",
            llm_calls=self.total_calls,
            total_tokens=self.total_tokens,
        )
