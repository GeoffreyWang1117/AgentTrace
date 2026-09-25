#!/usr/bin/env python3
"""
AgentTrace Demo — Root Cause Analysis in 30 seconds.

Run: python demo.py

This demo:
1. Creates a 5-agent workflow with a hidden bug
2. Builds a causal graph from the execution
3. Ranks candidate root causes
4. Shows the result with explanation
"""

from agenttrace.core.graph import CausalGraph
from agenttrace.core.node import Node, NodeType
from agenttrace.core.edge import Edge, EdgeType
from agenttrace.ranking.ranker import ImprovedAgentTrace
from agenttrace.ranking.scorer import CausalScorer
from datetime import datetime, timedelta
import time


def create_demo_trace():
    """Simulate a 5-agent research workflow with a bug at step 2."""
    graph = CausalGraph(run_id="demo_research")
    nodes = []

    steps = [
        ("Coordinator", "process",
         "Task: Find the GDP of France in 2023 and convert to JPY"),
        ("Researcher", "submit",
         "Found: France GDP is 2.8 trillion EUR (WRONG — actually 2.8T USD, unit error)"),
        ("Converter", "compute",
         "Converting 2.8T EUR to JPY at rate 160: result 448T JPY"),
        ("Validator", "verify",
         "Cross-checking: IMF reports France GDP ~3.0T USD. "
         "2.8T EUR ≈ 3.0T USD, looks consistent. Approved."),
        ("Reporter", "process",
         "Final report: France 2023 GDP = 448 trillion JPY. "
         "ERROR: actual value should be ~440T JPY (from USD, not EUR)"),
    ]

    for i, (agent, action, content) in enumerate(steps):
        node = Node(
            type=NodeType.DECISION,
            agent_id=agent,
            data={"step": i, "action": action, "content": content}
        )
        node.timestamp = datetime(2026, 1, 1) + timedelta(seconds=i)
        if i > 0:
            node.parent_ids = [nodes[-1].id]
        graph.add_node(node)
        nodes.append(node)

    return graph, nodes


def main():
    print("=" * 60)
    print("  AgentTrace Demo — Root Cause Analysis")
    print("=" * 60)

    # Create trace
    graph, nodes = create_demo_trace()
    root_cause_idx = 1  # Researcher made the unit error
    error_idx = 4       # Reporter shows the wrong final answer

    print(f"\n📋 Workflow: {graph.node_count} agents, {graph.edge_count} edges")
    print(f"   Error appears at: {nodes[error_idx].agent_id} (step {error_idx})")
    print(f"   True root cause:  {nodes[root_cause_idx].agent_id} (step {root_cause_idx})")
    print()

    # Show the trace
    print("Execution trace:")
    for i, node in enumerate(nodes):
        data = node.data
        marker = " ← ERROR" if i == error_idx else (" ← ROOT CAUSE" if i == root_cause_idx else "")
        content_preview = data['content'][:60] + "..." if len(data['content']) > 60 else data['content']
        print(f"  Step {i} [{node.agent_id:>12s}]: {content_preview}{marker}")

    # Run AgentTrace
    print(f"\n🔍 Running AgentTrace...")
    t0 = time.time()
    ranker = ImprovedAgentTrace()
    result = ranker.find_root_cause(graph, nodes[error_idx].id, top_k=5)
    elapsed = (time.time() - t0) * 1000

    print(f"   Completed in {elapsed:.1f}ms\n")

    # Show results
    print("📊 Ranked candidates:")
    for c in result.ranked_candidates:
        is_rc = " ✅ TRUE ROOT CAUSE" if c.node.id == nodes[root_cause_idx].id else ""
        is_err = " (error node)" if c.node.id == nodes[error_idx].id else ""
        print(f"  #{c.rank}: {c.node.agent_id:>12s}  score={c.score:.3f}  "
              f"confidence={c.confidence:.1%}{is_rc}{is_err}")

    # Verdict
    predicted = result.ranked_candidates[0].node if result.ranked_candidates else None
    if predicted and predicted.id == nodes[root_cause_idx].id:
        print(f"\n✅ Correct! AgentTrace identified {predicted.agent_id} as the root cause.")
    else:
        rank = result.get_rank_of(nodes[root_cause_idx].id)
        print(f"\n⚠️  Root cause ranked #{rank}. "
              f"In chain-structured traces, structural features favor high-reachability nodes.")
        print(f"   On real benchmarks, agent-level Hit@1 is 39-54% (Who&When, AgentRx).")

    print(f"\n💡 Key insight: {elapsed:.0f}ms, zero LLM calls, zero cost.")
    print(f"   Trade-off: a frontier LLM (DeepSeek-V4-Pro) reaches ~64% on Who&When,")
    print(f"   at ~$0.01 and ~20 s per trace. AgentTrace is a free first-pass triage, not a replacement.")
    print()


if __name__ == "__main__":
    main()
