"""
Simple multi-agent system demonstrating AgentTrace capabilities.

This example shows:
- Multiple agents collaborating on a task
- Tool usage tracking
- Error injection for debugging demonstration
- Causal chain visualization
"""

import random
import time
from dataclasses import dataclass
from typing import Any

from agenttrace import Tracer, trace_agent, trace_tool
from agenttrace.core.node import NodeType


# === Tools ===

@trace_tool(tool_name="web_search")
def web_search(query: str) -> list[dict]:
    """Simulate a web search."""
    time.sleep(0.1)  # Simulate latency

    # Simulate results
    results = [
        {"title": f"Result 1 for {query}", "url": "https://example.com/1", "snippet": "..."},
        {"title": f"Result 2 for {query}", "url": "https://example.com/2", "snippet": "..."},
    ]
    return results


@trace_tool(tool_name="database_query")
def database_query(sql: str) -> list[dict]:
    """Simulate a database query."""
    time.sleep(0.05)

    # Simulate a bug: certain queries fail
    if "DELETE" in sql.upper():
        raise ValueError("DELETE operations are not allowed")

    return [{"id": 1, "name": "Sample Data"}]


@trace_tool(tool_name="send_email")
def send_email(to: str, subject: str, body: str) -> bool:
    """Simulate sending an email."""
    time.sleep(0.1)
    return True


@trace_tool(tool_name="calculate")
def calculate(expression: str) -> float:
    """Evaluate a mathematical expression."""
    # Intentional bug: doesn't handle division by zero
    result = eval(expression)
    return result


# === Agents ===

@dataclass
class ResearchAgent:
    """Agent that researches topics using web search."""

    agent_id: str = "researcher"

    @trace_agent()
    def research(self, topic: str) -> dict:
        """Research a topic and return findings."""
        # Search for information
        results = web_search(topic)

        # Process results
        findings = {
            "topic": topic,
            "sources": len(results),
            "summary": f"Found {len(results)} sources about {topic}",
            "data": results,
        }

        return findings


@dataclass
class AnalystAgent:
    """Agent that analyzes data and makes decisions."""

    agent_id: str = "analyst"

    @trace_agent()
    def analyze(self, data: dict) -> dict:
        """Analyze data and produce insights."""
        # Simulate analysis
        insights = {
            "input_size": len(str(data)),
            "recommendation": "proceed" if random.random() > 0.3 else "review",
            "confidence": random.uniform(0.7, 0.99),
        }

        return insights

    @trace_agent()
    def calculate_metrics(self, values: list[float]) -> dict:
        """Calculate metrics from values."""
        if not values:
            raise ValueError("No values provided for analysis")

        # This might fail if values contain zero for division
        total = sum(values)
        avg = total / len(values)

        # Bug: division by minimum value (could be zero)
        normalized = [v / min(values) for v in values]

        return {
            "total": total,
            "average": avg,
            "normalized": normalized,
        }


@dataclass
class CoordinatorAgent:
    """Agent that coordinates other agents."""

    agent_id: str = "coordinator"
    researcher: ResearchAgent = None
    analyst: AnalystAgent = None

    def __post_init__(self):
        if self.researcher is None:
            self.researcher = ResearchAgent()
        if self.analyst is None:
            self.analyst = AnalystAgent()

    @trace_agent()
    def process_request(self, request: str) -> dict:
        """Process a user request by coordinating agents."""
        # Step 1: Research the topic
        research_results = self.researcher.research(request)

        # Step 2: Analyze the findings
        analysis = self.analyst.analyze(research_results)

        # Step 3: Make decision based on analysis
        if analysis["recommendation"] == "proceed":
            action = "execute"
        else:
            action = "hold"

        return {
            "request": request,
            "research": research_results,
            "analysis": analysis,
            "action": action,
        }

    @trace_agent()
    def process_with_calculation(self, values: list[float]) -> dict:
        """Process request with calculation (may trigger bugs)."""
        # This chain can fail if values include zero
        metrics = self.analyst.calculate_metrics(values)

        # Use calculation tool
        expr = f"{metrics['total']} / {metrics['average']}"
        ratio = calculate(expr)

        return {
            "metrics": metrics,
            "ratio": ratio,
        }


# === Demo Functions ===

def run_successful_flow():
    """Demonstrate a successful multi-agent flow."""
    print("\n=== Running Successful Flow ===\n")

    with Tracer(run_id="demo_success") as tracer:
        coordinator = CoordinatorAgent()

        # Process a simple request
        result = coordinator.process_request("machine learning trends 2024")

        print(f"Result: {result['action']}")
        print(f"Confidence: {result['analysis']['confidence']:.2%}")

        # Print trace stats
        stats = tracer.get_statistics()
        print(f"\nTrace Statistics:")
        print(f"  Nodes: {stats['node_count']}")
        print(f"  Edges: {stats['edge_count']}")
        print(f"  Agents: {stats['agents']}")

        return tracer


def run_error_flow():
    """Demonstrate error tracing."""
    print("\n=== Running Error Flow ===\n")

    with Tracer(run_id="demo_error") as tracer:
        coordinator = CoordinatorAgent()

        try:
            # This will fail with division by zero
            result = coordinator.process_with_calculation([1.0, 2.0, 0.0])
            print(f"Result: {result}")
        except Exception as e:
            print(f"Error caught: {type(e).__name__}: {e}")

            # Find and analyze errors
            errors = tracer.find_errors()
            print(f"\nFound {len(errors)} error nodes in trace")

            for error in errors:
                print(f"\n  Error in {error.agent_id}:")
                print(f"    Type: {error.data.get('error_type')}")
                print(f"    Message: {error.data.get('error_message')}")

                # Trace back to find cause
                causes = tracer.trace_backward(error.id)
                print(f"    Causal chain length: {len(causes)}")

        return tracer


def run_database_error_flow():
    """Demonstrate database error tracing."""
    print("\n=== Running Database Error Flow ===\n")

    with Tracer(run_id="demo_db_error") as tracer:
        try:
            # This will fail due to DELETE restriction
            database_query("DELETE FROM users WHERE id = 1")
        except Exception as e:
            print(f"Error caught: {type(e).__name__}: {e}")

            errors = tracer.find_errors()
            if errors:
                error = errors[0]
                analysis = tracer.analyze_error(error.id)
                print(f"\nError Analysis:")
                print(f"  Root causes: {len(analysis['root_causes'])}")
                print(f"  Causal chains: {analysis['chain_count']}")

        return tracer


def demonstrate_counterfactual():
    """Demonstrate counterfactual analysis."""
    print("\n=== Demonstrating Counterfactual Analysis ===\n")

    with Tracer(run_id="demo_counterfactual") as tracer:
        coordinator = CoordinatorAgent()

        # Run a flow
        result = coordinator.process_request("AI safety research")

        # Find a decision point
        decision_nodes = [
            n for n in tracer.graph
            if n.type == NodeType.AGENT_OUTPUT and n.agent_id == "analyst"
        ]

        if decision_nodes:
            decision = decision_nodes[0]
            print(f"Original decision: {decision.data}")

            # Create counterfactual: what if analyst recommended differently?
            alternative_data = {**decision.data, "recommendation": "review"}
            cf_graph = tracer.create_counterfactual(decision.id, alternative_data)

            comparison = tracer.compare_runs(cf_graph)
            print(f"\nCounterfactual Comparison:")
            print(f"  Common nodes: {comparison['common_nodes']}")
            print(f"  Different data: {len(comparison['different_data'])}")

        return tracer


def run_demo():
    """Run all demonstrations."""
    tracers = []

    # Run demos
    tracers.append(run_successful_flow())
    tracers.append(run_error_flow())
    tracers.append(run_database_error_flow())
    tracers.append(demonstrate_counterfactual())

    print("\n" + "=" * 50)
    print("All demos completed!")
    print("=" * 50)

    return tracers


if __name__ == "__main__":
    run_demo()
