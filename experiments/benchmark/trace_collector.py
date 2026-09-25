"""
Trace Collector for AgentTrace Benchmark.

This module executes generated scenarios using AgentTrace instrumentation
and collects the resulting traces for evaluation.
"""

import json
import asyncio
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Any, Optional
from datetime import datetime

from openai import AsyncOpenAI

from agenttrace import Tracer
from agenttrace.core.node import Node, NodeType
from agenttrace.core.edge import Edge, EdgeType
from agenttrace.hooks.decorators import trace_agent, trace_tool

from .scenario_generator import Scenario, Message, ScenarioDomain


@dataclass
class TraceResult:
    """Result of executing a scenario with tracing."""
    scenario_id: str
    trace_json: str  # Serialized CausalGraph
    execution_log: list[dict]
    error_occurred: bool
    error_message: Optional[str]
    execution_time_ms: float
    node_count: int
    edge_count: int
    collected_at: str


class SimulatedAgent:
    """
    A simulated agent that executes scenario steps.

    Uses GPT-4 to generate realistic agent responses based on the scenario.
    """

    def __init__(self, agent_id: str, role: str, capabilities: list[str], description: str, model: str = None):
        from experiments.config import OPENAI_MODELS
        self.agent_id = agent_id
        self.role = role
        self.capabilities = capabilities
        self.description = description
        self.client = AsyncOpenAI()
        self.model = model or OPENAI_MODELS.get("agent_simulation", "gpt-4o-mini")
        self.context: list[dict] = []

    @trace_agent()
    async def process(self, action: str, content: dict, inject_bug: bool = False, bug_description: str = None) -> dict:
        """Process an action and return a response."""

        # Build prompt for the agent
        prompt = f"""You are a {self.role} agent with capabilities: {', '.join(self.capabilities)}.
{self.description}

You received an action: {action}
With content: {json.dumps(content)}

Previous context: {json.dumps(self.context[-3:]) if self.context else 'None'}

{"IMPORTANT: You must introduce this bug in your response: " + bug_description if inject_bug and bug_description else ""}

Respond as this agent would, returning a JSON object with your response.
Keep the response concise and realistic."""

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": f"You are simulating a {self.role} agent. Always respond with valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            response_format={"type": "json_object"}
        )

        result = json.loads(response.choices[0].message.content)

        # Store in context
        self.context.append({
            "action": action,
            "content": content,
            "response": result
        })

        return result


class SimulatedTool:
    """A simulated tool that can be called by agents."""

    def __init__(self, tool_name: str, description: str):
        self.tool_name = tool_name
        self.description = description

    @trace_tool()
    async def execute(self, params: dict, inject_bug: bool = False, bug_description: str = None) -> dict:
        """Execute the tool with given parameters."""

        # Simulate tool execution
        if inject_bug:
            # Return buggy result
            return {
                "status": "error" if "error" in bug_description.lower() else "success",
                "result": f"Buggy result: {bug_description}",
                "params": params
            }

        return {
            "status": "success",
            "result": f"Tool {self.tool_name} executed successfully",
            "params": params
        }


class ScenarioExecutor:
    """Executes scenarios and collects traces."""

    def __init__(self, use_llm: bool = True):
        """
        Initialize the executor.

        Args:
            use_llm: If True, use GPT-4 to simulate agent responses.
                    If False, use simple rule-based simulation (faster, cheaper).
        """
        self.use_llm = use_llm

    async def execute_scenario(
        self,
        scenario: Scenario,
        use_buggy_flow: bool = True
    ) -> TraceResult:
        """
        Execute a scenario and collect the trace.

        Args:
            scenario: The scenario to execute
            use_buggy_flow: If True, execute the buggy flow. Otherwise, execute correct flow.

        Returns:
            TraceResult with the collected trace
        """
        flow = scenario.buggy_flow if use_buggy_flow else scenario.correct_flow

        start_time = datetime.now()
        execution_log = []
        error_occurred = False
        error_message = None

        # Create agents
        agents = {
            agent.id: SimulatedAgent(
                agent_id=agent.id,
                role=agent.role,
                capabilities=agent.capabilities,
                description=agent.description
            )
            for agent in scenario.agents
        }

        # Add a "user" agent for external inputs
        agents["user"] = SimulatedAgent(
            agent_id="user",
            role="User",
            capabilities=["input", "output"],
            description="External user providing input to the system"
        )

        with Tracer(run_id=f"scenario_{scenario.scenario_id}") as tracer:
            try:
                for msg in flow:
                    step_log = {
                        "step": msg.step,
                        "from": msg.from_agent,
                        "to": msg.to_agent,
                        "action": msg.action,
                        "is_bug_point": msg.is_bug_point
                    }

                    # Get the receiving agent
                    if msg.to_agent in agents:
                        agent = agents[msg.to_agent]

                        if self.use_llm:
                            # Use LLM to simulate agent response
                            response = await agent.process(
                                action=msg.action,
                                content=msg.content,
                                inject_bug=msg.is_bug_point,
                                bug_description=msg.bug_description
                            )
                        else:
                            # Simple rule-based simulation
                            response = self._simple_simulate(msg)

                        step_log["response"] = response
                    else:
                        step_log["error"] = f"Unknown agent: {msg.to_agent}"

                    execution_log.append(step_log)

                    # Check if this is the error manifestation point
                    if msg.step == scenario.error_manifestation_step and use_buggy_flow:
                        # Simulate error detection
                        if msg.is_bug_point or msg.step > scenario.bug_injection_point:
                            error_occurred = True
                            error_message = f"Error manifested at step {msg.step}"

            except Exception as e:
                error_occurred = True
                error_message = str(e)
                execution_log.append({"error": str(e)})

        end_time = datetime.now()
        execution_time_ms = (end_time - start_time).total_seconds() * 1000

        return TraceResult(
            scenario_id=scenario.scenario_id,
            trace_json=tracer.graph.to_json(),
            execution_log=execution_log,
            error_occurred=error_occurred,
            error_message=error_message,
            execution_time_ms=execution_time_ms,
            node_count=tracer.graph.node_count,
            edge_count=tracer.graph.edge_count,
            collected_at=datetime.now().isoformat()
        )

    def _simple_simulate(self, msg: Message) -> dict:
        """Simple rule-based simulation without LLM."""
        if msg.is_bug_point:
            return {
                "status": "error",
                "message": msg.bug_description or "An error occurred",
                "content": msg.content
            }
        return {
            "status": "success",
            "message": f"Processed {msg.action}",
            "content": msg.content
        }


class TraceCollector:
    """Collects traces for all scenarios in the benchmark."""

    def __init__(
        self,
        scenarios_dir: Path = Path("data/scenarios"),
        traces_dir: Path = Path("data/traces"),
        use_llm: bool = True
    ):
        self.scenarios_dir = scenarios_dir
        self.traces_dir = traces_dir
        self.traces_dir.mkdir(parents=True, exist_ok=True)
        self.executor = ScenarioExecutor(use_llm=use_llm)

    def load_scenarios(self, domain: Optional[ScenarioDomain] = None) -> list[Scenario]:
        """Load all scenarios, optionally filtered by domain."""
        scenarios = []

        if domain:
            domains = [domain]
        else:
            domains = list(ScenarioDomain)

        for d in domains:
            domain_dir = self.scenarios_dir / d.value
            if domain_dir.exists():
                for filepath in domain_dir.glob("*.json"):
                    with open(filepath) as f:
                        data = json.load(f)
                    scenarios.append(Scenario.from_dict(data))

        return scenarios

    async def collect_trace(self, scenario: Scenario) -> TraceResult:
        """Collect trace for a single scenario."""
        return await self.executor.execute_scenario(scenario, use_buggy_flow=True)

    async def collect_all_traces(
        self,
        scenarios: list[Scenario],
        max_concurrent: int = 5
    ) -> list[TraceResult]:
        """Collect traces for all scenarios with concurrency control."""
        semaphore = asyncio.Semaphore(max_concurrent)

        async def bounded_collect(scenario: Scenario) -> TraceResult:
            async with semaphore:
                print(f"  Collecting trace for {scenario.scenario_id}...")
                result = await self.collect_trace(scenario)
                self.save_trace(result)
                return result

        results = await asyncio.gather(
            *[bounded_collect(s) for s in scenarios],
            return_exceptions=True
        )

        valid_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                print(f"  Error collecting trace for {scenarios[i].scenario_id}: {result}")
            else:
                valid_results.append(result)

        return valid_results

    def save_trace(self, trace_result: TraceResult) -> Path:
        """Save a trace result to disk."""
        filepath = self.traces_dir / f"{trace_result.scenario_id}_trace.json"
        with open(filepath, 'w') as f:
            json.dump(asdict(trace_result), f, indent=2)
        return filepath

    def load_trace(self, scenario_id: str) -> Optional[TraceResult]:
        """Load a trace result from disk."""
        filepath = self.traces_dir / f"{scenario_id}_trace.json"
        if filepath.exists():
            with open(filepath) as f:
                data = json.load(f)
            return TraceResult(**data)
        return None


async def collect_all_benchmark_traces(
    scenarios_dir: Path = Path("data/scenarios"),
    traces_dir: Path = Path("data/traces"),
    use_llm: bool = True,
    max_concurrent: int = 5
) -> dict[str, list[TraceResult]]:
    """Collect traces for the entire benchmark."""
    collector = TraceCollector(
        scenarios_dir=scenarios_dir,
        traces_dir=traces_dir,
        use_llm=use_llm
    )

    results_by_domain = {}

    for domain in ScenarioDomain:
        print(f"\n{'='*50}")
        print(f"Collecting traces for {domain.value}...")
        print('='*50)

        scenarios = collector.load_scenarios(domain)
        if scenarios:
            results = await collector.collect_all_traces(scenarios, max_concurrent)
            results_by_domain[domain.value] = results
            print(f"  Collected {len(results)}/{len(scenarios)} traces")
        else:
            print(f"  No scenarios found for {domain.value}")

    return results_by_domain


# CLI interface
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Collect traces for scenarios")
    parser.add_argument("--domain", type=str, choices=[d.value for d in ScenarioDomain],
                        help="Collect for specific domain only")
    parser.add_argument("--scenarios-dir", type=str, default="data/scenarios",
                        help="Scenarios directory")
    parser.add_argument("--traces-dir", type=str, default="data/traces",
                        help="Output traces directory")
    parser.add_argument("--no-llm", action="store_true",
                        help="Use simple simulation instead of LLM")
    parser.add_argument("--max-concurrent", type=int, default=5,
                        help="Maximum concurrent executions")
    parser.add_argument("--single", type=str,
                        help="Collect trace for a single scenario file")

    args = parser.parse_args()

    async def main():
        if args.single:
            # Collect single trace
            from .scenario_generator import Scenario

            with open(args.single) as f:
                data = json.load(f)
            scenario = Scenario.from_dict(data)

            collector = TraceCollector(
                scenarios_dir=Path(args.scenarios_dir),
                traces_dir=Path(args.traces_dir),
                use_llm=not args.no_llm
            )

            print(f"Collecting trace for {scenario.scenario_id}...")
            result = await collector.collect_trace(scenario)
            filepath = collector.save_trace(result)

            print(f"Saved to: {filepath}")
            print(f"Nodes: {result.node_count}, Edges: {result.edge_count}")
            print(f"Error occurred: {result.error_occurred}")
            if result.error_message:
                print(f"Error message: {result.error_message}")
        else:
            # Collect all traces
            await collect_all_benchmark_traces(
                scenarios_dir=Path(args.scenarios_dir),
                traces_dir=Path(args.traces_dir),
                use_llm=not args.no_llm,
                max_concurrent=args.max_concurrent
            )

    asyncio.run(main())
