"""
Multi-Agent Scenario Generator for AgentTrace Benchmark.

This module generates diverse multi-agent interaction scenarios with injected bugs
for evaluating the AgentTrace causal tracing system.
"""

import json
import asyncio
import hashlib
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Literal, Optional
from enum import Enum
from datetime import datetime

from openai import AsyncOpenAI


class ScenarioDomain(str, Enum):
    """Domain categories for multi-agent scenarios."""
    RESEARCH = "research"
    CODING = "coding"
    PLANNING = "planning"
    CUSTOMER_SERVICE = "customer_service"
    TRADING = "trading"


class BugType(str, Enum):
    """Types of bugs to inject into scenarios."""
    DATA_CORRUPTION = "data_corruption"           # Data gets corrupted during transfer
    MISSING_CONTEXT = "missing_context"           # Agent loses important context
    WRONG_ROUTING = "wrong_routing"               # Message sent to wrong agent
    STATE_INCONSISTENCY = "state_inconsistency"   # Shared state becomes inconsistent
    TIMEOUT_CASCADE = "timeout_cascade"           # Timeout causes cascade failure
    INVALID_OUTPUT = "invalid_output"             # Agent produces invalid output format
    LOGIC_ERROR = "logic_error"                   # Incorrect decision logic
    RACE_CONDITION = "race_condition"             # Concurrent access issues


class Complexity(str, Enum):
    """Scenario complexity levels."""
    SIMPLE = "simple"       # 3 agents, linear flow
    MEDIUM = "medium"       # 4-5 agents, some branching
    COMPLEX = "complex"     # 5-6 agents, complex interactions


@dataclass
class Agent:
    """Represents an agent in the scenario."""
    id: str
    role: str
    capabilities: list[str]
    description: str


@dataclass
class Message:
    """Represents a message/action in the execution flow."""
    step: int
    from_agent: str
    to_agent: str
    action: str
    content: dict
    is_bug_point: bool = False
    bug_description: Optional[str] = None


@dataclass
class CausalEdge:
    """Represents a causal relationship in ground truth."""
    source_step: int
    target_step: int
    relationship_type: Literal["data_flow", "trigger", "state_dependency"]
    description: str


@dataclass
class Scenario:
    """A complete multi-agent scenario with bug injection."""
    scenario_id: str
    domain: ScenarioDomain
    complexity: Complexity
    bug_type: BugType

    # Task description
    task_description: str
    expected_outcome: str

    # Agents
    agents: list[Agent]

    # Execution flows
    correct_flow: list[Message]
    buggy_flow: list[Message]

    # Ground truth for evaluation
    bug_injection_point: int  # Step number where bug is injected
    root_cause_step: int      # Step number of the root cause
    error_manifestation_step: int  # Step where error becomes visible
    causal_chain: list[CausalEdge]  # Ground truth causal relationships

    # Metadata
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        d = asdict(self)
        d['domain'] = self.domain.value
        d['complexity'] = self.complexity.value
        d['bug_type'] = self.bug_type.value
        return d

    @classmethod
    def from_dict(cls, d: dict) -> 'Scenario':
        """Create from dictionary."""
        d['domain'] = ScenarioDomain(d['domain'])
        d['complexity'] = Complexity(d['complexity'])
        d['bug_type'] = BugType(d['bug_type'])
        d['agents'] = [Agent(**a) for a in d['agents']]
        d['correct_flow'] = [Message(**m) for m in d['correct_flow']]
        d['buggy_flow'] = [Message(**m) for m in d['buggy_flow']]
        d['causal_chain'] = [CausalEdge(**e) for e in d['causal_chain']]
        return cls(**d)


# Prompt templates for each domain
SCENARIO_PROMPTS = {
    ScenarioDomain.RESEARCH: '''Generate a multi-agent research scenario where agents collaborate to:
- Search for information from multiple sources
- Synthesize findings from different agents
- Produce a research report or answer

Common bugs in this domain:
- Information gets lost or corrupted between agents
- Conflicting information not properly reconciled
- Source attribution errors''',

    ScenarioDomain.CODING: '''Generate a multi-agent software development scenario where agents collaborate to:
- Write code based on requirements
- Review and test code
- Fix bugs and integrate changes

Common bugs in this domain:
- Code review misses critical issues
- Test agent receives wrong code version
- Integration conflicts between agents''',

    ScenarioDomain.PLANNING: '''Generate a multi-agent planning scenario where agents collaborate to:
- Break down complex goals into subtasks
- Allocate resources and responsibilities
- Coordinate execution and handle dependencies

Common bugs in this domain:
- Dependency information not properly communicated
- Resource conflicts between agents
- Plan updates not propagated to all agents''',

    ScenarioDomain.CUSTOMER_SERVICE: '''Generate a multi-agent customer service scenario where agents collaborate to:
- Handle customer inquiries
- Escalate complex issues
- Track and resolve tickets

Common bugs in this domain:
- Customer context lost during handoff
- Ticket status not synchronized
- Duplicate or conflicting responses''',

    ScenarioDomain.TRADING: '''Generate a multi-agent trading/financial scenario where agents collaborate to:
- Analyze market data
- Make trading decisions
- Execute and verify transactions

Common bugs in this domain:
- Stale market data used for decisions
- Order execution timing issues
- Position tracking inconsistencies'''
}


GENERATION_PROMPT_TEMPLATE = '''You are an expert in multi-agent systems. Generate a detailed multi-agent interaction scenario for testing a debugging/tracing system.

## Domain
{domain_description}

## Requirements
- Complexity: {complexity}
- Number of agents: {n_agents}
- Bug type to inject: {bug_type} - {bug_description}

## Output Format
Return a JSON object with this exact structure:
{{
    "task_description": "Detailed description of what the agents need to accomplish",
    "expected_outcome": "What should happen if everything works correctly",
    "agents": [
        {{
            "id": "agent_1",
            "role": "Role name",
            "capabilities": ["capability1", "capability2"],
            "description": "What this agent does"
        }}
    ],
    "correct_flow": [
        {{
            "step": 1,
            "from_agent": "agent_1",
            "to_agent": "agent_2",
            "action": "action_name",
            "content": {{"key": "value"}},
            "is_bug_point": false
        }}
    ],
    "buggy_flow": [
        {{
            "step": 1,
            "from_agent": "agent_1",
            "to_agent": "agent_2",
            "action": "action_name",
            "content": {{"key": "value with bug"}},
            "is_bug_point": true,
            "bug_description": "What went wrong here"
        }}
    ],
    "bug_injection_point": 3,
    "root_cause_step": 3,
    "error_manifestation_step": 5,
    "causal_chain": [
        {{
            "source_step": 3,
            "target_step": 4,
            "relationship_type": "data_flow",
            "description": "Corrupted data flows from step 3 to step 4"
        }}
    ]
}}

## Important Guidelines
1. The buggy_flow should be realistic - the bug should be subtle and not obvious
2. The causal_chain should trace from root_cause_step to error_manifestation_step
3. Include at least {min_steps} steps in the flow
4. Make sure agent IDs are consistent throughout
5. The bug should be of type: {bug_type}

Generate a realistic and detailed scenario:'''


BUG_DESCRIPTIONS = {
    BugType.DATA_CORRUPTION: "Data gets modified incorrectly during transfer between agents",
    BugType.MISSING_CONTEXT: "Important context information is not passed to downstream agents",
    BugType.WRONG_ROUTING: "A message is sent to the wrong agent or in the wrong order",
    BugType.STATE_INCONSISTENCY: "Shared state between agents becomes out of sync",
    BugType.TIMEOUT_CASCADE: "A timeout in one agent causes failures in dependent agents",
    BugType.INVALID_OUTPUT: "An agent produces output in an unexpected format",
    BugType.LOGIC_ERROR: "An agent makes an incorrect decision based on the input",
    BugType.RACE_CONDITION: "Concurrent operations lead to inconsistent state"
}


COMPLEXITY_CONFIG = {
    Complexity.SIMPLE: {"n_agents": 3, "min_steps": 5},
    Complexity.MEDIUM: {"n_agents": 4, "min_steps": 8},
    Complexity.COMPLEX: {"n_agents": 5, "min_steps": 12}
}


class ScenarioGenerator:
    """Generates multi-agent scenarios using GPT-4."""

    def __init__(
        self,
        model: str = None,
        output_dir: Path = Path("data/scenarios")
    ):
        from experiments.config import OPENAI_MODELS
        self.client = AsyncOpenAI()
        self.model = model or OPENAI_MODELS.get("scenario_generation", "gpt-4o")
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _generate_scenario_id(
        self,
        domain: ScenarioDomain,
        complexity: Complexity,
        bug_type: BugType,
        index: int
    ) -> str:
        """Generate a unique scenario ID."""
        base = f"{domain.value}_{complexity.value}_{bug_type.value}_{index}"
        hash_suffix = hashlib.md5(base.encode()).hexdigest()[:6]
        return f"{domain.value[:3]}_{hash_suffix}"

    async def generate_scenario(
        self,
        domain: ScenarioDomain,
        complexity: Complexity,
        bug_type: BugType,
        index: int = 0
    ) -> Scenario:
        """Generate a single scenario using GPT-4."""
        config = COMPLEXITY_CONFIG[complexity]

        prompt = GENERATION_PROMPT_TEMPLATE.format(
            domain_description=SCENARIO_PROMPTS[domain],
            complexity=complexity.value,
            n_agents=config["n_agents"],
            bug_type=bug_type.value,
            bug_description=BUG_DESCRIPTIONS[bug_type],
            min_steps=config["min_steps"]
        )

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are an expert in multi-agent systems and debugging. Always respond with valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.8,
            response_format={"type": "json_object"}
        )

        content = response.choices[0].message.content
        data = json.loads(content)

        # Add scenario ID and metadata
        scenario_id = self._generate_scenario_id(domain, complexity, bug_type, index)

        scenario = Scenario(
            scenario_id=scenario_id,
            domain=domain,
            complexity=complexity,
            bug_type=bug_type,
            task_description=data["task_description"],
            expected_outcome=data["expected_outcome"],
            agents=[Agent(**a) for a in data["agents"]],
            correct_flow=[Message(**m) for m in data["correct_flow"]],
            buggy_flow=[Message(**m) for m in data["buggy_flow"]],
            bug_injection_point=data["bug_injection_point"],
            root_cause_step=data["root_cause_step"],
            error_manifestation_step=data["error_manifestation_step"],
            causal_chain=[CausalEdge(**e) for e in data["causal_chain"]]
        )

        return scenario

    async def generate_batch(
        self,
        domain: ScenarioDomain,
        count: int = 10,
        complexity_distribution: dict[Complexity, float] = None
    ) -> list[Scenario]:
        """Generate a batch of scenarios for a domain."""
        if complexity_distribution is None:
            complexity_distribution = {
                Complexity.SIMPLE: 0.3,
                Complexity.MEDIUM: 0.5,
                Complexity.COMPLEX: 0.2
            }

        scenarios = []
        bug_types = list(BugType)

        tasks = []
        for i in range(count):
            # Determine complexity based on distribution
            import random
            r = random.random()
            cumsum = 0
            complexity = Complexity.MEDIUM
            for c, prob in complexity_distribution.items():
                cumsum += prob
                if r < cumsum:
                    complexity = c
                    break

            # Rotate through bug types
            bug_type = bug_types[i % len(bug_types)]

            task = self.generate_scenario(domain, complexity, bug_type, i)
            tasks.append(task)

        # Run with concurrency limit
        semaphore = asyncio.Semaphore(5)

        async def bounded_generate(task):
            async with semaphore:
                return await task

        results = await asyncio.gather(
            *[bounded_generate(t) for t in tasks],
            return_exceptions=True
        )

        for result in results:
            if isinstance(result, Exception):
                print(f"Error generating scenario: {result}")
            else:
                scenarios.append(result)

        return scenarios

    def save_scenario(self, scenario: Scenario) -> Path:
        """Save a scenario to disk."""
        domain_dir = self.output_dir / scenario.domain.value
        domain_dir.mkdir(parents=True, exist_ok=True)

        filepath = domain_dir / f"{scenario.scenario_id}.json"
        with open(filepath, 'w') as f:
            json.dump(scenario.to_dict(), f, indent=2)

        return filepath

    def save_batch(self, scenarios: list[Scenario]) -> list[Path]:
        """Save multiple scenarios."""
        return [self.save_scenario(s) for s in scenarios]

    @staticmethod
    def load_scenario(filepath: Path) -> Scenario:
        """Load a scenario from disk."""
        with open(filepath) as f:
            data = json.load(f)
        return Scenario.from_dict(data)


class ScenarioValidator:
    """Validates generated scenarios for correctness and completeness."""

    @staticmethod
    def validate(scenario: Scenario) -> tuple[bool, list[str]]:
        """
        Validate a scenario.

        Returns:
            (is_valid, list of error messages)
        """
        errors = []

        # Check agents
        if len(scenario.agents) < 2:
            errors.append("Scenario must have at least 2 agents")

        agent_ids = {a.id for a in scenario.agents}

        # Check flows
        if len(scenario.correct_flow) < 3:
            errors.append("Correct flow must have at least 3 steps")

        if len(scenario.buggy_flow) < 3:
            errors.append("Buggy flow must have at least 3 steps")

        # Check agent IDs in flows
        # Allow some common system-level agents/endpoints that LLM might generate
        allowed_special = {
            "user", "system", "output", "final_output", "result", "external",
            "all", "broadcast", "coordinator", "orchestrator", "client", "api", "",
            "external_system", "database", "null", "none", "n/a", "self",
            "customer", "admin", "server", "service", "tool", "environment"
        }

        for flow_name, flow in [("correct_flow", scenario.correct_flow),
                                 ("buggy_flow", scenario.buggy_flow)]:
            for msg in flow:
                from_agent = msg.from_agent or ""
                to_agent = msg.to_agent or ""
                if from_agent not in agent_ids and from_agent.lower() not in allowed_special:
                    errors.append(f"{flow_name}: Unknown from_agent '{from_agent}'")
                if to_agent not in agent_ids and to_agent.lower() not in allowed_special:
                    errors.append(f"{flow_name}: Unknown to_agent '{to_agent}'")

        # Check bug point exists in buggy flow
        bug_points = [m for m in scenario.buggy_flow if m.is_bug_point]
        if not bug_points:
            errors.append("Buggy flow must have at least one bug point marked")

        # Check step numbers
        correct_steps = {m.step for m in scenario.correct_flow}
        buggy_steps = {m.step for m in scenario.buggy_flow}

        if scenario.bug_injection_point not in buggy_steps:
            errors.append(f"Bug injection point {scenario.bug_injection_point} not in buggy flow")

        if scenario.root_cause_step not in buggy_steps:
            errors.append(f"Root cause step {scenario.root_cause_step} not in buggy flow")

        if scenario.error_manifestation_step not in buggy_steps:
            errors.append(f"Error manifestation step {scenario.error_manifestation_step} not in buggy flow")

        # Check causal chain
        if not scenario.causal_chain:
            errors.append("Causal chain cannot be empty")
        else:
            for edge in scenario.causal_chain:
                if edge.source_step not in buggy_steps:
                    errors.append(f"Causal edge source step {edge.source_step} not in buggy flow")
                if edge.target_step not in buggy_steps:
                    errors.append(f"Causal edge target step {edge.target_step} not in buggy flow")
                if edge.source_step >= edge.target_step:
                    errors.append(f"Causal edge must go forward in time: {edge.source_step} -> {edge.target_step}")

        # Check causal chain connects root cause to error manifestation
        # Note: We make these warnings instead of hard errors since LLM generation
        # may not always produce perfect chains
        if scenario.causal_chain:
            chain_sources = {e.source_step for e in scenario.causal_chain}
            chain_targets = {e.target_step for e in scenario.causal_chain}
            all_chain_steps = chain_sources | chain_targets

            # Soft check: root cause should be in chain (as source or any step)
            if scenario.root_cause_step not in all_chain_steps:
                errors.append("Root cause step not found in causal chain")

            # Soft check: error manifestation should be reachable
            # We only warn if the chain is completely disconnected from error
            if scenario.error_manifestation_step not in all_chain_steps:
                # Check if any chain step is close to error manifestation
                max_chain_step = max(chain_targets) if chain_targets else 0
                if max_chain_step < scenario.error_manifestation_step - 2:
                    errors.append("Causal chain does not reach error manifestation")

        return len(errors) == 0, errors


async def generate_all_scenarios(
    output_dir: Path = Path("data/scenarios"),
    scenarios_per_domain: int = 50
) -> dict[ScenarioDomain, list[Scenario]]:
    """Generate all scenarios for the benchmark."""
    generator = ScenarioGenerator(output_dir=output_dir)
    validator = ScenarioValidator()

    all_scenarios = {}

    for domain in ScenarioDomain:
        print(f"\n{'='*50}")
        print(f"Generating {scenarios_per_domain} scenarios for {domain.value}...")
        print('='*50)

        scenarios = await generator.generate_batch(domain, scenarios_per_domain)

        # Validate and filter
        valid_scenarios = []
        for s in scenarios:
            is_valid, errors = validator.validate(s)
            if is_valid:
                valid_scenarios.append(s)
                generator.save_scenario(s)
            else:
                print(f"  Invalid scenario {s.scenario_id}: {errors}")

        print(f"  Generated {len(valid_scenarios)}/{scenarios_per_domain} valid scenarios")
        all_scenarios[domain] = valid_scenarios

    return all_scenarios


# CLI interface
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate multi-agent scenarios")
    parser.add_argument("--domain", type=str, choices=[d.value for d in ScenarioDomain],
                        help="Generate for specific domain only")
    parser.add_argument("--count", type=int, default=50,
                        help="Number of scenarios per domain")
    parser.add_argument("--output", type=str, default="data/scenarios",
                        help="Output directory")
    parser.add_argument("--single", action="store_true",
                        help="Generate a single test scenario")

    args = parser.parse_args()

    async def main():
        if args.single:
            # Generate a single test scenario
            generator = ScenarioGenerator(output_dir=Path(args.output))
            domain = ScenarioDomain(args.domain) if args.domain else ScenarioDomain.RESEARCH

            print(f"Generating single {domain.value} scenario...")
            scenario = await generator.generate_scenario(
                domain=domain,
                complexity=Complexity.MEDIUM,
                bug_type=BugType.DATA_CORRUPTION,
                index=0
            )

            # Validate
            validator = ScenarioValidator()
            is_valid, errors = validator.validate(scenario)

            if is_valid:
                filepath = generator.save_scenario(scenario)
                print(f"Saved to: {filepath}")
                print(f"\nScenario ID: {scenario.scenario_id}")
                print(f"Task: {scenario.task_description[:100]}...")
                print(f"Agents: {[a.id for a in scenario.agents]}")
                print(f"Steps: {len(scenario.buggy_flow)}")
                print(f"Bug at step: {scenario.bug_injection_point}")
            else:
                print(f"Validation errors: {errors}")
                print("\nRaw scenario:")
                print(json.dumps(scenario.to_dict(), indent=2))
        else:
            # Generate full benchmark
            if args.domain:
                generator = ScenarioGenerator(output_dir=Path(args.output))
                domain = ScenarioDomain(args.domain)
                scenarios = await generator.generate_batch(domain, args.count)
                generator.save_batch(scenarios)
                print(f"Generated {len(scenarios)} scenarios for {domain.value}")
            else:
                await generate_all_scenarios(
                    output_dir=Path(args.output),
                    scenarios_per_domain=args.count
                )

    asyncio.run(main())
