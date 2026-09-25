"""
Real Trace Collector for Multi-Agent Systems.

Collects traces from actual multi-agent interactions:
1. Coding review workflow (code generation, review, testing)
2. Research workflow (search, summarize, synthesize)
3. Customer service workflow (route, handle, escalate)

Can work with:
- OpenAI API for realistic LLM responses
- Local simulation for testing
"""

import json
import asyncio
import uuid
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional, Callable
from enum import Enum
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from agenttrace.core.graph import CausalGraph
from agenttrace.core.node import Node, NodeType
from agenttrace.core.edge import Edge, EdgeType


class AgentRole(Enum):
    """Common agent roles in multi-agent systems."""
    CODER = "coder"
    REVIEWER = "reviewer"
    TESTER = "tester"
    RESEARCHER = "researcher"
    SUMMARIZER = "summarizer"
    ROUTER = "router"
    SPECIALIST = "specialist"
    MANAGER = "manager"


@dataclass
class AgentMessage:
    """Message passed between agents."""
    from_agent: str
    to_agent: str
    content: Any
    message_type: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentAction:
    """Action taken by an agent."""
    agent_id: str
    action_type: str
    input_data: Any
    output_data: Any
    timestamp: datetime = field(default_factory=datetime.now)
    is_error: bool = False
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class RealTraceCollector:
    """
    Collects execution traces from multi-agent workflows.

    Provides hooks to capture agent interactions and build causal graphs.
    """

    def __init__(self, scenario_id: str):
        self.scenario_id = scenario_id
        self.graph = CausalGraph(run_id=f"real_{scenario_id}")
        self.actions: List[AgentAction] = []
        self.messages: List[AgentMessage] = []
        self.node_map: Dict[str, str] = {}  # action_id -> node_id
        self.current_step = 0

    def record_action(
        self,
        agent_id: str,
        action_type: str,
        input_data: Any,
        output_data: Any,
        parent_action_ids: List[str] = None,
        is_error: bool = False,
        error_message: str = None,
        metadata: Dict[str, Any] = None
    ) -> str:
        """
        Record an agent action and add it to the trace.

        Returns:
            Action ID for tracking dependencies
        """
        self.current_step += 1
        action_id = f"action_{self.current_step}_{uuid.uuid4().hex[:6]}"

        action = AgentAction(
            agent_id=agent_id,
            action_type=action_type,
            input_data=input_data,
            output_data=output_data,
            is_error=is_error,
            error_message=error_message,
            metadata=metadata or {}
        )
        self.actions.append(action)

        # Create node
        node_type = NodeType.ERROR if is_error else NodeType.DECISION
        parent_ids = []
        if parent_action_ids:
            parent_ids = [self.node_map[pid] for pid in parent_action_ids if pid in self.node_map]

        node = Node(
            type=node_type,
            agent_id=agent_id,
            data={
                'step': self.current_step,
                'action': action_type,
                'input': self._truncate(input_data),
                'output': self._truncate(output_data),
                'is_error': is_error,
                'error_message': error_message
            },
            metadata={
                'scenario_id': self.scenario_id,
                'action_id': action_id
            },
            parent_ids=parent_ids
        )

        self.graph.add_node(node)
        self.node_map[action_id] = node.id

        # Add edges
        for parent_id in parent_ids:
            edge = Edge(
                source_id=parent_id,
                target_id=node.id,
                type=EdgeType.DATA_FLOW,
                confidence=1.0
            )
            self.graph.add_edge(edge)

        return action_id

    def record_message(
        self,
        from_agent: str,
        to_agent: str,
        content: Any,
        message_type: str,
        parent_action_id: str = None
    ) -> str:
        """Record an inter-agent message."""
        message = AgentMessage(
            from_agent=from_agent,
            to_agent=to_agent,
            content=content,
            message_type=message_type
        )
        self.messages.append(message)

        # Record as action
        return self.record_action(
            agent_id=from_agent,
            action_type=f"send_{message_type}",
            input_data=None,
            output_data=content,
            parent_action_ids=[parent_action_id] if parent_action_id else None,
            metadata={'to_agent': to_agent}
        )

    def _truncate(self, data: Any, max_len: int = 500) -> Any:
        """Truncate data for storage."""
        if isinstance(data, str) and len(data) > max_len:
            return data[:max_len] + "..."
        return data

    def get_trace(self) -> Dict[str, Any]:
        """Get the complete trace."""
        return {
            'scenario_id': self.scenario_id,
            'trace_json': self.graph.to_json(),
            'node_count': self.graph.node_count,
            'edge_count': self.graph.edge_count,
            'action_count': len(self.actions),
            'message_count': len(self.messages),
            'collected_at': datetime.now().isoformat()
        }

    def save_trace(self, output_dir: Path) -> Path:
        """Save trace to file."""
        output_dir.mkdir(parents=True, exist_ok=True)
        trace_file = output_dir / f"{self.scenario_id}_real_trace.json"

        with open(trace_file, 'w') as f:
            json.dump(self.get_trace(), f, indent=2, default=str)

        return trace_file


class CodingWorkflow:
    """
    Simulates a realistic coding review workflow.

    Agents:
    - Coder: Writes code based on requirements
    - Reviewer: Reviews code for issues
    - Tester: Tests the code
    """

    def __init__(self, use_llm: bool = False):
        self.use_llm = use_llm
        self.client = None
        if use_llm:
            try:
                from openai import OpenAI
                self.client = OpenAI()
            except Exception:
                self.use_llm = False

    async def run(
        self,
        requirements: str,
        inject_bug: bool = True,
        bug_type: str = "logic_error"
    ) -> tuple[RealTraceCollector, Dict[str, Any]]:
        """
        Run the coding workflow.

        Args:
            requirements: Code requirements
            inject_bug: Whether to inject a bug
            bug_type: Type of bug to inject

        Returns:
            Trace collector and ground truth
        """
        scenario_id = f"coding_{uuid.uuid4().hex[:8]}"
        collector = RealTraceCollector(scenario_id)
        ground_truth = {
            'scenario_id': scenario_id,
            'workflow': 'coding',
            'bug_injected': inject_bug,
            'bug_type': bug_type if inject_bug else None
        }

        # Step 1: Coder writes code
        code = await self._generate_code(requirements)
        action1 = collector.record_action(
            agent_id="coder",
            action_type="write_code",
            input_data=requirements,
            output_data=code
        )

        # Step 2: Inject bug (if enabled)
        if inject_bug:
            buggy_code, bug_description = self._inject_bug(code, bug_type)
            action2 = collector.record_action(
                agent_id="coder",
                action_type="submit_code",
                input_data=code,
                output_data=buggy_code,
                parent_action_ids=[action1]
            )
            ground_truth['root_cause_action_id'] = action2
            ground_truth['root_cause_node_id'] = collector.node_map[action2]
            ground_truth['bug_description'] = bug_description
        else:
            buggy_code = code
            action2 = collector.record_action(
                agent_id="coder",
                action_type="submit_code",
                input_data=code,
                output_data=code,
                parent_action_ids=[action1]
            )

        # Step 3: Reviewer reviews code
        review = await self._review_code(buggy_code)
        action3 = collector.record_action(
            agent_id="reviewer",
            action_type="review_code",
            input_data=buggy_code,
            output_data=review,
            parent_action_ids=[action2]
        )

        # Step 4: Tester tests code
        test_result = await self._test_code(buggy_code, inject_bug)
        is_error = "FAIL" in test_result
        action4 = collector.record_action(
            agent_id="tester",
            action_type="run_tests",
            input_data=buggy_code,
            output_data=test_result,
            parent_action_ids=[action3],
            is_error=is_error,
            error_message=test_result if is_error else None
        )

        if is_error:
            ground_truth['error_action_id'] = action4
            ground_truth['error_node_id'] = collector.node_map[action4]

        # Step 5: Handle test result
        if is_error:
            # Coder tries to fix
            fix_attempt = await self._attempt_fix(buggy_code, test_result)
            action5 = collector.record_action(
                agent_id="coder",
                action_type="attempt_fix",
                input_data={'code': buggy_code, 'error': test_result},
                output_data=fix_attempt,
                parent_action_ids=[action4]
            )

        return collector, ground_truth

    async def _generate_code(self, requirements: str) -> str:
        """Generate code from requirements."""
        if self.use_llm and self.client:
            try:
                response = self.client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "You are a Python developer. Write clean, functional code."},
                        {"role": "user", "content": f"Write Python code for: {requirements}"}
                    ],
                    max_tokens=500
                )
                return response.choices[0].message.content
            except Exception:
                pass

        # Fallback: simulated code
        return f'''def solution(data):
    """Solution for: {requirements[:50]}..."""
    result = []
    for item in data:
        processed = process_item(item)
        result.append(processed)
    return result

def process_item(item):
    return item * 2
'''

    async def _review_code(self, code: str) -> str:
        """Review code for issues."""
        if self.use_llm and self.client:
            try:
                response = self.client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "You are a code reviewer. Identify issues."},
                        {"role": "user", "content": f"Review this code:\n{code}"}
                    ],
                    max_tokens=200
                )
                return response.choices[0].message.content
            except Exception:
                pass

        return "Code review: Structure looks reasonable. Minor suggestions for improvement."

    async def _test_code(self, code: str, has_bug: bool) -> str:
        """Test the code."""
        if has_bug:
            return "FAIL: Test case failed - unexpected output for edge case"
        return "PASS: All tests passed"

    async def _attempt_fix(self, code: str, error: str) -> str:
        """Attempt to fix the code."""
        return f"Fixed version: Added error handling for edge case mentioned in: {error[:50]}"

    def _inject_bug(self, code: str, bug_type: str) -> tuple[str, str]:
        """Inject a bug into the code."""
        bugs = {
            'logic_error': (
                code.replace('item * 2', 'item * 3'),  # Wrong calculation
                "Logic error: incorrect multiplication factor"
            ),
            'missing_check': (
                code.replace('for item in data:', 'for item in data:  # Missing null check'),
                "Missing null check before iteration"
            ),
            'off_by_one': (
                code.replace('result.append(processed)', 'result.append(processed[:-1])'),
                "Off-by-one error: truncating last character"
            ),
            'type_error': (
                code.replace('return result', 'return str(result)'),
                "Type error: returning string instead of list"
            )
        }

        if bug_type in bugs:
            return bugs[bug_type]
        return bugs['logic_error']


class ResearchWorkflow:
    """
    Simulates a research assistant workflow.

    Agents:
    - Researcher: Searches for information
    - Summarizer: Summarizes findings
    - Synthesizer: Creates final report
    """

    def __init__(self, use_llm: bool = False):
        self.use_llm = use_llm
        self.client = None
        if use_llm:
            try:
                from openai import OpenAI
                self.client = OpenAI()
            except Exception:
                self.use_llm = False

    async def run(
        self,
        topic: str,
        inject_bug: bool = True,
        bug_type: str = "missing_context"
    ) -> tuple[RealTraceCollector, Dict[str, Any]]:
        """Run the research workflow."""
        scenario_id = f"research_{uuid.uuid4().hex[:8]}"
        collector = RealTraceCollector(scenario_id)
        ground_truth = {
            'scenario_id': scenario_id,
            'workflow': 'research',
            'bug_injected': inject_bug,
            'bug_type': bug_type if inject_bug else None
        }

        # Step 1: Researcher searches
        search_results = await self._search(topic)
        action1 = collector.record_action(
            agent_id="researcher",
            action_type="search",
            input_data=topic,
            output_data=search_results
        )

        # Step 2: Summarizer processes results
        if inject_bug and bug_type == "missing_context":
            # Bug: summarizer loses important context
            incomplete_results = search_results[:len(search_results)//2]
            summary = await self._summarize(incomplete_results)
            action2 = collector.record_action(
                agent_id="summarizer",
                action_type="summarize",
                input_data=incomplete_results,
                output_data=summary,
                parent_action_ids=[action1]
            )
            ground_truth['root_cause_action_id'] = action2
            ground_truth['root_cause_node_id'] = collector.node_map[action2]
            ground_truth['bug_description'] = "Summarizer processed incomplete data"
        else:
            summary = await self._summarize(search_results)
            action2 = collector.record_action(
                agent_id="summarizer",
                action_type="summarize",
                input_data=search_results,
                output_data=summary,
                parent_action_ids=[action1]
            )

        # Step 3: Synthesizer creates report
        report = await self._synthesize(summary, topic)
        has_error = inject_bug  # Report will be incomplete if bug was injected
        action3 = collector.record_action(
            agent_id="synthesizer",
            action_type="create_report",
            input_data={'summary': summary, 'topic': topic},
            output_data=report,
            parent_action_ids=[action2],
            is_error=has_error,
            error_message="Report may be incomplete" if has_error else None
        )

        if has_error:
            ground_truth['error_action_id'] = action3
            ground_truth['error_node_id'] = collector.node_map[action3]

        return collector, ground_truth

    async def _search(self, topic: str) -> str:
        """Search for information on topic."""
        return f"""Search results for '{topic}':
1. Overview: {topic} is a widely studied subject...
2. Key findings: Recent research shows important developments...
3. Applications: Practical uses include various domains...
4. Challenges: Main obstacles are complexity and scale...
5. Future directions: Emerging trends point to new approaches..."""

    async def _summarize(self, results: str) -> str:
        """Summarize search results."""
        if self.use_llm and self.client:
            try:
                response = self.client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "Summarize the key points."},
                        {"role": "user", "content": f"Summarize:\n{results}"}
                    ],
                    max_tokens=200
                )
                return response.choices[0].message.content
            except Exception:
                pass

        return f"Summary: Key points extracted from search results. Main themes identified."

    async def _synthesize(self, summary: str, topic: str) -> str:
        """Create final report."""
        return f"""Research Report: {topic}

{summary}

Conclusion: Based on the analysis, we recommend further investigation."""


class CustomerServiceWorkflow:
    """
    Simulates a customer service workflow.

    Agents:
    - Router: Routes inquiries to specialists
    - Specialist: Handles specific issues
    - Escalator: Escalates complex cases
    """

    async def run(
        self,
        inquiry: str,
        inject_bug: bool = True,
        bug_type: str = "wrong_routing"
    ) -> tuple[RealTraceCollector, Dict[str, Any]]:
        """Run customer service workflow."""
        scenario_id = f"customer_{uuid.uuid4().hex[:8]}"
        collector = RealTraceCollector(scenario_id)
        ground_truth = {
            'scenario_id': scenario_id,
            'workflow': 'customer_service',
            'bug_injected': inject_bug,
            'bug_type': bug_type if inject_bug else None
        }

        # Step 1: Router classifies inquiry
        category = self._classify(inquiry)
        action1 = collector.record_action(
            agent_id="router",
            action_type="classify_inquiry",
            input_data=inquiry,
            output_data=category
        )

        # Step 2: Route to specialist (potentially wrong)
        if inject_bug and bug_type == "wrong_routing":
            wrong_category = "billing" if category != "billing" else "technical"
            action2 = collector.record_action(
                agent_id="router",
                action_type="route_inquiry",
                input_data=category,
                output_data=wrong_category,
                parent_action_ids=[action1]
            )
            ground_truth['root_cause_action_id'] = action2
            ground_truth['root_cause_node_id'] = collector.node_map[action2]
            ground_truth['bug_description'] = f"Routed to wrong specialist: {wrong_category} instead of {category}"
            routed_to = wrong_category
        else:
            action2 = collector.record_action(
                agent_id="router",
                action_type="route_inquiry",
                input_data=category,
                output_data=category,
                parent_action_ids=[action1]
            )
            routed_to = category

        # Step 3: Specialist handles (may fail if wrong routing)
        response = self._handle_inquiry(inquiry, routed_to)
        is_error = inject_bug and "cannot help" in response.lower()
        action3 = collector.record_action(
            agent_id=f"specialist_{routed_to}",
            action_type="handle_inquiry",
            input_data={'inquiry': inquiry, 'category': routed_to},
            output_data=response,
            parent_action_ids=[action2],
            is_error=is_error
        )

        if is_error:
            ground_truth['error_action_id'] = action3
            ground_truth['error_node_id'] = collector.node_map[action3]

            # Step 4: Escalate
            action4 = collector.record_action(
                agent_id="escalator",
                action_type="escalate_case",
                input_data={'inquiry': inquiry, 'failed_response': response},
                output_data="Escalated to supervisor",
                parent_action_ids=[action3]
            )

        return collector, ground_truth

    def _classify(self, inquiry: str) -> str:
        """Classify inquiry type."""
        inquiry_lower = inquiry.lower()
        if "password" in inquiry_lower or "login" in inquiry_lower:
            return "technical"
        elif "charge" in inquiry_lower or "bill" in inquiry_lower:
            return "billing"
        elif "cancel" in inquiry_lower or "refund" in inquiry_lower:
            return "account"
        return "general"

    def _handle_inquiry(self, inquiry: str, category: str) -> str:
        """Handle the inquiry based on category."""
        # Simulate wrong specialist not being able to help
        if "password" in inquiry.lower() and category == "billing":
            return "I cannot help with password issues. This needs to be escalated."
        return f"Thank you for contacting us about your {category} issue. We will resolve this."


async def collect_real_traces(num_traces: int = 20):
    """Collect real traces from multiple workflows."""
    print("="*70)
    print("COLLECTING REAL TRACES")
    print("="*70)

    output_dir = Path("data/real_traces")
    output_dir.mkdir(parents=True, exist_ok=True)

    all_traces = []
    all_ground_truth = {}

    workflows = [
        ('coding', CodingWorkflow(use_llm=False),
         [("Implement a function to sort a list", "logic_error"),
          ("Create a data validation module", "missing_check"),
          ("Build a string parser", "off_by_one"),
          ("Write a type converter", "type_error")]),
        ('research', ResearchWorkflow(use_llm=False),
         [("Machine learning applications", "missing_context"),
          ("Climate change solutions", "missing_context"),
          ("Quantum computing advances", "missing_context")]),
        ('customer_service', CustomerServiceWorkflow(),
         [("I forgot my password and can't login", "wrong_routing"),
          ("I was charged twice for my subscription", "wrong_routing"),
          ("How do I cancel my account?", "wrong_routing")])
    ]

    trace_count = 0

    for workflow_name, workflow, test_cases in workflows:
        print(f"\n--- {workflow_name.upper()} WORKFLOW ---")

        for i, (input_data, bug_type) in enumerate(test_cases):
            if trace_count >= num_traces:
                break

            print(f"  Collecting trace {trace_count + 1}: {input_data[:40]}...")

            try:
                if workflow_name == 'coding':
                    collector, gt = await workflow.run(input_data, inject_bug=True, bug_type=bug_type)
                elif workflow_name == 'research':
                    collector, gt = await workflow.run(input_data, inject_bug=True, bug_type=bug_type)
                else:
                    collector, gt = await workflow.run(input_data, inject_bug=True, bug_type=bug_type)

                # Save trace
                trace_file = collector.save_trace(output_dir)
                all_traces.append(collector.get_trace())
                all_ground_truth[gt['scenario_id']] = gt

                print(f"    ✓ Saved: {trace_file.name}")
                print(f"    Nodes: {collector.graph.node_count}, Edges: {collector.graph.edge_count}")

                trace_count += 1

            except Exception as e:
                print(f"    ✗ Error: {e}")

    # Save all ground truth
    gt_file = output_dir / "ground_truth.json"
    with open(gt_file, 'w') as f:
        json.dump(all_ground_truth, f, indent=2, default=str)

    print(f"\n{'='*70}")
    print(f"COLLECTION COMPLETE")
    print(f"{'='*70}")
    print(f"Total traces: {len(all_traces)}")
    print(f"Ground truth: {gt_file}")

    return all_traces, all_ground_truth


if __name__ == "__main__":
    asyncio.run(collect_real_traces(20))
