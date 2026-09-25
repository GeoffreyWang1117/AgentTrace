"""
Advanced Multi-Agent Scenarios for Real Trace Collection.

Creates more complex and challenging scenarios:
1. Longer causal chains (5+ steps)
2. Multiple agents interacting
3. Concurrent operations
4. Cascading failures
"""

import json
import asyncio
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Tuple
import sys
import random

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from experiments.real_traces.trace_collector import RealTraceCollector


class DataPipelineWorkflow:
    """
    Simulates a data processing pipeline with multiple stages.

    Stages: Extract -> Validate -> Transform -> Aggregate -> Report

    This creates longer causal chains (5+ steps).
    """

    async def run(
        self,
        data_source: str,
        inject_bug: bool = True,
        bug_step: int = 2  # Which step to inject bug
    ) -> Tuple[RealTraceCollector, Dict[str, Any]]:
        """Run data pipeline workflow."""
        scenario_id = f"pipeline_{uuid.uuid4().hex[:8]}"
        collector = RealTraceCollector(scenario_id)
        ground_truth = {
            'scenario_id': scenario_id,
            'workflow': 'data_pipeline',
            'bug_injected': inject_bug,
            'bug_step': bug_step if inject_bug else None
        }

        # Step 1: Extract data
        extracted_data = f"Extracted data from {data_source}: [row1, row2, row3, row4, row5]"
        action1 = collector.record_action(
            agent_id="extractor",
            action_type="extract_data",
            input_data=data_source,
            output_data=extracted_data
        )

        # Step 2: Validate data
        if inject_bug and bug_step == 2:
            validated_data = "[row1, row2, null, row4, row5]"  # Bug: null data passed through
            bug_desc = "Validation passed null values"
            action2 = collector.record_action(
                agent_id="validator",
                action_type="validate_data",
                input_data=extracted_data,
                output_data=validated_data,
                parent_action_ids=[action1]
            )
            ground_truth['root_cause_action_id'] = action2
            ground_truth['root_cause_node_id'] = collector.node_map[action2]
            ground_truth['bug_description'] = bug_desc
        else:
            validated_data = "[row1, row2, row3, row4, row5]"
            action2 = collector.record_action(
                agent_id="validator",
                action_type="validate_data",
                input_data=extracted_data,
                output_data=validated_data,
                parent_action_ids=[action1]
            )

        # Step 3: Transform data
        if inject_bug and bug_step == 3:
            transformed_data = "[processed_1, ERROR_2, processed_3]"
            bug_desc = "Transform failed on corrupted input"
            action3 = collector.record_action(
                agent_id="transformer",
                action_type="transform_data",
                input_data=validated_data,
                output_data=transformed_data,
                parent_action_ids=[action2]
            )
            ground_truth['root_cause_action_id'] = action3
            ground_truth['root_cause_node_id'] = collector.node_map[action3]
            ground_truth['bug_description'] = bug_desc
        else:
            transformed_data = "[processed_1, processed_2, processed_3, processed_4, processed_5]"
            action3 = collector.record_action(
                agent_id="transformer",
                action_type="transform_data",
                input_data=validated_data,
                output_data=transformed_data,
                parent_action_ids=[action2]
            )

        # Step 4: Aggregate data
        aggregated_data = f"Aggregated: count=5, sum=100, avg=20"
        action4 = collector.record_action(
            agent_id="aggregator",
            action_type="aggregate_data",
            input_data=transformed_data,
            output_data=aggregated_data,
            parent_action_ids=[action3]
        )

        # Step 5: Generate report (error manifests here)
        if inject_bug:
            report = "ERROR: Report generation failed - invalid data in pipeline"
            action5 = collector.record_action(
                agent_id="reporter",
                action_type="generate_report",
                input_data=aggregated_data,
                output_data=report,
                parent_action_ids=[action4],
                is_error=True,
                error_message="Invalid data detected"
            )
            ground_truth['error_action_id'] = action5
            ground_truth['error_node_id'] = collector.node_map[action5]
        else:
            report = "Report: Pipeline completed successfully"
            action5 = collector.record_action(
                agent_id="reporter",
                action_type="generate_report",
                input_data=aggregated_data,
                output_data=report,
                parent_action_ids=[action4]
            )

        return collector, ground_truth


class MultiAgentDebateWorkflow:
    """
    Simulates a multi-agent debate/discussion scenario.

    Agents: Proposer, Critic, Mediator, Decider

    Creates complex interaction patterns.
    """

    async def run(
        self,
        topic: str,
        inject_bug: bool = True,
        bug_type: str = "misunderstanding"
    ) -> Tuple[RealTraceCollector, Dict[str, Any]]:
        """Run debate workflow."""
        scenario_id = f"debate_{uuid.uuid4().hex[:8]}"
        collector = RealTraceCollector(scenario_id)
        ground_truth = {
            'scenario_id': scenario_id,
            'workflow': 'debate',
            'bug_injected': inject_bug,
            'bug_type': bug_type if inject_bug else None
        }

        # Round 1: Proposer makes initial argument
        proposal = f"I propose that {topic} should be implemented because of benefits A, B, C"
        action1 = collector.record_action(
            agent_id="proposer",
            action_type="make_proposal",
            input_data=topic,
            output_data=proposal
        )

        # Round 2: Critic responds
        if inject_bug and bug_type == "misunderstanding":
            # Bug: Critic misunderstands the proposal
            critique = "I disagree because of point D (which wasn't mentioned)"
            action2 = collector.record_action(
                agent_id="critic",
                action_type="provide_critique",
                input_data=proposal,
                output_data=critique,
                parent_action_ids=[action1]
            )
            ground_truth['root_cause_action_id'] = action2
            ground_truth['root_cause_node_id'] = collector.node_map[action2]
            ground_truth['bug_description'] = "Critic misunderstood the original proposal"
        else:
            critique = "I have concerns about benefit B - here's why..."
            action2 = collector.record_action(
                agent_id="critic",
                action_type="provide_critique",
                input_data=proposal,
                output_data=critique,
                parent_action_ids=[action1]
            )

        # Round 3: Proposer responds to critique
        response = f"Addressing the critique: {critique[:30]}..."
        action3 = collector.record_action(
            agent_id="proposer",
            action_type="respond_to_critique",
            input_data=critique,
            output_data=response,
            parent_action_ids=[action2]
        )

        # Round 4: Mediator summarizes
        summary = "Both sides have valid points. Key issues are..."
        action4 = collector.record_action(
            agent_id="mediator",
            action_type="summarize_debate",
            input_data={'proposal': proposal, 'critique': critique, 'response': response},
            output_data=summary,
            parent_action_ids=[action3]
        )

        # Round 5: Decider makes final call
        if inject_bug:
            decision = "ERROR: Cannot make decision - arguments are based on misunderstanding"
            action5 = collector.record_action(
                agent_id="decider",
                action_type="make_decision",
                input_data=summary,
                output_data=decision,
                parent_action_ids=[action4],
                is_error=True
            )
            ground_truth['error_action_id'] = action5
            ground_truth['error_node_id'] = collector.node_map[action5]
        else:
            decision = f"Decision: Proceed with {topic} with modifications"
            action5 = collector.record_action(
                agent_id="decider",
                action_type="make_decision",
                input_data=summary,
                output_data=decision,
                parent_action_ids=[action4]
            )

        return collector, ground_truth


class DistributedTaskWorkflow:
    """
    Simulates a distributed task execution with multiple workers.

    Agents: Coordinator, Worker1, Worker2, Worker3, Aggregator

    Creates parallel execution patterns.
    """

    async def run(
        self,
        task: str,
        inject_bug: bool = True,
        failing_worker: int = 2
    ) -> Tuple[RealTraceCollector, Dict[str, Any]]:
        """Run distributed task workflow."""
        scenario_id = f"distributed_{uuid.uuid4().hex[:8]}"
        collector = RealTraceCollector(scenario_id)
        ground_truth = {
            'scenario_id': scenario_id,
            'workflow': 'distributed',
            'bug_injected': inject_bug,
            'failing_worker': failing_worker if inject_bug else None
        }

        # Step 1: Coordinator splits task
        subtasks = [f"subtask_{i}" for i in range(1, 4)]
        action_coord = collector.record_action(
            agent_id="coordinator",
            action_type="split_task",
            input_data=task,
            output_data=subtasks
        )

        # Step 2: Workers execute subtasks (parallel in real system)
        worker_actions = []
        for i, subtask in enumerate(subtasks, 1):
            worker_id = f"worker_{i}"

            if inject_bug and i == failing_worker:
                # This worker fails
                result = f"ERROR: Worker {i} failed processing {subtask}"
                action = collector.record_action(
                    agent_id=worker_id,
                    action_type="process_subtask",
                    input_data=subtask,
                    output_data=result,
                    parent_action_ids=[action_coord],
                    is_error=True
                )
                ground_truth['root_cause_action_id'] = action
                ground_truth['root_cause_node_id'] = collector.node_map[action]
                ground_truth['bug_description'] = f"Worker {i} failed to process subtask"
            else:
                result = f"Completed: {subtask} -> result_{i}"
                action = collector.record_action(
                    agent_id=worker_id,
                    action_type="process_subtask",
                    input_data=subtask,
                    output_data=result,
                    parent_action_ids=[action_coord]
                )

            worker_actions.append(action)

        # Step 3: Aggregator combines results
        all_results = [collector.actions[-3 + i].output_data for i in range(3)]
        if inject_bug:
            aggregated = "PARTIAL: Some subtasks failed"
            action_agg = collector.record_action(
                agent_id="aggregator",
                action_type="aggregate_results",
                input_data=all_results,
                output_data=aggregated,
                parent_action_ids=worker_actions,
                is_error=True
            )
            ground_truth['error_action_id'] = action_agg
            ground_truth['error_node_id'] = collector.node_map[action_agg]
        else:
            aggregated = "SUCCESS: All subtasks completed"
            action_agg = collector.record_action(
                agent_id="aggregator",
                action_type="aggregate_results",
                input_data=all_results,
                output_data=aggregated,
                parent_action_ids=worker_actions
            )

        return collector, ground_truth


class APIOrchestrationWorkflow:
    """
    Simulates an API orchestration scenario.

    Agents: Gateway, AuthService, DataService, CacheService, ResponseBuilder

    Common in microservices architectures.
    """

    async def run(
        self,
        request: Dict[str, Any],
        inject_bug: bool = True,
        bug_type: str = "auth_failure"
    ) -> Tuple[RealTraceCollector, Dict[str, Any]]:
        """Run API orchestration workflow."""
        scenario_id = f"api_{uuid.uuid4().hex[:8]}"
        collector = RealTraceCollector(scenario_id)
        ground_truth = {
            'scenario_id': scenario_id,
            'workflow': 'api_orchestration',
            'bug_injected': inject_bug,
            'bug_type': bug_type if inject_bug else None
        }

        # Step 1: Gateway receives request
        action1 = collector.record_action(
            agent_id="gateway",
            action_type="receive_request",
            input_data=request,
            output_data={"request_id": uuid.uuid4().hex[:8], **request}
        )

        # Step 2: Auth service validates
        if inject_bug and bug_type == "auth_failure":
            auth_result = {"valid": True, "user_id": None}  # Bug: missing user_id
            action2 = collector.record_action(
                agent_id="auth_service",
                action_type="validate_auth",
                input_data=request.get("token", ""),
                output_data=auth_result,
                parent_action_ids=[action1]
            )
            ground_truth['root_cause_action_id'] = action2
            ground_truth['root_cause_node_id'] = collector.node_map[action2]
            ground_truth['bug_description'] = "Auth returned null user_id"
        else:
            auth_result = {"valid": True, "user_id": "user_123"}
            action2 = collector.record_action(
                agent_id="auth_service",
                action_type="validate_auth",
                input_data=request.get("token", ""),
                output_data=auth_result,
                parent_action_ids=[action1]
            )

        # Step 3: Cache lookup
        cache_result = {"hit": False, "data": None}
        action3 = collector.record_action(
            agent_id="cache_service",
            action_type="lookup_cache",
            input_data=request,
            output_data=cache_result,
            parent_action_ids=[action2]
        )

        # Step 4: Data service fetch
        if inject_bug and bug_type == "data_error":
            data_result = {"error": "Database connection timeout"}
            action4 = collector.record_action(
                agent_id="data_service",
                action_type="fetch_data",
                input_data={"user_id": auth_result.get("user_id")},
                output_data=data_result,
                parent_action_ids=[action3]
            )
            ground_truth['root_cause_action_id'] = action4
            ground_truth['root_cause_node_id'] = collector.node_map[action4]
            ground_truth['bug_description'] = "Database timeout"
        else:
            data_result = {"data": [{"id": 1, "value": "item1"}]}
            action4 = collector.record_action(
                agent_id="data_service",
                action_type="fetch_data",
                input_data={"user_id": auth_result.get("user_id")},
                output_data=data_result,
                parent_action_ids=[action3]
            )

        # Step 5: Build response (error manifests here)
        if inject_bug:
            response = {"status": 500, "error": "Internal error processing request"}
            action5 = collector.record_action(
                agent_id="response_builder",
                action_type="build_response",
                input_data={"auth": auth_result, "data": data_result},
                output_data=response,
                parent_action_ids=[action4],
                is_error=True
            )
            ground_truth['error_action_id'] = action5
            ground_truth['error_node_id'] = collector.node_map[action5]
        else:
            response = {"status": 200, "data": data_result["data"]}
            action5 = collector.record_action(
                agent_id="response_builder",
                action_type="build_response",
                input_data={"auth": auth_result, "data": data_result},
                output_data=response,
                parent_action_ids=[action4]
            )

        return collector, ground_truth


async def collect_advanced_traces(num_traces: int = 20):
    """Collect advanced traces from complex scenarios."""
    print("="*70)
    print("COLLECTING ADVANCED TRACES")
    print("="*70)

    output_dir = Path("data/real_traces")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load existing ground truth if available
    gt_file = output_dir / "ground_truth.json"
    if gt_file.exists():
        with open(gt_file) as f:
            all_ground_truth = json.load(f)
    else:
        all_ground_truth = {}

    trace_count = 0

    # Data Pipeline scenarios
    print("\n--- DATA PIPELINE SCENARIOS ---")
    pipeline = DataPipelineWorkflow()
    for bug_step in [2, 3]:
        for source in ["database", "api", "file"]:
            if trace_count >= num_traces:
                break
            print(f"  Collecting pipeline trace (bug at step {bug_step})...")
            collector, gt = await pipeline.run(source, inject_bug=True, bug_step=bug_step)
            collector.save_trace(output_dir)
            all_ground_truth[gt['scenario_id']] = gt
            print(f"    ✓ {gt['scenario_id']}: {collector.graph.node_count} nodes")
            trace_count += 1

    # Debate scenarios
    print("\n--- DEBATE SCENARIOS ---")
    debate = MultiAgentDebateWorkflow()
    for topic in ["new feature implementation", "architecture redesign", "budget allocation"]:
        if trace_count >= num_traces:
            break
        print(f"  Collecting debate trace: {topic[:30]}...")
        collector, gt = await debate.run(topic, inject_bug=True)
        collector.save_trace(output_dir)
        all_ground_truth[gt['scenario_id']] = gt
        print(f"    ✓ {gt['scenario_id']}: {collector.graph.node_count} nodes")
        trace_count += 1

    # Distributed task scenarios
    print("\n--- DISTRIBUTED TASK SCENARIOS ---")
    distributed = DistributedTaskWorkflow()
    for worker in [1, 2, 3]:
        if trace_count >= num_traces:
            break
        print(f"  Collecting distributed trace (worker {worker} fails)...")
        collector, gt = await distributed.run("process batch", inject_bug=True, failing_worker=worker)
        collector.save_trace(output_dir)
        all_ground_truth[gt['scenario_id']] = gt
        print(f"    ✓ {gt['scenario_id']}: {collector.graph.node_count} nodes")
        trace_count += 1

    # API orchestration scenarios
    print("\n--- API ORCHESTRATION SCENARIOS ---")
    api = APIOrchestrationWorkflow()
    for bug_type in ["auth_failure", "data_error"]:
        if trace_count >= num_traces:
            break
        print(f"  Collecting API trace ({bug_type})...")
        collector, gt = await api.run({"endpoint": "/users", "token": "abc123"}, inject_bug=True, bug_type=bug_type)
        collector.save_trace(output_dir)
        all_ground_truth[gt['scenario_id']] = gt
        print(f"    ✓ {gt['scenario_id']}: {collector.graph.node_count} nodes")
        trace_count += 1

    # Save updated ground truth
    with open(gt_file, 'w') as f:
        json.dump(all_ground_truth, f, indent=2, default=str)

    print(f"\n{'='*70}")
    print(f"ADVANCED COLLECTION COMPLETE")
    print(f"{'='*70}")
    print(f"New traces: {trace_count}")
    print(f"Total ground truth entries: {len(all_ground_truth)}")


if __name__ == "__main__":
    asyncio.run(collect_advanced_traces(20))
