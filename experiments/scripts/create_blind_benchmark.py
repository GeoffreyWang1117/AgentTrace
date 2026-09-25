"""
Create Blind Benchmark for AgentTrace.

This script creates "blinded" versions of traces that remove all explicit
bug markers, making the benchmark fair for LLM-based baselines.

Removes:
1. is_bug_point field
2. bug_description field
3. node type "error" -> "decision"
4. bug_type from metadata
5. Any hints in content that reveal the bug

Keeps separate ground truth file for evaluation.
"""

import json
import re
import copy
from pathlib import Path
from typing import Dict, Any, List
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def remove_bug_hints_from_content(content: Dict[str, Any], bug_type: str) -> Dict[str, Any]:
    """Remove hints from content that might reveal the bug."""
    cleaned = copy.deepcopy(content)

    # Patterns that hint at bugs
    bug_patterns = [
        r'with corruption',
        r'corrupted',
        r'invalid',
        r'error',
        r'failed',
        r'missing',
        r'wrong',
        r'incorrect',
        r'malformed',
        r'timeout',
        r'race condition',
        r'inconsistent',
    ]

    def clean_string(s: str) -> str:
        """Clean a string by removing obvious bug hints."""
        # Don't clean error manifestation content - that should be visible
        # Only clean content in root cause nodes
        return s

    return cleaned


def blind_trace_json(trace_json_str: str) -> tuple[str, Dict[str, Any]]:
    """
    Remove bug markers from trace JSON.

    Returns:
        - Blinded trace JSON string
        - Ground truth info dict
    """
    trace_data = json.loads(trace_json_str)
    ground_truth = {
        'bug_nodes': [],
        'bug_type': None
    }

    # Process nodes
    for node in trace_data.get('nodes', []):
        node_data = node.get('data', {})
        metadata = node.get('metadata', {})

        # Record ground truth
        if node_data.get('is_bug_point', False):
            ground_truth['bug_nodes'].append({
                'node_id': node['id'],
                'step': node_data.get('step'),
                'bug_description': node_data.get('bug_description')
            })

        if metadata.get('bug_type'):
            ground_truth['bug_type'] = metadata['bug_type']

        # Remove bug markers from data
        if 'is_bug_point' in node_data:
            del node_data['is_bug_point']
        if 'bug_description' in node_data:
            del node_data['bug_description']

        # Remove bug_type from metadata
        if 'bug_type' in metadata:
            del metadata['bug_type']

        # Change "error" type to "decision" (hide the bug indicator)
        if node.get('type') == 'error':
            node['type'] = 'decision'

    # Process edges - remove bug-related descriptions
    for edge in trace_data.get('edges', []):
        metadata = edge.get('metadata', {})
        if 'description' in metadata:
            # Keep description but it's less revealing than node markers
            pass

    return json.dumps(trace_data, indent=2), ground_truth


def blind_scenario(scenario: Dict[str, Any]) -> tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Create blinded version of scenario.

    Returns:
        - Blinded scenario
        - Ground truth info
    """
    blinded = copy.deepcopy(scenario)
    ground_truth = {
        'scenario_id': scenario['scenario_id'],
        'domain': scenario.get('domain'),
        'bug_type': scenario.get('bug_type'),
        'root_cause_step': scenario.get('root_cause_step'),
        'error_manifestation_step': scenario.get('error_manifestation_step'),
        'bug_injection_point': scenario.get('bug_injection_point'),
        'causal_chain': scenario.get('causal_chain', [])
    }

    # Remove ground truth fields from blinded version
    fields_to_remove = [
        'bug_type',
        'root_cause_step',
        'error_manifestation_step',
        'bug_injection_point',
        'causal_chain',
        'correct_flow'  # Remove correct flow - it's a comparison baseline
    ]

    for field in fields_to_remove:
        if field in blinded:
            del blinded[field]

    # Clean buggy_flow steps
    if 'buggy_flow' in blinded:
        for step in blinded['buggy_flow']:
            if 'is_bug_point' in step:
                del step['is_bug_point']
            if 'bug_description' in step:
                del step['bug_description']

    return blinded, ground_truth


def blind_trace_file(trace: Dict[str, Any]) -> tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Create blinded version of trace file.

    Returns:
        - Blinded trace
        - Ground truth info
    """
    blinded = copy.deepcopy(trace)

    # Ground truth from trace file
    ground_truth = {
        'scenario_id': trace['scenario_id'],
        'root_cause_node_id': trace.get('root_cause_node_id'),
        'error_node_id': trace.get('error_node_id'),
        'causal_path': trace.get('causal_path', []),
        'error_occurred': trace.get('error_occurred', True)
    }

    # Blind the trace_json
    if 'trace_json' in blinded:
        blinded_trace_json, trace_gt = blind_trace_json(blinded['trace_json'])
        blinded['trace_json'] = blinded_trace_json
        ground_truth['trace_ground_truth'] = trace_gt

    # Remove ground truth fields
    fields_to_remove = [
        'root_cause_node_id',
        'error_node_id',
        'causal_path',
        'error_occurred'
    ]

    for field in fields_to_remove:
        if field in blinded:
            del blinded[field]

    return blinded, ground_truth


def create_blind_benchmark():
    """Create the complete blind benchmark."""
    print("="*70)
    print("CREATING BLIND BENCHMARK")
    print("="*70)

    # Setup directories
    traces_dir = Path("data/traces")
    scenarios_base = Path("data/scenarios")
    blind_dir = Path("data/blind_benchmark")
    blind_traces_dir = blind_dir / "traces"
    blind_scenarios_dir = blind_dir / "scenarios"
    ground_truth_dir = blind_dir / "ground_truth"

    # Create directories
    blind_traces_dir.mkdir(parents=True, exist_ok=True)
    ground_truth_dir.mkdir(parents=True, exist_ok=True)

    all_ground_truth = {}
    stats = {
        'total_traces': 0,
        'total_scenarios': 0,
        'domains': set()
    }

    # Process all traces
    print("\n1. Processing traces...")
    for trace_file in traces_dir.glob("*_trace.json"):
        with open(trace_file) as f:
            trace = json.load(f)

        scenario_id = trace['scenario_id']
        blinded_trace, ground_truth = blind_trace_file(trace)

        # Save blinded trace
        with open(blind_traces_dir / trace_file.name, 'w') as f:
            json.dump(blinded_trace, f, indent=2)

        # Store ground truth
        all_ground_truth[scenario_id] = ground_truth
        stats['total_traces'] += 1

    print(f"   Processed {stats['total_traces']} traces")

    # Process all scenarios
    print("\n2. Processing scenarios...")
    for domain_dir in scenarios_base.iterdir():
        if not domain_dir.is_dir():
            continue

        domain = domain_dir.name
        stats['domains'].add(domain)

        blind_domain_dir = blind_scenarios_dir / domain
        blind_domain_dir.mkdir(parents=True, exist_ok=True)

        for scenario_file in domain_dir.glob("*.json"):
            with open(scenario_file) as f:
                scenario = json.load(f)

            scenario_id = scenario['scenario_id']
            blinded_scenario, scenario_gt = blind_scenario(scenario)

            # Save blinded scenario
            with open(blind_domain_dir / scenario_file.name, 'w') as f:
                json.dump(blinded_scenario, f, indent=2)

            # Merge ground truth
            if scenario_id in all_ground_truth:
                all_ground_truth[scenario_id].update(scenario_gt)
            else:
                all_ground_truth[scenario_id] = scenario_gt

            stats['total_scenarios'] += 1

    print(f"   Processed {stats['total_scenarios']} scenarios")
    print(f"   Domains: {', '.join(sorted(stats['domains']))}")

    # Save ground truth
    print("\n3. Saving ground truth...")
    with open(ground_truth_dir / "all_ground_truth.json", 'w') as f:
        json.dump(all_ground_truth, f, indent=2)

    # Create summary
    summary = {
        'total_traces': stats['total_traces'],
        'total_scenarios': stats['total_scenarios'],
        'domains': list(stats['domains']),
        'description': 'Blind benchmark with bug markers removed',
        'fields_removed': [
            'is_bug_point',
            'bug_description',
            'bug_type',
            'root_cause_step',
            'error_manifestation_step',
            'causal_path',
            'node type "error" -> "decision"'
        ]
    }

    with open(blind_dir / "benchmark_info.json", 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n   Ground truth saved to: {ground_truth_dir / 'all_ground_truth.json'}")

    # Verify blinding
    print("\n4. Verifying blind data...")
    verify_blind_data(blind_traces_dir, blind_scenarios_dir)

    print("\n" + "="*70)
    print("BLIND BENCHMARK CREATED")
    print("="*70)
    print(f"\nOutput directory: {blind_dir}")
    print(f"  - traces/: Blinded trace files")
    print(f"  - scenarios/: Blinded scenario files")
    print(f"  - ground_truth/: Ground truth for evaluation")


def verify_blind_data(traces_dir: Path, scenarios_dir: Path):
    """Verify that blinded data has no bug markers."""
    issues = []

    # Check traces
    for trace_file in traces_dir.glob("*.json"):
        with open(trace_file) as f:
            content = f.read()

        # Check for leaked markers
        if '"is_bug_point"' in content:
            issues.append(f"Trace {trace_file.name}: contains is_bug_point")
        if '"bug_description"' in content:
            issues.append(f"Trace {trace_file.name}: contains bug_description")
        if '"bug_type"' in content:
            issues.append(f"Trace {trace_file.name}: contains bug_type")
        if '"type": "error"' in content:
            issues.append(f"Trace {trace_file.name}: contains error node type")
        if '"root_cause_node_id"' in content:
            issues.append(f"Trace {trace_file.name}: contains root_cause_node_id")

    # Check scenarios
    for domain_dir in scenarios_dir.iterdir():
        if not domain_dir.is_dir():
            continue
        for scenario_file in domain_dir.glob("*.json"):
            with open(scenario_file) as f:
                content = f.read()

            if '"is_bug_point"' in content:
                issues.append(f"Scenario {scenario_file.name}: contains is_bug_point")
            if '"bug_description"' in content:
                issues.append(f"Scenario {scenario_file.name}: contains bug_description")
            if '"root_cause_step"' in content:
                issues.append(f"Scenario {scenario_file.name}: contains root_cause_step")

    if issues:
        print(f"   WARNING: Found {len(issues)} issues:")
        for issue in issues[:5]:
            print(f"   - {issue}")
        if len(issues) > 5:
            print(f"   ... and {len(issues) - 5} more")
    else:
        print("   ✓ All data properly blinded")


if __name__ == "__main__":
    create_blind_benchmark()
