"""
Case Study Analysis for AgentTrace Paper.

Generates detailed case studies showing:
1. Successful cases where AgentTrace excels
2. Failure cases for error analysis
3. Comparison with baseline methods
"""

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from agenttrace.core.graph import CausalGraph
from experiments.benchmark.scenario_generator import ScenarioDomain


def load_all_data():
    """Load traces, scenarios, and baseline results."""
    traces_dir = Path("data/traces")
    scenarios_dir = Path("data/scenarios")
    results_dir = Path("data/results")

    data = {}

    for trace_file in traces_dir.glob("*_trace.json"):
        with open(trace_file) as f:
            trace = json.load(f)
            sid = trace['scenario_id']
            data[sid] = {'trace': trace}

    for domain in ScenarioDomain:
        domain_dir = scenarios_dir / domain.value
        if domain_dir.exists():
            for f in domain_dir.glob("*.json"):
                with open(f) as file:
                    scenario = json.load(file)
                    if scenario['scenario_id'] in data:
                        data[scenario['scenario_id']]['scenario'] = scenario

    # Load baseline results
    for sid in data:
        # LLM-Direct
        ld_file = results_dir / "llm_direct" / f"{sid}_llm_direct.json"
        if ld_file.exists():
            with open(ld_file) as f:
                data[sid]['llm_direct'] = json.load(f)

        # CoT
        cot_file = results_dir / "cot" / f"{sid}_cot.json"
        if cot_file.exists():
            with open(cot_file) as f:
                data[sid]['cot'] = json.load(f)

        # ReAct
        react_file = results_dir / "react" / f"{sid}_react.json"
        if react_file.exists():
            with open(react_file) as f:
                data[sid]['react'] = json.load(f)

    return data


def analyze_agenttrace_result(trace_data):
    """Analyze AgentTrace result for a scenario."""
    gt_root = trace_data.get('root_cause_node_id')
    error_node = trace_data.get('error_node_id')
    causal_path = trace_data.get('causal_path', [])

    graph = CausalGraph.from_json(trace_data['trace_json'])

    try:
        backward_nodes = graph.trace_backward(error_node)
        backward_ids = [n.id for n in backward_nodes]

        rank = backward_ids.index(gt_root) + 1 if gt_root in backward_ids else -1

        return {
            'found': gt_root in backward_ids,
            'rank': rank,
            'path_length': len(causal_path),
            'trace_length': len(backward_ids),
            'backward_nodes': backward_nodes[:5]  # First 5 for display
        }
    except Exception as e:
        return {'found': False, 'error': str(e)}


def find_best_cases(all_data, n=3):
    """Find best AgentTrace cases for showcase."""
    cases = []

    for sid, data in all_data.items():
        if 'trace' not in data or 'scenario' not in data:
            continue

        trace = data['trace']
        scenario = data['scenario']

        result = analyze_agenttrace_result(trace)
        if not result.get('found'):
            continue

        # Score: lower rank + diverse domains + diverse bug types
        score = -result['rank']  # Lower rank is better

        cases.append({
            'scenario_id': sid,
            'domain': scenario.get('domain'),
            'bug_type': scenario.get('bug_type'),
            'rank': result['rank'],
            'path_length': result['path_length'],
            'score': score,
            'data': data
        })

    # Sort by rank (best first)
    cases.sort(key=lambda x: x['rank'])

    # Select diverse cases
    selected = []
    domains_seen = set()
    bug_types_seen = set()

    for case in cases:
        if case['domain'] not in domains_seen or case['bug_type'] not in bug_types_seen:
            selected.append(case)
            domains_seen.add(case['domain'])
            bug_types_seen.add(case['bug_type'])

        if len(selected) >= n:
            break

    return selected


def find_failure_cases(all_data, n=3):
    """Find cases where AgentTrace failed or performed poorly."""
    cases = []

    for sid, data in all_data.items():
        if 'trace' not in data or 'scenario' not in data:
            continue

        trace = data['trace']
        scenario = data['scenario']

        result = analyze_agenttrace_result(trace)

        # Cases with rank > 3 (not in Hit@3)
        if result.get('rank', -1) > 3 or not result.get('found'):
            cases.append({
                'scenario_id': sid,
                'domain': scenario.get('domain'),
                'bug_type': scenario.get('bug_type'),
                'rank': result.get('rank', -1),
                'path_length': result.get('path_length', 0),
                'data': data
            })

    # Sort by rank (worst first)
    cases.sort(key=lambda x: -x.get('rank', 0))

    return cases[:n]


def generate_case_study_latex(case, case_type="success"):
    """Generate LaTeX for a single case study."""
    sid = case['scenario_id']
    data = case['data']
    scenario = data['scenario']
    trace = data['trace']

    latex = f"""
\\subsection{{{case_type.title()} Case: {scenario.get('domain', '').title()} Domain}}
\\label{{case:{sid}}}

\\textbf{{Scenario:}} {scenario.get('task_description', 'N/A')[:200]}...

\\textbf{{Bug Type:}} {scenario.get('bug_type', 'N/A').replace('_', ' ').title()}

\\textbf{{Ground Truth:}} Error at step {scenario.get('error_manifestation_step')},
root cause at step {scenario.get('root_cause_step')}.

\\begin{{table}}[h]
\\centering
\\caption{{Method Comparison for Case {sid}}}
\\begin{{tabular}}{{lcc}}
\\toprule
\\textbf{{Method}} & \\textbf{{Identified Root Cause}} & \\textbf{{Correct?}} \\\\
\\midrule
"""

    # AgentTrace result
    at_result = analyze_agenttrace_result(trace)
    at_correct = at_result.get('rank', -1) <= 3
    latex += f"AgentTrace & Rank {at_result.get('rank', 'N/A')} in backward trace & {'\\checkmark' if at_correct else '$\\times$'} \\\\\n"

    # LLM-Direct
    if 'llm_direct' in data:
        ld = data['llm_direct']
        ld_correct = f"step {scenario.get('root_cause_step')}" in ld.get('identified_root_cause', '').lower()
        latex += f"LLM-Direct & {ld.get('identified_root_cause', 'N/A')[:50]}... & {'\\checkmark' if ld_correct else '$\\times$'} \\\\\n"

    # CoT
    if 'cot' in data:
        cot = data['cot']
        cot_correct = f"step {scenario.get('root_cause_step')}" in str(cot.get('reasoning_steps', [])).lower()
        latex += f"CoT & {cot.get('identified_root_cause', 'N/A')[:50]}... & {'\\checkmark' if cot_correct else '$\\times$'} \\\\\n"

    latex += """
\\bottomrule
\\end{tabular}
\\end{table}

"""

    # Causal path visualization
    causal_path = trace.get('causal_path', [])
    if causal_path:
        graph = CausalGraph.from_json(trace['trace_json'])
        latex += "\\textbf{Causal Path (AgentTrace):}\n\\begin{enumerate}\n"

        for node_id in causal_path[:5]:
            node = graph.get_node(node_id)
            if node:
                action = node.data.get('action', 'unknown') if isinstance(node.data, dict) else 'unknown'
                is_bug = node.data.get('is_bug_point', False) if isinstance(node.data, dict) else False
                latex += f"  \\item [{node.agent_id}] {action}"
                if is_bug:
                    latex += " \\textbf{(BUG)}"
                latex += "\n"

        latex += "\\end{enumerate}\n"

    return latex


def generate_all_case_studies(all_data):
    """Generate complete case study section."""
    print("Finding best cases...")
    best_cases = find_best_cases(all_data, n=3)

    print("Finding failure cases...")
    failure_cases = find_failure_cases(all_data, n=2)

    latex = """
\\section{Case Studies}
\\label{sec:case_studies}

This section presents detailed case studies illustrating AgentTrace's capabilities and limitations.

\\subsection{Successful Cases}

"""

    for i, case in enumerate(best_cases):
        print(f"  Generating case study for {case['scenario_id']}...")
        latex += generate_case_study_latex(case, "success")

    latex += """
\\subsection{Failure Analysis}

The following cases illustrate scenarios where AgentTrace's performance was suboptimal:

"""

    for case in failure_cases:
        print(f"  Generating failure case for {case['scenario_id']}...")
        latex += generate_case_study_latex(case, "failure")

    return latex


def generate_summary_statistics(all_data):
    """Generate summary statistics for case studies."""
    stats = {
        'total': 0,
        'hit_at_1': 0,
        'hit_at_3': 0,
        'by_domain': {},
        'by_bug_type': {},
        'by_distance': {}
    }

    for sid, data in all_data.items():
        if 'trace' not in data or 'scenario' not in data:
            continue

        trace = data['trace']
        scenario = data['scenario']

        stats['total'] += 1
        result = analyze_agenttrace_result(trace)

        domain = scenario.get('domain', 'unknown')
        bug_type = scenario.get('bug_type', 'unknown')
        distance = len(trace.get('causal_path', [])) - 1

        if domain not in stats['by_domain']:
            stats['by_domain'][domain] = {'total': 0, 'hit_at_3': 0}
        stats['by_domain'][domain]['total'] += 1

        if bug_type not in stats['by_bug_type']:
            stats['by_bug_type'][bug_type] = {'total': 0, 'hit_at_3': 0}
        stats['by_bug_type'][bug_type]['total'] += 1

        if distance not in stats['by_distance']:
            stats['by_distance'][distance] = {'total': 0, 'hit_at_3': 0}
        stats['by_distance'][distance]['total'] += 1

        if result.get('rank') == 1:
            stats['hit_at_1'] += 1
        if result.get('rank', 99) <= 3:
            stats['hit_at_3'] += 1
            stats['by_domain'][domain]['hit_at_3'] += 1
            stats['by_bug_type'][bug_type]['hit_at_3'] += 1
            stats['by_distance'][distance]['hit_at_3'] += 1

    return stats


def main():
    """Generate case study analysis."""
    print("="*70)
    print("CASE STUDY ANALYSIS")
    print("="*70)

    print("\nLoading all data...")
    all_data = load_all_data()
    print(f"Loaded {len(all_data)} scenarios with complete data")

    # Generate summary statistics
    print("\nGenerating summary statistics...")
    stats = generate_summary_statistics(all_data)

    print(f"\nOverall Performance:")
    print(f"  Total: {stats['total']}")
    print(f"  Hit@1: {stats['hit_at_1']} ({stats['hit_at_1']/stats['total']:.1%})")
    print(f"  Hit@3: {stats['hit_at_3']} ({stats['hit_at_3']/stats['total']:.1%})")

    print(f"\nBy Causal Distance:")
    for dist in sorted(stats['by_distance'].keys()):
        d = stats['by_distance'][dist]
        rate = d['hit_at_3'] / d['total'] if d['total'] > 0 else 0
        print(f"  Distance {dist}: {rate:.1%} (n={d['total']})")

    # Generate LaTeX case studies
    print("\nGenerating case studies...")
    latex = generate_all_case_studies(all_data)

    output_dir = Path("data/results")
    with open(output_dir / "case_studies.tex", 'w') as f:
        f.write(latex)
    print(f"\nCase studies saved to: {output_dir / 'case_studies.tex'}")

    # Save statistics
    with open(output_dir / "case_study_stats.json", 'w') as f:
        json.dump(stats, f, indent=2)

    print("\n" + "="*70)
    print("CASE STUDY ANALYSIS COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
