"""
Error Analysis for AgentTrace.

Analyzes cases where AgentTrace failed to correctly rank root cause:
1. Why did it fail?
2. What patterns lead to failure?
3. How can we improve?
"""

import json
from pathlib import Path
from collections import defaultdict
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from agenttrace.core.graph import CausalGraph
from experiments.benchmark.scenario_generator import ScenarioDomain


def load_data():
    """Load traces and scenarios."""
    traces_dir = Path("data/traces")
    scenarios_dir = Path("data/scenarios")

    data = {}

    for trace_file in traces_dir.glob("*_trace.json"):
        with open(trace_file) as f:
            trace = json.load(f)
            data[trace['scenario_id']] = {'trace': trace}

    for domain in ScenarioDomain:
        domain_dir = scenarios_dir / domain.value
        if domain_dir.exists():
            for f in domain_dir.glob("*.json"):
                with open(f) as file:
                    scenario = json.load(file)
                    if scenario['scenario_id'] in data:
                        data[scenario['scenario_id']]['scenario'] = scenario

    return data


def analyze_failures(data):
    """Analyze all failure cases."""
    failures = []
    successes = []

    for sid, item in data.items():
        if 'trace' not in item or 'scenario' not in item:
            continue

        trace = item['trace']
        scenario = item['scenario']

        gt_root = trace.get('root_cause_node_id')
        error_node = trace.get('error_node_id')

        if not gt_root or not error_node:
            continue

        graph = CausalGraph.from_json(trace['trace_json'])

        try:
            backward_nodes = graph.trace_backward(error_node)
            backward_ids = [n.id for n in backward_nodes]

            if gt_root in backward_ids:
                rank = backward_ids.index(gt_root) + 1
            else:
                rank = -1
        except Exception as e:
            rank = -1

        result = {
            'scenario_id': sid,
            'domain': scenario.get('domain'),
            'bug_type': scenario.get('bug_type'),
            'rank': rank,
            'causal_distance': len(trace.get('causal_path', [])) - 1,
            'node_count': trace.get('node_count', 0),
            'edge_count': trace.get('edge_count', 0),
        }

        if rank > 3 or rank == -1:
            # Failure: not in top 3
            failures.append(result)
        else:
            successes.append(result)

    return failures, successes


def analyze_failure_patterns(failures, successes):
    """Find patterns in failure cases."""
    print("\n" + "="*60)
    print("FAILURE PATTERN ANALYSIS")
    print("="*60)

    # By domain
    print("\n1. Failures by Domain:")
    domain_stats = defaultdict(lambda: {'failures': 0, 'total': 0})
    for f in failures:
        domain_stats[f['domain']]['failures'] += 1
        domain_stats[f['domain']]['total'] += 1
    for s in successes:
        domain_stats[s['domain']]['total'] += 1

    for domain, stats in sorted(domain_stats.items()):
        rate = stats['failures'] / stats['total'] if stats['total'] > 0 else 0
        print(f"  {domain}: {stats['failures']}/{stats['total']} failures ({rate:.1%})")

    # By bug type
    print("\n2. Failures by Bug Type:")
    bug_stats = defaultdict(lambda: {'failures': 0, 'total': 0})
    for f in failures:
        bug_stats[f['bug_type']]['failures'] += 1
        bug_stats[f['bug_type']]['total'] += 1
    for s in successes:
        bug_stats[s['bug_type']]['total'] += 1

    for bug, stats in sorted(bug_stats.items(), key=lambda x: -x[1]['failures']/max(x[1]['total'], 1)):
        rate = stats['failures'] / stats['total'] if stats['total'] > 0 else 0
        print(f"  {bug}: {stats['failures']}/{stats['total']} failures ({rate:.1%})")

    # By causal distance
    print("\n3. Failures by Causal Distance:")
    dist_stats = defaultdict(lambda: {'failures': 0, 'total': 0})
    for f in failures:
        dist_stats[f['causal_distance']]['failures'] += 1
        dist_stats[f['causal_distance']]['total'] += 1
    for s in successes:
        dist_stats[s['causal_distance']]['total'] += 1

    for dist in sorted(dist_stats.keys()):
        stats = dist_stats[dist]
        rate = stats['failures'] / stats['total'] if stats['total'] > 0 else 0
        print(f"  Distance {dist}: {stats['failures']}/{stats['total']} failures ({rate:.1%})")

    # By node count
    print("\n4. Failures by Graph Size:")
    size_buckets = {'small (1-5)': [], 'medium (6-10)': [], 'large (11+)': []}
    for f in failures:
        if f['node_count'] <= 5:
            size_buckets['small (1-5)'].append(f)
        elif f['node_count'] <= 10:
            size_buckets['medium (6-10)'].append(f)
        else:
            size_buckets['large (11+)'].append(f)

    all_by_size = {'small (1-5)': 0, 'medium (6-10)': 0, 'large (11+)': 0}
    for item in failures + successes:
        if item['node_count'] <= 5:
            all_by_size['small (1-5)'] += 1
        elif item['node_count'] <= 10:
            all_by_size['medium (6-10)'] += 1
        else:
            all_by_size['large (11+)'] += 1

    for size, items in size_buckets.items():
        total = all_by_size[size]
        rate = len(items) / total if total > 0 else 0
        print(f"  {size}: {len(items)}/{total} failures ({rate:.1%})")

    return {
        'by_domain': dict(domain_stats),
        'by_bug_type': dict(bug_stats),
        'by_distance': dict(dist_stats),
        'by_size': {k: len(v) for k, v in size_buckets.items()}
    }


def identify_root_causes_of_failure(failures, data):
    """Identify specific reasons for failures."""
    print("\n" + "="*60)
    print("ROOT CAUSE OF FAILURES")
    print("="*60)

    reasons = defaultdict(list)

    for f in failures[:10]:  # Analyze first 10
        sid = f['scenario_id']
        trace = data[sid]['trace']
        scenario = data[sid]['scenario']

        graph = CausalGraph.from_json(trace['trace_json'])
        gt_root = trace.get('root_cause_node_id')
        error_node = trace.get('error_node_id')

        try:
            backward_nodes = graph.trace_backward(error_node)
            backward_ids = [n.id for n in backward_nodes]

            gt_node = graph.get_node(gt_root)
            first_node = backward_nodes[0] if backward_nodes else None

            # Analyze why rank is high
            if f['causal_distance'] > 2:
                reasons['long_causal_chain'].append(sid)
            elif f['node_count'] > 10:
                reasons['large_graph'].append(sid)
            elif gt_root not in backward_ids:
                reasons['root_cause_not_in_trace'].append(sid)
            else:
                # Root cause is in trace but ranked low
                rank = backward_ids.index(gt_root) + 1

                # Check if there are many intermediate nodes
                if rank > 3:
                    reasons['many_intermediate_nodes'].append(sid)

        except Exception as e:
            reasons['tracing_error'].append(sid)

    print("\nFailure Reasons:")
    for reason, sids in reasons.items():
        print(f"  {reason}: {len(sids)} cases")
        for sid in sids[:2]:
            print(f"    - {sid}")


def generate_improvement_suggestions(patterns):
    """Generate suggestions for improvement."""
    suggestions = []

    # Check distance pattern
    dist_failures = patterns['by_distance']
    high_dist_failures = sum(
        v['failures'] for k, v in dist_failures.items()
        if isinstance(k, int) and k > 2
    )
    if high_dist_failures > 5:
        suggestions.append({
            'issue': 'Poor performance on long causal chains',
            'suggestion': 'Implement learning-to-rank model that weighs nodes by likelihood of being root cause',
            'priority': 'high'
        })

    # Check bug type pattern
    bug_failures = patterns['by_bug_type']
    worst_bugs = sorted(bug_failures.items(),
                       key=lambda x: x[1]['failures']/max(x[1]['total'], 1),
                       reverse=True)[:2]
    for bug, stats in worst_bugs:
        if stats['failures'] > 3:
            suggestions.append({
                'issue': f'High failure rate for {bug} bugs',
                'suggestion': f'Add specialized detection heuristics for {bug} patterns',
                'priority': 'medium'
            })

    # General suggestions
    suggestions.append({
        'issue': 'Low Hit@1 rate',
        'suggestion': 'Train a ranking model using labeled examples to prioritize root cause nodes',
        'priority': 'high'
    })

    suggestions.append({
        'issue': 'Causal distance affects accuracy',
        'suggestion': 'Implement causal strength scoring based on edge types and data flow patterns',
        'priority': 'medium'
    })

    return suggestions


def generate_error_analysis_table(patterns, suggestions):
    """Generate LaTeX table for error analysis."""
    table = r"""
\begin{table}[t]
\centering
\caption{Error Analysis: Failure Patterns}
\label{tab:error_analysis}
\begin{tabular}{lcc}
\toprule
\textbf{Factor} & \textbf{Failure Rate} & \textbf{Correlation} \\
\midrule
"""

    # By causal distance
    dist = patterns['by_distance']
    for d in sorted([k for k in dist.keys() if isinstance(k, int)]):
        stats = dist[d]
        rate = stats['failures'] / stats['total'] if stats['total'] > 0 else 0
        corr = 'Strong' if rate > 0.2 else 'Moderate' if rate > 0.1 else 'Weak'
        table += f"Distance = {d} & {rate:.1%} & {corr} \\\\\n"

    table += r"""
\bottomrule
\end{tabular}
\end{table}

\begin{table}[t]
\centering
\caption{Improvement Suggestions}
\label{tab:improvements}
\begin{tabular}{lp{8cm}c}
\toprule
\textbf{Issue} & \textbf{Suggestion} & \textbf{Priority} \\
\midrule
"""

    for s in suggestions[:4]:
        table += f"{s['issue'][:30]}... & {s['suggestion'][:60]}... & {s['priority'].upper()} \\\\\n"

    table += r"""
\bottomrule
\end{tabular}
\end{table}
"""

    return table


def main():
    """Run error analysis."""
    print("="*70)
    print("ERROR ANALYSIS")
    print("="*70)

    print("\nLoading data...")
    data = load_data()
    print(f"Loaded {len(data)} scenarios")

    print("\nAnalyzing failures...")
    failures, successes = analyze_failures(data)
    print(f"  Failures: {len(failures)}")
    print(f"  Successes: {len(successes)}")
    print(f"  Success Rate: {len(successes)/(len(failures)+len(successes)):.1%}")

    patterns = analyze_failure_patterns(failures, successes)

    identify_root_causes_of_failure(failures, data)

    print("\n" + "="*60)
    print("IMPROVEMENT SUGGESTIONS")
    print("="*60)

    suggestions = generate_improvement_suggestions(patterns)
    for i, s in enumerate(suggestions):
        print(f"\n{i+1}. [{s['priority'].upper()}] {s['issue']}")
        print(f"   Suggestion: {s['suggestion']}")

    # Generate table
    table = generate_error_analysis_table(patterns, suggestions)

    output_dir = Path("data/results")
    with open(output_dir / "error_analysis_table.tex", 'w') as f:
        f.write(table)
    print(f"\nError analysis table saved to: {output_dir / 'error_analysis_table.tex'}")

    # Save results
    with open(output_dir / "error_analysis_results.json", 'w') as f:
        json.dump({
            'failure_count': len(failures),
            'success_count': len(successes),
            'patterns': {k: {str(k2): v2 for k2, v2 in v.items()} for k, v in patterns.items()},
            'suggestions': suggestions
        }, f, indent=2)

    print("\n" + "="*70)
    print("ERROR ANALYSIS COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
