"""
Statistical Significance Tests for AgentTrace.

Performs:
1. McNemar's test for pairwise method comparison
2. Bootstrap confidence intervals for Hit@K
3. Effect size calculation (Cohen's h)
4. Stratified analysis by domain (original vs new)
"""

import json
import numpy as np
from pathlib import Path
from scipy import stats
from typing import Dict, List, Tuple
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def load_evaluation_data() -> Tuple[Dict, Dict]:
    """Load all evaluation results and ground truth."""
    blind_dir = Path("data/blind_benchmark")

    with open(blind_dir / "ground_truth" / "all_ground_truth.json") as f:
        ground_truth = json.load(f)

    # Load individual trace results
    traces_dir = blind_dir / "traces"
    traces = {}
    for trace_file in traces_dir.glob("*_trace.json"):
        scenario_id = trace_file.stem.replace("_trace", "")
        with open(trace_file) as f:
            traces[scenario_id] = json.load(f)

    return traces, ground_truth


def run_ranking_evaluation(traces: Dict, ground_truth: Dict) -> Dict[str, List[bool]]:
    """Run all methods and record per-scenario correctness."""
    from agenttrace.ranking.ranker import NodeRanker
    from agenttrace.core.graph import CausalGraph

    ranker = NodeRanker()

    # Results: method -> list of (correct_at_1, correct_at_3)
    results = {
        'AgentTrace-Ranked': [],
        'AgentTrace-Original': [],
        'Betweenness': [],
        'AnomalyScore': [],
        'RandomWalk': [],
    }

    scenario_domains = {}

    for scenario_id, trace in traces.items():
        if scenario_id not in ground_truth:
            continue

        gt = ground_truth[scenario_id]
        root_cause_id = gt.get('root_cause_node_id')
        error_node_id = gt.get('error_node_id')
        domain = gt.get('domain', 'unknown')
        scenario_domains[scenario_id] = domain

        if not root_cause_id or not error_node_id:
            continue

        # Parse trace
        trace_json = trace.get('trace_json', '{}')
        graph = CausalGraph.from_json(trace_json)

        # Use error node from ground truth
        error_node = graph.get_node(error_node_id)
        if not error_node:
            continue

        backward_nodes = graph.trace_backward(error_node_id)

        if not backward_nodes:
            continue

        # AgentTrace-Ranked
        ranking_result = ranker.rank_candidates(graph, error_node_id)
        if ranking_result.ranked_candidates:
            top1_correct = ranking_result.ranked_candidates[0].node.id == root_cause_id
            top3_ids = [r.node.id for r in ranking_result.ranked_candidates[:3]]
            top3_correct = root_cause_id in top3_ids
            results['AgentTrace-Ranked'].append((top1_correct, top3_correct, scenario_id))

        # AgentTrace-Original (causal distance ordering)
        original_order = sorted(backward_nodes, key=lambda n: -len(graph.trace_backward(n.id)))
        if original_order:
            top1_correct = original_order[0].id == root_cause_id
            top3_ids = [n.id for n in original_order[:3]]
            top3_correct = root_cause_id in top3_ids
            results['AgentTrace-Original'].append((top1_correct, top3_correct, scenario_id))

        # Betweenness centrality baseline
        import networkx as nx
        betweenness = nx.betweenness_centrality(graph._graph)
        backward_ids = [n.id for n in backward_nodes]
        betweenness_order = sorted(backward_ids, key=lambda x: -betweenness.get(x, 0))
        if betweenness_order:
            top1_correct = betweenness_order[0] == root_cause_id
            top3_correct = root_cause_id in betweenness_order[:3]
            results['Betweenness'].append((top1_correct, top3_correct, scenario_id))

        # AnomalyScore baseline (degree-based)
        anomaly_scores = {}
        for n in backward_nodes:
            in_deg = graph._graph.in_degree(n.id)
            out_deg = graph._graph.out_degree(n.id)
            anomaly_scores[n.id] = abs(out_deg - in_deg) + 0.1 * out_deg
        anomaly_order = sorted(backward_ids, key=lambda x: -anomaly_scores.get(x, 0))
        if anomaly_order:
            top1_correct = anomaly_order[0] == root_cause_id
            top3_correct = root_cause_id in anomaly_order[:3]
            results['AnomalyScore'].append((top1_correct, top3_correct, scenario_id))

        # RandomWalk baseline
        if backward_ids:
            random_order = backward_ids.copy()
            np.random.seed(42)
            np.random.shuffle(random_order)
            top1_correct = random_order[0] == root_cause_id
            top3_correct = root_cause_id in random_order[:3]
            results['RandomWalk'].append((top1_correct, top3_correct, scenario_id))

    return results, scenario_domains


def mcnemar_test(method1_results: List, method2_results: List) -> Dict:
    """
    Perform McNemar's test for paired binary outcomes.

    Returns test statistic, p-value, and contingency table.
    """
    # Align by scenario_id
    method1_dict = {r[2]: r[0] for r in method1_results}
    method2_dict = {r[2]: r[0] for r in method2_results}

    common_scenarios = set(method1_dict.keys()) & set(method2_dict.keys())

    # Build contingency table
    # b = method1 correct, method2 wrong
    # c = method1 wrong, method2 correct
    b = sum(1 for s in common_scenarios if method1_dict[s] and not method2_dict[s])
    c = sum(1 for s in common_scenarios if not method1_dict[s] and method2_dict[s])
    a = sum(1 for s in common_scenarios if method1_dict[s] and method2_dict[s])
    d = sum(1 for s in common_scenarios if not method1_dict[s] and not method2_dict[s])

    # McNemar's test (with continuity correction)
    if b + c > 0:
        chi2 = (abs(b - c) - 1) ** 2 / (b + c)
        p_value = 1 - stats.chi2.cdf(chi2, df=1)
    else:
        chi2 = 0
        p_value = 1.0

    return {
        'contingency_table': {'a': a, 'b': b, 'c': c, 'd': d},
        'chi2': chi2,
        'p_value': p_value,
        'n': len(common_scenarios)
    }


def bootstrap_ci(results: List, metric: str = 'hit1', n_bootstrap: int = 10000,
                 confidence: float = 0.95) -> Dict:
    """
    Compute bootstrap confidence interval for Hit@K.
    """
    if metric == 'hit1':
        values = [1 if r[0] else 0 for r in results]
    else:  # hit3
        values = [1 if r[1] else 0 for r in results]

    values = np.array(values)
    n = len(values)

    # Bootstrap resampling
    np.random.seed(42)
    bootstrap_means = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(values, size=n, replace=True)
        bootstrap_means.append(np.mean(sample))

    bootstrap_means = np.array(bootstrap_means)

    # Percentile method
    alpha = 1 - confidence
    lower = np.percentile(bootstrap_means, alpha/2 * 100)
    upper = np.percentile(bootstrap_means, (1 - alpha/2) * 100)

    return {
        'mean': np.mean(values),
        'std': np.std(bootstrap_means),
        'ci_lower': lower,
        'ci_upper': upper,
        'confidence': confidence,
        'n_samples': n
    }


def cohens_h(p1: float, p2: float) -> float:
    """
    Calculate Cohen's h effect size for proportions.

    Interpretation:
    - 0.2: small
    - 0.5: medium
    - 0.8: large
    """
    phi1 = 2 * np.arcsin(np.sqrt(p1))
    phi2 = 2 * np.arcsin(np.sqrt(p2))
    return abs(phi1 - phi2)


def stratified_analysis(results: Dict, scenario_domains: Dict) -> Dict:
    """
    Analyze performance by domain groups.
    """
    original_domains = {'coding', 'customer_service', 'research', 'planning', 'trading'}
    new_domains = {'healthcare', 'legal', 'education', 'finance', 'devops'}

    analysis = {}

    for method, method_results in results.items():
        original_hits = []
        new_hits = []
        per_domain = {}

        for hit1, hit3, scenario_id in method_results:
            domain = scenario_domains.get(scenario_id, 'unknown')

            if domain not in per_domain:
                per_domain[domain] = {'hit1': 0, 'hit3': 0, 'total': 0}
            per_domain[domain]['total'] += 1
            if hit1:
                per_domain[domain]['hit1'] += 1
            if hit3:
                per_domain[domain]['hit3'] += 1

            if domain in original_domains:
                original_hits.append(hit1)
            elif domain in new_domains:
                new_hits.append(hit1)

        analysis[method] = {
            'original_domains': {
                'hit1': np.mean(original_hits) if original_hits else 0,
                'n': len(original_hits)
            },
            'new_domains': {
                'hit1': np.mean(new_hits) if new_hits else 0,
                'n': len(new_hits)
            },
            'per_domain': {
                d: {
                    'hit1': stats['hit1'] / stats['total'] if stats['total'] > 0 else 0,
                    'hit3': stats['hit3'] / stats['total'] if stats['total'] > 0 else 0,
                    'n': stats['total']
                }
                for d, stats in per_domain.items()
            }
        }

    return analysis


def main():
    print("=" * 70)
    print("STATISTICAL SIGNIFICANCE ANALYSIS")
    print("=" * 70)

    # Load data
    print("\nLoading evaluation data...")
    traces, ground_truth = load_evaluation_data()
    print(f"Loaded {len(traces)} traces, {len(ground_truth)} ground truth entries")

    # Run evaluations
    print("\nRunning method evaluations...")
    results, scenario_domains = run_ranking_evaluation(traces, ground_truth)

    # 1. Bootstrap Confidence Intervals
    print("\n" + "=" * 70)
    print("1. BOOTSTRAP CONFIDENCE INTERVALS (95%)")
    print("=" * 70)

    ci_results = {}
    print(f"\n{'Method':<25} {'Hit@1':>10} {'95% CI':>20} {'Hit@3':>10} {'95% CI':>20}")
    print("-" * 85)

    for method, method_results in results.items():
        if not method_results:
            continue
        ci_hit1 = bootstrap_ci(method_results, 'hit1')
        ci_hit3 = bootstrap_ci(method_results, 'hit3')
        ci_results[method] = {'hit1': ci_hit1, 'hit3': ci_hit3}

        print(f"{method:<25} {ci_hit1['mean']*100:>9.1f}% "
              f"[{ci_hit1['ci_lower']*100:.1f}, {ci_hit1['ci_upper']*100:.1f}] "
              f"{ci_hit3['mean']*100:>9.1f}% "
              f"[{ci_hit3['ci_lower']*100:.1f}, {ci_hit3['ci_upper']*100:.1f}]")

    # 2. McNemar's Tests
    print("\n" + "=" * 70)
    print("2. McNEMAR'S TEST (AgentTrace-Ranked vs Baselines)")
    print("=" * 70)

    mcnemar_results = {}
    baseline_methods = ['Betweenness', 'AnomalyScore', 'AgentTrace-Original', 'RandomWalk']

    print(f"\n{'Comparison':<45} {'Chi2':>10} {'p-value':>15} {'Sig.':>8}")
    print("-" * 80)

    for baseline in baseline_methods:
        if baseline in results and 'AgentTrace-Ranked' in results:
            test = mcnemar_test(results['AgentTrace-Ranked'], results[baseline])
            mcnemar_results[baseline] = test

            sig = "***" if test['p_value'] < 0.001 else "**" if test['p_value'] < 0.01 else "*" if test['p_value'] < 0.05 else "n.s."
            print(f"AgentTrace-Ranked vs {baseline:<25} {test['chi2']:>10.2f} {test['p_value']:>15.2e} {sig:>8}")

    # 3. Effect Sizes
    print("\n" + "=" * 70)
    print("3. EFFECT SIZE (Cohen's h)")
    print("=" * 70)

    effect_sizes = {}
    agenttrace_hit1 = ci_results['AgentTrace-Ranked']['hit1']['mean']

    print(f"\n{'Comparison':<45} {'Cohen h':>10} {'Interpretation':>15}")
    print("-" * 70)

    for baseline in baseline_methods:
        if baseline in ci_results:
            baseline_hit1 = ci_results[baseline]['hit1']['mean']
            h = cohens_h(agenttrace_hit1, baseline_hit1)
            effect_sizes[baseline] = h

            interp = "large" if h >= 0.8 else "medium" if h >= 0.5 else "small"
            print(f"AgentTrace-Ranked vs {baseline:<25} {h:>10.3f} {interp:>15}")

    # 4. Stratified Analysis
    print("\n" + "=" * 70)
    print("4. STRATIFIED ANALYSIS BY DOMAIN GROUP")
    print("=" * 70)

    stratified = stratified_analysis(results, scenario_domains)

    print("\n--- AgentTrace-Ranked Performance ---")
    agenttrace_strat = stratified['AgentTrace-Ranked']
    print(f"Original 5 domains: {agenttrace_strat['original_domains']['hit1']*100:.1f}% Hit@1 (n={agenttrace_strat['original_domains']['n']})")
    print(f"New 5 domains:      {agenttrace_strat['new_domains']['hit1']*100:.1f}% Hit@1 (n={agenttrace_strat['new_domains']['n']})")

    print("\n--- Per-Domain Breakdown ---")
    print(f"{'Domain':<20} {'Hit@1':>10} {'Hit@3':>10} {'N':>8}")
    print("-" * 50)

    for domain, stats in sorted(agenttrace_strat['per_domain'].items()):
        print(f"{domain:<20} {stats['hit1']*100:>9.1f}% {stats['hit3']*100:>9.1f}% {stats['n']:>8}")

    # 5. Two-proportion z-test for domain groups
    print("\n" + "=" * 70)
    print("5. DOMAIN GROUP COMPARISON (Z-test)")
    print("=" * 70)

    orig = agenttrace_strat['original_domains']
    new = agenttrace_strat['new_domains']

    if orig['n'] > 0 and new['n'] > 0:
        # Pooled proportion
        p_pooled = (orig['hit1'] * orig['n'] + new['hit1'] * new['n']) / (orig['n'] + new['n'])
        se = np.sqrt(p_pooled * (1 - p_pooled) * (1/orig['n'] + 1/new['n']))

        if se > 0:
            z = (orig['hit1'] - new['hit1']) / se
            from scipy.stats import norm
            p_value = 2 * (1 - norm.cdf(abs(z)))

            print(f"\nOriginal domains Hit@1: {orig['hit1']*100:.1f}%")
            print(f"New domains Hit@1:      {new['hit1']*100:.1f}%")
            print(f"Difference:             {(orig['hit1'] - new['hit1'])*100:.1f}%")
            print(f"Z-statistic:            {z:.3f}")
            print(f"p-value:                {p_value:.2e}")
            print(f"Significant (p<0.05):   {'Yes' if p_value < 0.05 else 'No'}")

    # Save results
    output = {
        'confidence_intervals': {
            method: {
                'hit1': {k: float(v) if isinstance(v, (np.floating, float)) else v
                        for k, v in ci['hit1'].items()},
                'hit3': {k: float(v) if isinstance(v, (np.floating, float)) else v
                        for k, v in ci['hit3'].items()}
            }
            for method, ci in ci_results.items()
        },
        'mcnemar_tests': {
            k: {kk: float(vv) if isinstance(vv, (np.floating, float)) else vv
                for kk, vv in v.items()}
            for k, v in mcnemar_results.items()
        },
        'effect_sizes': {k: float(v) for k, v in effect_sizes.items()},
        'stratified_analysis': {
            method: {
                'original_domains': {k: float(v) if isinstance(v, (np.floating, float)) else v
                                    for k, v in data['original_domains'].items()},
                'new_domains': {k: float(v) if isinstance(v, (np.floating, float)) else v
                               for k, v in data['new_domains'].items()},
                'per_domain': {
                    domain: {k: float(v) if isinstance(v, (np.floating, float)) else v
                            for k, v in stats.items()}
                    for domain, stats in data['per_domain'].items()
                }
            }
            for method, data in stratified.items()
        }
    }

    output_file = Path("data/results/statistical_tests.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n\nResults saved to: {output_file}")

    # Summary for paper
    print("\n" + "=" * 70)
    print("SUMMARY FOR PAPER")
    print("=" * 70)

    print("""
Table 1: Main Results with Statistical Significance

Method               Hit@1        95% CI          Hit@3      p-value
---------------------------------------------------------------------""")

    for method in ['AgentTrace-Ranked', 'Betweenness', 'AnomalyScore', 'AgentTrace-Original', 'RandomWalk']:
        if method in ci_results:
            ci = ci_results[method]
            p_str = "-" if method == 'AgentTrace-Ranked' else f"{mcnemar_results.get(method, {}).get('p_value', 1.0):.2e}"
            print(f"{method:<20} {ci['hit1']['mean']*100:>6.1f}%   [{ci['hit1']['ci_lower']*100:.1f}, {ci['hit1']['ci_upper']*100:.1f}]   "
                  f"{ci['hit3']['mean']*100:>6.1f}%   {p_str:>12}")

    print("""
Note: p-values from McNemar's test comparing each method to AgentTrace-Ranked.
*** p < 0.001, ** p < 0.01, * p < 0.05
""")


if __name__ == "__main__":
    main()
