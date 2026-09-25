"""
Generate publication-quality figures for AgentTrace paper.
"""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

# Use non-interactive backend
matplotlib.use('Agg')

# Set publication style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'serif',
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'figure.figsize': (8, 6),
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

# Color palette (colorblind-friendly)
COLORS = {
    'agenttrace': '#2ecc71',  # Green
    'llm_direct': '#3498db',  # Blue
    'log_only': '#e74c3c',    # Red
    'accent': '#9b59b6',      # Purple
}


def load_results():
    """Load all experimental results."""
    results_dir = Path("data/results")

    with open(results_dir / "full_evaluation_results.json") as f:
        full_results = json.load(f)

    with open(results_dir / "ablation_results.json") as f:
        ablation_results = json.load(f)

    return full_results, ablation_results


def fig1_main_comparison(full_results, output_dir):
    """Figure 1: Main comparison of methods."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: Bar chart of Hit@K metrics
    ax1 = axes[0]
    metrics = ['Hit@1', 'Hit@3', 'Hit@5']
    agenttrace_vals = [
        full_results['agenttrace']['hit_at_1_rate'] * 100,
        full_results['agenttrace']['hit_at_3_rate'] * 100,
        full_results['agenttrace']['hit_at_5_rate'] * 100,
    ]
    llm_direct_vals = [
        full_results['llm_direct']['accuracy'] * 100,
        0,  # No Hit@3 for LLM-Direct
        0,
    ]

    x = np.arange(len(metrics))
    width = 0.35

    bars1 = ax1.bar(x - width/2, agenttrace_vals, width, label='AgentTrace',
                     color=COLORS['agenttrace'], edgecolor='black', linewidth=1)
    bars2 = ax1.bar(x + width/2, llm_direct_vals, width, label='LLM-Direct (GPT-4o)',
                     color=COLORS['llm_direct'], edgecolor='black', linewidth=1)

    ax1.set_ylabel('Accuracy (%)')
    ax1.set_xlabel('Metric')
    ax1.set_title('(a) Root Cause Localization Performance')
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics)
    ax1.legend()
    ax1.set_ylim(0, 110)

    # Add value labels
    for bar, val in zip(bars1, agenttrace_vals):
        if val > 0:
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                    f'{val:.1f}%', ha='center', va='bottom', fontsize=10)
    for bar, val in zip(bars2, llm_direct_vals):
        if val > 0:
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                    f'{val:.1f}%', ha='center', va='bottom', fontsize=10)

    # Right: LLM-as-Judge scores
    ax2 = axes[1]
    criteria = ['Correctness', 'Completeness', 'Clarity', 'Efficiency']
    judge = full_results['llm_judge']
    agenttrace_scores = [
        judge['avg_scores_agenttrace']['correctness'],
        judge['avg_scores_agenttrace']['completeness'],
        judge['avg_scores_agenttrace']['clarity'],
        judge['avg_scores_agenttrace']['efficiency'],
    ]
    llm_direct_scores = [
        judge['avg_scores_llm_direct']['correctness'],
        judge['avg_scores_llm_direct']['completeness'],
        judge['avg_scores_llm_direct']['clarity'],
        judge['avg_scores_llm_direct']['efficiency'],
    ]

    x = np.arange(len(criteria))
    bars1 = ax2.bar(x - width/2, agenttrace_scores, width, label='AgentTrace',
                     color=COLORS['agenttrace'], edgecolor='black', linewidth=1)
    bars2 = ax2.bar(x + width/2, llm_direct_scores, width, label='LLM-Direct',
                     color=COLORS['llm_direct'], edgecolor='black', linewidth=1)

    ax2.set_ylabel('Score (1-5)')
    ax2.set_xlabel('Evaluation Criterion')
    ax2.set_title('(b) LLM-as-Judge Evaluation Scores')
    ax2.set_xticks(x)
    ax2.set_xticklabels(criteria, rotation=15, ha='right')
    ax2.legend()
    ax2.set_ylim(0, 5.5)
    ax2.axhline(y=5, color='gray', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_dir / 'fig1_main_comparison.pdf')
    plt.savefig(output_dir / 'fig1_main_comparison.png')
    plt.close()
    print("Generated: fig1_main_comparison.pdf")


def fig2_ablation_domain(ablation_results, output_dir):
    """Figure 2: Performance by domain and bug type."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: By Domain
    ax1 = axes[0]
    domains = list(ablation_results['by_domain'].keys())
    hit3_rates = [ablation_results['by_domain'][d]['hit_at_3_rate'] * 100 for d in domains]
    counts = [ablation_results['by_domain'][d]['total'] for d in domains]

    # Sort by Hit@3
    sorted_idx = np.argsort(hit3_rates)[::-1]
    domains = [domains[i] for i in sorted_idx]
    hit3_rates = [hit3_rates[i] for i in sorted_idx]
    counts = [counts[i] for i in sorted_idx]

    # Clean domain names
    domain_labels = [d.replace('_', '\n').title() for d in domains]

    colors = plt.cm.Greens(np.linspace(0.4, 0.8, len(domains)))
    bars = ax1.bar(domain_labels, hit3_rates, color=colors, edgecolor='black', linewidth=1)

    ax1.set_ylabel('Hit@3 (%)')
    ax1.set_xlabel('Domain')
    ax1.set_title('(a) Performance by Domain')
    ax1.set_ylim(0, 100)

    # Add count labels
    for bar, count in zip(bars, counts):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'n={count}', ha='center', va='bottom', fontsize=9)

    # Right: By Bug Type
    ax2 = axes[1]
    bug_types = list(ablation_results['by_bug_type'].keys())
    hit3_rates = [ablation_results['by_bug_type'][b]['hit_at_3_rate'] * 100 for b in bug_types]
    counts = [ablation_results['by_bug_type'][b]['total'] for b in bug_types]

    # Sort by Hit@3
    sorted_idx = np.argsort(hit3_rates)[::-1]
    bug_types = [bug_types[i] for i in sorted_idx]
    hit3_rates = [hit3_rates[i] for i in sorted_idx]
    counts = [counts[i] for i in sorted_idx]

    # Clean bug type names
    bug_labels = [b.replace('_', '\n').title() for b in bug_types]

    colors = plt.cm.Blues(np.linspace(0.4, 0.8, len(bug_types)))
    bars = ax2.barh(bug_labels, hit3_rates, color=colors, edgecolor='black', linewidth=1)

    ax2.set_xlabel('Hit@3 (%)')
    ax2.set_ylabel('Bug Type')
    ax2.set_title('(b) Performance by Bug Type')
    ax2.set_xlim(0, 100)
    ax2.invert_yaxis()  # Highest at top

    # Add count labels
    for bar, count in zip(bars, counts):
        ax2.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                f'n={count}', ha='left', va='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_dir / 'fig2_ablation_domain_bugtype.pdf')
    plt.savefig(output_dir / 'fig2_ablation_domain_bugtype.png')
    plt.close()
    print("Generated: fig2_ablation_domain_bugtype.pdf")


def fig3_causal_distance(ablation_results, output_dir):
    """Figure 3: Performance vs causal distance."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: Causal Distance
    ax1 = axes[0]
    causal_data = ablation_results['causal_distance']
    distances = sorted([int(d) for d in causal_data.keys()])
    hit3_rates = []
    counts = []

    for d in distances:
        data = causal_data[str(d)]
        rate = data['correct'] / data['total'] * 100 if data['total'] > 0 else 0
        hit3_rates.append(rate)
        counts.append(data['total'])

    ax1.plot(distances, hit3_rates, 'o-', color=COLORS['agenttrace'],
             linewidth=2, markersize=10, markeredgecolor='black')

    # Add count annotations
    for d, rate, count in zip(distances, hit3_rates, counts):
        ax1.annotate(f'n={count}', (d, rate), textcoords="offset points",
                    xytext=(0, 10), ha='center', fontsize=9)

    ax1.set_xlabel('Causal Distance (steps)')
    ax1.set_ylabel('Hit@3 (%)')
    ax1.set_title('(a) Performance vs. Causal Distance')
    ax1.set_ylim(0, 110)
    ax1.set_xticks(distances)

    # Add trend line
    z = np.polyfit(distances, hit3_rates, 1)
    p = np.poly1d(z)
    ax1.plot(distances, p(distances), '--', color='gray', alpha=0.7, label='Trend')
    ax1.legend()

    # Right: Trace Length
    ax2 = axes[1]
    length_data = ablation_results['by_length']
    lengths = ['short (1-5)', 'medium (6-10)', 'long (11+)']
    hit3_rates = [length_data[l]['hit_at_3_rate'] * 100 for l in lengths]
    counts = [length_data[l]['total'] for l in lengths]

    colors = [COLORS['agenttrace'], '#27ae60', '#1e8449']
    bars = ax2.bar(lengths, hit3_rates, color=colors, edgecolor='black', linewidth=1)

    ax2.set_xlabel('Trace Length (nodes)')
    ax2.set_ylabel('Hit@3 (%)')
    ax2.set_title('(b) Performance vs. Trace Length')
    ax2.set_ylim(0, 100)

    # Add count labels
    for bar, count, rate in zip(bars, counts, hit3_rates):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{rate:.1f}%\n(n={count})', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_dir / 'fig3_causal_distance_length.pdf')
    plt.savefig(output_dir / 'fig3_causal_distance_length.png')
    plt.close()
    print("Generated: fig3_causal_distance_length.pdf")


def fig4_llm_judge_breakdown(full_results, output_dir):
    """Figure 4: LLM-as-Judge detailed breakdown."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    judge = full_results['llm_judge']

    # Left: Win Rate Pie Chart
    ax1 = axes[0]
    wins = [judge['agenttrace_wins'], judge['llm_direct_wins'], judge['ties']]
    labels = ['AgentTrace', 'LLM-Direct', 'Tie']
    colors = [COLORS['agenttrace'], COLORS['llm_direct'], 'gray']

    # Filter out zero values
    non_zero = [(w, l, c) for w, l, c in zip(wins, labels, colors) if w > 0]
    wins, labels, colors = zip(*non_zero) if non_zero else ([], [], [])

    wedges, texts, autotexts = ax1.pie(wins, labels=labels, colors=colors,
                                        autopct='%1.1f%%', startangle=90,
                                        explode=[0.05] * len(wins))
    ax1.set_title('(a) LLM-as-Judge Win Rates (n=50)')

    # Right: Radar Chart for criteria scores
    ax2 = axes[1]
    categories = ['Correctness', 'Completeness', 'Clarity', 'Efficiency']

    agenttrace_scores = [
        judge['avg_scores_agenttrace']['correctness'],
        judge['avg_scores_agenttrace']['completeness'],
        judge['avg_scores_agenttrace']['clarity'],
        judge['avg_scores_agenttrace']['efficiency'],
    ]
    llm_direct_scores = [
        judge['avg_scores_llm_direct']['correctness'],
        judge['avg_scores_llm_direct']['completeness'],
        judge['avg_scores_llm_direct']['clarity'],
        judge['avg_scores_llm_direct']['efficiency'],
    ]

    # Create radar chart
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]  # Complete the loop

    agenttrace_scores += agenttrace_scores[:1]
    llm_direct_scores += llm_direct_scores[:1]

    ax2 = fig.add_subplot(122, polar=True)
    ax2.plot(angles, agenttrace_scores, 'o-', linewidth=2,
             label='AgentTrace', color=COLORS['agenttrace'])
    ax2.fill(angles, agenttrace_scores, alpha=0.25, color=COLORS['agenttrace'])

    ax2.plot(angles, llm_direct_scores, 'o-', linewidth=2,
             label='LLM-Direct', color=COLORS['llm_direct'])
    ax2.fill(angles, llm_direct_scores, alpha=0.25, color=COLORS['llm_direct'])

    ax2.set_xticks(angles[:-1])
    ax2.set_xticklabels(categories)
    ax2.set_ylim(0, 5)
    ax2.set_title('(b) Evaluation Criteria Comparison')
    ax2.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))

    plt.tight_layout()
    plt.savefig(output_dir / 'fig4_llm_judge_breakdown.pdf')
    plt.savefig(output_dir / 'fig4_llm_judge_breakdown.png')
    plt.close()
    print("Generated: fig4_llm_judge_breakdown.pdf")


def fig5_summary(full_results, ablation_results, output_dir):
    """Figure 5: Summary comparison figure for paper."""
    fig = plt.figure(figsize=(10, 8))

    # Main comparison: AgentTrace vs LLM-Direct
    ax = fig.add_subplot(111)

    methods = ['AgentTrace', 'LLM-Direct\n(GPT-4o)']

    # Key metrics
    recall = [100, 0]  # LLM-Direct doesn't have recall
    hit3 = [full_results['agenttrace']['hit_at_3_rate'] * 100,
            full_results['llm_direct']['accuracy'] * 100]
    correctness = [full_results['llm_judge']['avg_scores_agenttrace']['correctness'] * 20,
                   full_results['llm_judge']['avg_scores_llm_direct']['correctness'] * 20]

    x = np.arange(len(methods))
    width = 0.25

    bars1 = ax.bar(x - width, recall, width, label='Recall', color=COLORS['agenttrace'],
                   edgecolor='black', linewidth=1)
    bars2 = ax.bar(x, hit3, width, label='Hit@3 / Accuracy', color=COLORS['llm_direct'],
                   edgecolor='black', linewidth=1)
    bars3 = ax.bar(x + width, correctness, width, label='Correctness (×20)', color=COLORS['accent'],
                   edgecolor='black', linewidth=1)

    ax.set_ylabel('Score (%)')
    ax.set_title('AgentTrace vs. LLM-Direct: Key Metrics Comparison\n(250 scenarios, 5 domains, 8 bug types)')
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.legend(loc='upper right')
    ax.set_ylim(0, 110)

    # Add value labels
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2, height + 2,
                       f'{height:.0f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

    # Add annotation
    ax.annotate('AgentTrace achieves\n2x better Hit@3',
                xy=(0, hit3[0]), xytext=(0.5, 60),
                arrowprops=dict(arrowstyle='->', color='black'),
                fontsize=11, ha='center')

    plt.tight_layout()
    plt.savefig(output_dir / 'fig5_summary.pdf')
    plt.savefig(output_dir / 'fig5_summary.png')
    plt.close()
    print("Generated: fig5_summary.pdf")


def main():
    """Generate all figures."""
    print("="*60)
    print("Generating Publication Figures")
    print("="*60)

    # Create output directory
    output_dir = Path("data/results/figures")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load results
    full_results, ablation_results = load_results()

    # Generate figures
    fig1_main_comparison(full_results, output_dir)
    fig2_ablation_domain(ablation_results, output_dir)
    fig3_causal_distance(ablation_results, output_dir)
    fig4_llm_judge_breakdown(full_results, output_dir)
    fig5_summary(full_results, ablation_results, output_dir)

    print("\n" + "="*60)
    print(f"All figures saved to: {output_dir}")
    print("="*60)


if __name__ == "__main__":
    main()
