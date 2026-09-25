"""Configuration for AgentTrace experiments."""

from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
DATA_DIR = PROJECT_ROOT / "data"

SCENARIOS_DIR = DATA_DIR / "scenarios"
TRACES_DIR = DATA_DIR / "traces"
LABELS_DIR = DATA_DIR / "labels"
RESULTS_DIR = DATA_DIR / "results"

# Model configurations - 选择最适合的模型
# GPT-4o: 高质量、快速、支持JSON mode，适合大多数任务
# GPT-4o-mini: 便宜快速，适合大量调用
# GPT-5-mini: 最新模型，性能更好但成本更高

OPENAI_MODELS = {
    # 场景生成：需要创造力和准确的JSON输出
    "scenario_generation": "gpt-4o",

    # Ground Truth标注：需要精确的因果分析能力
    "ground_truth_labeling": "gpt-4o",

    # LLM-as-Judge：需要公正客观的评估
    "llm_judge": "gpt-4o",

    # LLM-Direct baseline：与AgentTrace公平对比
    "llm_direct_baseline": "gpt-4o",

    # Agent模拟：大量调用，用便宜模型
    "agent_simulation": "gpt-4o-mini",
}

ANTHROPIC_MODELS = {
    "cross_validation": "claude-3-opus-20240229",
}

# Benchmark configuration
BENCHMARK_CONFIG = {
    "scenarios_per_domain": 50,
    "total_scenarios": 250,
    "domains": ["research", "coding", "planning", "customer_service", "trading"],
    "complexity_distribution": {
        "simple": 0.3,
        "medium": 0.5,
        "complex": 0.2,
    },
}

# Evaluation configuration
EVALUATION_CONFIG = {
    "metrics": [
        "precision",
        "recall",
        "f1",
        "chain_accuracy",
        "root_cause_hit_at_1",
        "root_cause_hit_at_3",
        "mrr",  # Mean Reciprocal Rank
    ],
    "llm_judge_criteria": [
        "correctness",
        "completeness",
        "clarity",
        "efficiency",
    ],
}

# API rate limiting
API_CONFIG = {
    "max_concurrent_requests": 5,
    "retry_attempts": 3,
    "retry_delay_seconds": 2,
}
