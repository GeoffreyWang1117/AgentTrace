# AgentTrace

**Causal Graph Tracing for Root Cause Analysis in Deployed Multi-Agent Systems**

> This branch contains the code, data, and experimental artifacts for the camera-ready version accepted at the **ICLR 2026 Workshop on Agents in the Wild: Safety, Security, and Beyond**.

## Paper

- **arXiv**: [2603.14688](https://arxiv.org/abs/2603.14688)
- **OpenReview**: Published April 20, 2026

**Citation:**
```bibtex
@inproceedings{wang2026agenttrace,
  title={AgentTrace: Causal Graph Tracing for Root Cause Analysis in Deployed Multi-Agent Systems},
  author={Wang, Zhaohui Geoffrey},
  booktitle={ICLR 2026 Workshop on Agents in the Wild: Safety, Security, and Beyond},
  year={2026}
}
```

## Key Results

| Method | Hit@1 | Hit@3 | MRR |
|--------|-------|-------|-----|
| Random | 9.1% | 27.3% | 0.18 |
| First Node | 3.6% | 10.9% | 0.07 |
| Last Node | 12.7% | 38.2% | 0.25 |
| LLM Analysis (GPT-4) | 68.5% | 81.4% | 0.74 |
| **AgentTrace** | **94.9%** | **98.4%** | **0.97** |

## Repository Structure

```
agenttrace/          # Core framework
  core/              #   Graph construction, backward tracing
  ranking/           #   Node ranking with 17 features (5 groups)
  causal/            #   Causal inference utilities
  inference/         #   Root cause inference pipeline
experiments/
  benchmark/         # 550-scenario benchmark (10 domains)
  baselines/         # Random, First/Last Node, LLM Analysis
  evaluation/        # Evaluation metrics (Hit@k, MRR)
  ablation/          # Feature ablation experiments
  scripts/           # Experiment reproduction scripts
data/
  traces/            # 550 execution trace JSON files
  results/           # Pre-computed experimental results
  ground_truth/      # Ground truth annotations
```

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# For LLM baselines (optional)
pip install -e ".[llm]"
```

## Reproducing Results

```bash
# Run full benchmark evaluation
python -m experiments.scripts.run_paper_experiments

# Run ablation study
python -m experiments.ablation.run_ablation

# Run specific baseline
python -m experiments.baselines.traditional_baselines
python -m experiments.baselines.llm_direct
```

## License

Apache 2.0
