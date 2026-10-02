# AgentTrace

**Zero-inference failure localization for multi-agent LLM systems.**

[![arXiv](https://img.shields.io/badge/arXiv-2603.14688-b31b1b.svg)](https://arxiv.org/abs/2603.14688)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

AgentTrace reconstructs a causal graph from a multi-agent execution log and ranks the steps and
agents most likely to have caused a failure. It makes **no LLM calls at diagnosis time**, runs on
CPU in tens of milliseconds per trace, and costs nothing to run. Use it as a first-pass triage
before you spend an LLM budget, not as a replacement for one.

<p align="center">
  <img src="assets/framework_overview.png" width="100%" alt="AgentTrace framework overview"/>
</p>

## Install

```bash
git clone https://github.com/GeoffreyWang1117/AgentTrace.git
cd AgentTrace
pip install -e ".[dev]"      # add ".[llm]" for the optional LLM baselines
python demo.py               # 5-agent trace with a planted bug, ranked in ~1 ms
pytest -q                    # 51 tests
```

## Use

```python
from agenttrace.core.graph import CausalGraph
from agenttrace.core.node import Node, NodeType
from agenttrace.ranking.ranker import ImprovedAgentTrace

graph = CausalGraph(run_id="my_run")
# add one Node per agent step (agent_id, content, timestamp, parent_ids) ...
# then rank root-cause candidates for the failing node
```

`demo.py` is the complete runnable example. The CLI (`agenttrace demo | analyze <trace.json> |
serve`) wraps the same pipeline. Adapters for public failure-attribution benchmarks
(Who&When, AgentRx, MAST, TRAIL, TraceElephant) are in `experiments/adapters/`.

## Layout

| path | contents |
|---|---|
| `agenttrace/core` | graph, node, and edge types |
| `agenttrace/inference` | edge inference: temporal, data-flow, optional semantic (sentence-embedding) edges |
| `agenttrace/ranking`, `agenttrace/causal` | structural scorers and the counterfactual (SCM) scorer |
| `agenttrace/hooks`, `tracer.py` | decorators for capturing traces from a live system |
| `experiments/adapters` | benchmark loaders |
| `experiments/baselines` | position, LLM-direct, CoT, ReAct, GNN, causal-discovery baselines |
| `data/` | the synthetic benchmark and results of the workshop paper |

## What to expect: known limitations

These findings came out of evaluating AgentTrace on public benchmarks after the workshop paper
was published. Please read them before you rely on the tool.

- **Chain-shaped traces.** When a trace is a single chain (as in most current frameworks'
  logs), every upstream step is a cut vertex, so the counterfactual responsibility score is
  identical for all candidates and cannot discriminate between them. The ranking then falls back to
  structural features. Ties are broken deterministically by node id, so a tie is still a tie:
  treat the top-1 answer on a fully tied chain as no better than picking a node uniformly at random.
- **Accuracy on public benchmarks** (agent-level Hit@1, expected over ties, adapters as of 2026-10;
  see *Corrections* below). Who&When, all 184 traces: the counterfactual scorer (`CausalAttributor`)
  reaches 38.3%, exactly what picking a step uniformly at random gives, because these traces are
  chains. The default structural ranker (`ImprovedAgentTrace`) reaches 34.2%: it ranks the human task
  message first on every Hand-Crafted trace, so it scores 0.0% on that subset and 50.0% on
  Algorithm-Generated. AgentRx Magentic-One (43 traces with a resolvable root cause): 49.4% for the
  counterfactual scorer, 49.4% uniform, 9.3% for the ranker. Earlier versions of this README quoted
  39% / 54% and a comparison with an LLM judge; those numbers came from the defective adapters and
  are withdrawn. We have not re-measured the LLM comparison.
- **Semantic edges.** The optional semantic edges change rankings substantially and in either
  direction. The earlier evidence that they hurt on AgentRx came from the defective AgentRx adapter
  and is withdrawn. Validate them on your own traces before enabling them by default.

## Corrections (2026-10)

Two benchmark adapters in `experiments/adapters/` were wrong; both are fixed in this version.

- **Who&When** (`who_and_when_adapter.py`): `mistake_step` is a 0-based index into `history` (the
  speaker of `history[k]` matches `mistake_agent` in 179/184 traces, against 42/184 for
  `history[k-1]`). The adapter read it as 1-based, so every root cause was one step early, and the 20
  traces labelled step 0 were mapped to the last node. The adapter now uses the 0-based index and
  appends an explicit outcome node as the error node, so all 184 traces load.
- **AgentRx** (`agentrx_adapter.py`): the old adapter paired the category-named example
  trajectories with ground truth through a fallback that always returned the first entry, built
  placeholder traces for the τ-bench ground truth (whose trajectories are not in the release) with
  the root-cause text on the root-cause node only, and used the earliest failure instead of the
  annotated root cause. It now loads only the Magentic-One trajectories that have ground truth,
  uses the root-cause failure, and converts its 1-based `step_number`.

Numbers computed with the old adapters, including the ones previously quoted in this README, are not
valid. The workshop paper (tag below) used its own synthetic benchmark, not these adapters.

## Reproducing the workshop paper

The exact artifact cited by arXiv:2603.14688 (ICLR 2026 Workshop on Agents in the Wild) is
preserved at tag
[`iclr2026-aiwild-camera-ready`](https://github.com/GeoffreyWang1117/AgentTrace/tree/iclr2026-aiwild-camera-ready):

```bash
git checkout iclr2026-aiwild-camera-ready
python -m experiments.scripts.run_paper_experiments
```

`main` has since changed. Scorer ties are now resolved deterministically, and the benchmark
adapters have been corrected, including TRAIL ground-truth resolution. Some benchmark numbers
in the workshop paper are therefore superseded, in particular the comparisons against LLM-based
attribution. Treat the tag as the record of what that paper ran, and `main` as the current tool.

## Citation

```bibtex
@article{wang2026agenttrace,
  title   = {AgentTrace: Causal Graph Tracing for Root Cause Analysis in Deployed Multi-Agent Systems},
  author  = {Wang, Zhaohui Geoffrey},
  journal = {arXiv preprint arXiv:2603.14688},
  year    = {2026},
  note    = {ICLR 2026 Workshop on Agents in the Wild}
}
```

## License

Apache 2.0. Benchmark data loaded by the adapters is **not** redistributed here. Obtain each
benchmark from its original release, under its own license.
