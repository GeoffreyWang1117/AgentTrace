# Bug Injection Methodology

This document describes the bug injection methodology used in AgentTrace's
benchmark construction, addressing reproducibility concerns.

## Overview

AgentTrace evaluates on **two complementary data sources**:

1. **Public benchmarks with existing ground truth** (primary):
   Who&When, AgentRx, MAST, TRAIL — bugs are naturally occurring failures
   annotated by the respective benchmark authors.

2. **Synthetic benchmark** (supplementary):
   Controlled scenarios with injected bugs for topology diversity analysis.

---

## A. Public Benchmark Ground Truth

### Who&When (ICML 2025)
- **Source**: 184 dialogue-based MAS failure scenarios
- **GT annotation**: `mistake_step` (1-indexed) + `mistake_agent` per scenario
- **Failure types**: Knowledge errors, procedural errors, coordination failures
- **No injection needed**: Bugs are part of the original benchmark design

### AgentRx (Microsoft Research)
- **Source**: 115 trajectories from Magentic-One, τ-bench, Flash
- **GT annotation**: `failures` list with `step_number` (0-indexed) + `failed_agent`
- **Failure types**: Real task failures during agent execution
- **No injection needed**: Real-world execution failures with expert annotation

### MAST (NeurIPS 2025)
- **Source**: AG2 framework traces with 22 failure mode taxonomy
- **GT annotation**: `correct=False` flag + failure notes with agent attribution
- **Failure modes**: agent_error, verification, coordination, tool_use, etc.
- **No injection needed**: Annotated by benchmark authors

### TRAIL (Patronus AI)
- **Source**: 148 OpenTelemetry traces from GAIA and SWE-Bench tasks
- **GT annotation**: Error spans with `location` (span_id), `impact` (HIGH/MEDIUM/LOW)
- **No injection needed**: Errors annotated on real execution traces

---

## B. Synthetic Benchmark Bug Injection

### Bug Types (5 categories)

#### 1. Logic Error
Incorrect conditional logic, wrong comparison operator, off-by-one errors.

```python
# Example injection in a research planning scenario:
# CORRECT: agent checks if all sources are verified before synthesis
Message(step=3, from_agent="researcher", to_agent="synthesizer",
        action="send_sources", content="3 verified sources ready",
        is_bug_point=False)

# BUGGY: researcher sends unverified sources
Message(step=3, from_agent="researcher", to_agent="synthesizer",
        action="send_sources", content="3 sources (1 unverified) sent as verified",
        is_bug_point=True,
        bug_description="Logic error: skipped verification check for source 2")
```

#### 2. Communication Failure
Missing or malformed messages between agents, dropped context.

```python
# BUGGY: coordinator drops critical context in handoff
Message(step=4, from_agent="coordinator", to_agent="executor",
        action="delegate_task",
        content="Execute analysis (missing: required format specification)",
        is_bug_point=True,
        bug_description="Communication failure: format requirement dropped during delegation")
```

#### 3. Data Corruption
Invalid data transformation, encoding errors, truncation.

```python
# BUGGY: data processor truncates numerical precision
Message(step=5, from_agent="data_processor", to_agent="analyst",
        action="send_processed_data",
        content="Processed results: [3.1, 2.7, 4.0] (original: [3.14159, 2.71828, 4.00001])",
        is_bug_point=True,
        bug_description="Data corruption: floating point precision lost during processing")
```

#### 4. Missing Validation
Absent input/output validation, unchecked assumptions.

```python
# BUGGY: executor doesn't validate input format
Message(step=2, from_agent="executor", to_agent="tool_api",
        action="call_tool",
        content="API call with unvalidated parameters: {'date': 'invalid-format'}",
        is_bug_point=True,
        bug_description="Missing validation: date format not checked before API call")
```

#### 5. Role Confusion
Agent acting outside its designated role, wrong agent selected.

```python
# BUGGY: wrong agent handles a specialized task
Message(step=6, from_agent="general_assistant", to_agent="user",
        action="provide_medical_advice",
        content="Based on symptoms: likely diagnosis is...",
        is_bug_point=True,
        bug_description="Role confusion: general assistant providing medical advice "
                        "instead of routing to medical specialist agent")
```

### Injection Procedure

1. **Scenario generation**: Define N-step agent interaction flows with
   explicit causal chains (which step's output feeds into which step's input).

2. **Bug placement**: For each scenario, select one step as the bug point.
   - Distribution: controlled via `relative_position` parameter
   - Positions: early (0-33%), middle (33-66%), late (66-100%)

3. **Causal chain construction**: Define explicit edges from bug point
   through intermediate steps to error manifestation.

4. **Validation**:
   - Every bug scenario has `is_bug_point=True` on exactly one step
   - The `root_cause_step` matches the `is_bug_point` step
   - The `error_manifestation_step` is always the last step
   - The causal chain connects root cause to error via intermediate steps
   - `BlindTraceBuilder` strips all GT fields before evaluation

### Anti-Leakage Measures

The `BlindTraceBuilder` class ensures no ground truth information leaks
into the evaluation graph:

1. **No `is_bug_point` in node data**: Stripped during blind trace construction
2. **No `bug_description` in content**: Removed from node metadata
3. **No causal edges**: Only temporal/sequential edges in blind traces
4. **Node type not set to ERROR**: Bug nodes use DECISION type like all others
5. **Ground truth returned separately**: Stored in `GroundTruth` dataclass

```python
# TraceBuilder (GT-aware, for analysis only):
graph, ground_truth = TraceBuilder().build_trace(scenario)
# graph contains causal edges, is_bug_point markers

# BlindTraceBuilder (evaluation-safe):
blind_graph, gt = BlindTraceBuilder().build_blind_trace(scenario)
# blind_graph has ONLY temporal edges, no markers
# gt has root_cause_node_id, causal_edges (for metric computation)
```

---

## C. Verification Protocol

### Automated Checks
- Each generated scenario is validated by `ScenarioValidator`:
  - `root_cause_step` exists in `buggy_flow`
  - `error_manifestation_step` exists and is after root cause
  - Exactly one step has `is_bug_point=True`
  - Causal chain is connected (source→target forms a path)
  - All referenced steps exist in the flow

### Consistency Checks
- Cross-reference: the step marked as `is_bug_point` must equal `root_cause_step`
- Causal chain must start from `root_cause_step` and end at `error_manifestation_step`
- No orphan edges (all source/target steps exist in the flow)

### Reproducibility
- All scenarios use fixed random seed (SEED=42)
- Scenario definitions are JSON files stored in `data/scenarios/`
- Blind traces are deterministically generated from scenarios
- Complete generation scripts in `experiments/benchmark/`
