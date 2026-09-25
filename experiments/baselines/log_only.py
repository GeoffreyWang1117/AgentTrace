"""
Log-Only Baseline for AgentTrace Benchmark.

This baseline represents traditional log-based debugging where:
- Only sequential logs are available
- No causal relationships are inferred
- Root cause is identified by searching for error patterns
"""

import re
from dataclasses import dataclass
from typing import Optional


@dataclass
class LogEntry:
    """A single log entry."""
    timestamp: str
    level: str  # INFO, WARNING, ERROR
    agent_id: str
    message: str
    data: Optional[dict] = None


@dataclass
class LogOnlyResult:
    """Result of log-only analysis."""
    scenario_id: str

    # Identified errors
    error_entries: list[LogEntry]

    # Root cause guess (based on first error)
    guessed_root_cause: Optional[str]
    guessed_root_cause_index: int

    # Trace (just sequential log entries)
    log_trace: list[str]

    # No causal chain (this is the limitation)
    causal_chain: list = None  # Always None for this baseline


class LogOnlyAnalyzer:
    """
    Analyzes execution traces using only log information.

    This represents the baseline approach of reading logs sequentially
    without any causal inference.
    """

    def __init__(self):
        self.error_patterns = [
            r"error",
            r"exception",
            r"failed",
            r"failure",
            r"invalid",
            r"timeout",
            r"crash",
        ]

    def parse_trace_to_logs(self, trace_json: str) -> list[LogEntry]:
        """Convert a trace to simple log entries."""
        import json
        from agenttrace.core.graph import CausalGraph

        graph = CausalGraph.from_json(trace_json)
        logs = []

        for node in sorted(graph, key=lambda n: n.timestamp):
            # Determine log level
            level = "INFO"
            if node.type.value == "error":
                level = "ERROR"
            elif "error" in str(node.data).lower():
                level = "ERROR"
            elif "warning" in str(node.data).lower():
                level = "WARNING"

            logs.append(LogEntry(
                timestamp=node.timestamp.isoformat(),
                level=level,
                agent_id=node.agent_id,
                message=f"{node.type.value}: {node.data}",
                data=node.data if isinstance(node.data, dict) else {"raw": node.data}
            ))

        return logs

    def find_errors(self, logs: list[LogEntry]) -> list[LogEntry]:
        """Find all error entries in logs."""
        errors = []
        for log in logs:
            if log.level == "ERROR":
                errors.append(log)
            else:
                # Check for error patterns in message
                for pattern in self.error_patterns:
                    if re.search(pattern, log.message, re.IGNORECASE):
                        errors.append(log)
                        break
        return errors

    def guess_root_cause(self, logs: list[LogEntry], errors: list[LogEntry]) -> tuple[Optional[str], int]:
        """
        Guess root cause based on simple heuristics.

        Strategy: First error in the log is likely the root cause.
        This is a naive approach that often fails in complex systems.
        """
        if not errors:
            return None, -1

        # Find the first error
        first_error = errors[0]
        first_error_index = logs.index(first_error) if first_error in logs else 0

        return first_error.message, first_error_index

    def analyze(self, trace_json: str, scenario_id: str) -> LogOnlyResult:
        """Analyze a trace using log-only approach."""
        logs = self.parse_trace_to_logs(trace_json)
        errors = self.find_errors(logs)
        root_cause, rc_index = self.guess_root_cause(logs, errors)

        return LogOnlyResult(
            scenario_id=scenario_id,
            error_entries=errors,
            guessed_root_cause=root_cause,
            guessed_root_cause_index=rc_index,
            log_trace=[f"[{l.timestamp}] [{l.level}] [{l.agent_id}] {l.message}" for l in logs],
            causal_chain=None  # No causal inference
        )

    def format_output(self, result: LogOnlyResult) -> str:
        """Format result for LLM judge evaluation."""
        lines = [
            f"=== Log-Only Analysis for {result.scenario_id} ===",
            "",
            "Found Errors:",
        ]

        for i, err in enumerate(result.error_entries[:5]):  # Limit to 5
            lines.append(f"  {i+1}. [{err.agent_id}] {err.message[:100]}")

        lines.extend([
            "",
            f"Guessed Root Cause: {result.guessed_root_cause or 'Unknown'}",
            f"Root Cause Location: Log entry #{result.guessed_root_cause_index + 1}",
            "",
            "Causal Chain: NOT AVAILABLE (log-only baseline)",
            "",
            "Analysis Method: Sequential log scanning with error pattern matching",
        ])

        return "\n".join(lines)


def run_log_only_baseline(
    traces_dir: str = "data/traces",
    output_dir: str = "data/results/log_only"
) -> dict[str, LogOnlyResult]:
    """Run log-only baseline on all traces."""
    import json
    from pathlib import Path

    traces_path = Path(traces_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    analyzer = LogOnlyAnalyzer()
    results = {}

    for trace_file in traces_path.glob("*_trace.json"):
        with open(trace_file) as f:
            trace_data = json.load(f)

        scenario_id = trace_data["scenario_id"]
        trace_json = trace_data["trace_json"]

        result = analyzer.analyze(trace_json, scenario_id)
        results[scenario_id] = result

        # Save result
        output_file = output_path / f"{scenario_id}_log_only.json"
        with open(output_file, 'w') as f:
            json.dump({
                "scenario_id": result.scenario_id,
                "guessed_root_cause": result.guessed_root_cause,
                "guessed_root_cause_index": result.guessed_root_cause_index,
                "error_count": len(result.error_entries),
                "formatted_output": analyzer.format_output(result)
            }, f, indent=2)

    return results


if __name__ == "__main__":
    results = run_log_only_baseline()
    print(f"Analyzed {len(results)} traces with log-only baseline")
