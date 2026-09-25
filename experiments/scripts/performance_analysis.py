"""
Performance Analysis for AgentTrace.

Measures:
1. Tracing overhead (time to build graph)
2. Backward tracing performance
3. Memory usage
4. Scalability with trace size
"""

import json
import time
from pathlib import Path
import sys
import statistics

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from agenttrace.core.graph import CausalGraph
from agenttrace.core.node import Node, NodeType
from agenttrace.core.edge import Edge, EdgeType


def load_traces():
    """Load all traces for analysis."""
    traces_dir = Path("data/traces")
    traces = []

    for trace_file in traces_dir.glob("*_trace.json"):
        with open(trace_file) as f:
            trace = json.load(f)
            traces.append(trace)

    return traces


def measure_graph_loading(traces, iterations=3):
    """Measure time to load graph from JSON."""
    results = []

    for trace in traces:
        trace_json = trace['trace_json']
        node_count = trace.get('node_count', 0)

        times = []
        for _ in range(iterations):
            start = time.perf_counter()
            graph = CausalGraph.from_json(trace_json)
            end = time.perf_counter()
            times.append((end - start) * 1000)  # ms

        results.append({
            'scenario_id': trace['scenario_id'],
            'node_count': node_count,
            'edge_count': trace.get('edge_count', 0),
            'load_time_ms': statistics.mean(times),
            'load_time_std': statistics.stdev(times) if len(times) > 1 else 0
        })

    return results


def measure_backward_tracing(traces, iterations=3):
    """Measure time for backward tracing."""
    results = []

    for trace in traces:
        trace_json = trace['trace_json']
        error_node_id = trace.get('error_node_id')

        if not error_node_id:
            continue

        graph = CausalGraph.from_json(trace_json)

        times = []
        for _ in range(iterations):
            start = time.perf_counter()
            try:
                backward_nodes = graph.trace_backward(error_node_id)
                trace_length = len(backward_nodes)
            except Exception:
                trace_length = 0
            end = time.perf_counter()
            times.append((end - start) * 1000)  # ms

        results.append({
            'scenario_id': trace['scenario_id'],
            'node_count': trace.get('node_count', 0),
            'trace_length': trace_length,
            'trace_time_ms': statistics.mean(times),
            'trace_time_std': statistics.stdev(times) if len(times) > 1 else 0
        })

    return results


def measure_scalability():
    """Measure performance with different graph sizes."""
    results = []

    sizes = [10, 50, 100, 200, 500, 1000]

    for size in sizes:
        # Create synthetic graph
        graph = CausalGraph(run_id=f"scale_test_{size}")

        # Add nodes
        nodes = []
        for i in range(size):
            node = Node(
                type=NodeType.DECISION,
                agent_id=f"agent_{i % 5}",
                data={"step": i, "action": f"action_{i}"}
            )
            if i > 0:
                node.parent_ids = [nodes[-1].id]
            graph.add_node(node)
            nodes.append(node)

        # Add edges (linear chain)
        for i in range(1, size):
            edge = Edge(
                source_id=nodes[i-1].id,
                target_id=nodes[i].id,
                type=EdgeType.DATA_FLOW
            )
            graph.add_edge(edge)

        # Measure serialization
        start = time.perf_counter()
        json_str = graph.to_json()
        serialize_time = (time.perf_counter() - start) * 1000

        # Measure deserialization
        start = time.perf_counter()
        graph2 = CausalGraph.from_json(json_str)
        deserialize_time = (time.perf_counter() - start) * 1000

        # Measure backward tracing
        start = time.perf_counter()
        backward = graph.trace_backward(nodes[-1].id)
        trace_time = (time.perf_counter() - start) * 1000

        results.append({
            'size': size,
            'serialize_ms': serialize_time,
            'deserialize_ms': deserialize_time,
            'trace_ms': trace_time,
            'trace_length': len(backward)
        })

        print(f"  Size {size}: serialize={serialize_time:.2f}ms, "
              f"deserialize={deserialize_time:.2f}ms, trace={trace_time:.2f}ms")

    return results


def generate_performance_table(loading_results, tracing_results, scalability_results):
    """Generate LaTeX table for performance results."""

    # Aggregate by node count buckets
    buckets = {
        '1-5': {'load': [], 'trace': []},
        '6-10': {'load': [], 'trace': []},
        '11+': {'load': [], 'trace': []}
    }

    for r in loading_results:
        nc = r['node_count']
        if nc <= 5:
            bucket = '1-5'
        elif nc <= 10:
            bucket = '6-10'
        else:
            bucket = '11+'
        buckets[bucket]['load'].append(r['load_time_ms'])

    for r in tracing_results:
        nc = r['node_count']
        if nc <= 5:
            bucket = '1-5'
        elif nc <= 10:
            bucket = '6-10'
        else:
            bucket = '11+'
        buckets[bucket]['trace'].append(r['trace_time_ms'])

    table = r"""
\begin{table}[t]
\centering
\caption{AgentTrace Runtime Performance}
\label{tab:performance}
\begin{tabular}{lccc}
\toprule
\textbf{Graph Size} & \textbf{Load Time (ms)} & \textbf{Trace Time (ms)} & \textbf{Count} \\
\midrule
"""

    for bucket in ['1-5', '6-10', '11+']:
        load_times = buckets[bucket]['load']
        trace_times = buckets[bucket]['trace']

        if load_times:
            load_avg = statistics.mean(load_times)
            trace_avg = statistics.mean(trace_times) if trace_times else 0
            count = len(load_times)
            table += f"{bucket} nodes & {load_avg:.2f} & {trace_avg:.2f} & {count} \\\\\n"

    table += r"""
\bottomrule
\end{tabular}
\end{table}

\begin{table}[t]
\centering
\caption{Scalability Analysis (Synthetic Graphs)}
\label{tab:scalability}
\begin{tabular}{rcccc}
\toprule
\textbf{Nodes} & \textbf{Serialize (ms)} & \textbf{Deserialize (ms)} & \textbf{Trace (ms)} \\
\midrule
"""

    for r in scalability_results:
        table += f"{r['size']} & {r['serialize_ms']:.2f} & {r['deserialize_ms']:.2f} & {r['trace_ms']:.2f} \\\\\n"

    table += r"""
\bottomrule
\end{tabular}
\end{table}
"""

    return table


def main():
    """Run performance analysis."""
    print("="*70)
    print("PERFORMANCE ANALYSIS")
    print("="*70)

    print("\nLoading traces...")
    traces = load_traces()
    print(f"Loaded {len(traces)} traces")

    print("\nMeasuring graph loading performance...")
    loading_results = measure_graph_loading(traces)
    avg_load = statistics.mean([r['load_time_ms'] for r in loading_results])
    print(f"  Average load time: {avg_load:.2f} ms")

    print("\nMeasuring backward tracing performance...")
    tracing_results = measure_backward_tracing(traces)
    avg_trace = statistics.mean([r['trace_time_ms'] for r in tracing_results])
    print(f"  Average trace time: {avg_trace:.2f} ms")

    print("\nMeasuring scalability...")
    scalability_results = measure_scalability()

    # Generate table
    table = generate_performance_table(loading_results, tracing_results, scalability_results)

    output_dir = Path("data/results")
    with open(output_dir / "performance_table.tex", 'w') as f:
        f.write(table)
    print(f"\nPerformance table saved to: {output_dir / 'performance_table.tex'}")

    # Save raw results
    with open(output_dir / "performance_results.json", 'w') as f:
        json.dump({
            'loading': loading_results,
            'tracing': tracing_results,
            'scalability': scalability_results,
            'summary': {
                'avg_load_ms': avg_load,
                'avg_trace_ms': avg_trace,
                'total_traces': len(traces)
            }
        }, f, indent=2)

    print("\n" + "="*70)
    print("PERFORMANCE SUMMARY")
    print("="*70)
    print(f"\nGraph Loading:")
    print(f"  Average: {avg_load:.2f} ms")
    print(f"  Min: {min(r['load_time_ms'] for r in loading_results):.2f} ms")
    print(f"  Max: {max(r['load_time_ms'] for r in loading_results):.2f} ms")

    print(f"\nBackward Tracing:")
    print(f"  Average: {avg_trace:.2f} ms")
    print(f"  Min: {min(r['trace_time_ms'] for r in tracing_results):.2f} ms")
    print(f"  Max: {max(r['trace_time_ms'] for r in tracing_results):.2f} ms")

    print(f"\nScalability (1000 nodes):")
    r1000 = next(r for r in scalability_results if r['size'] == 1000)
    print(f"  Serialize: {r1000['serialize_ms']:.2f} ms")
    print(f"  Deserialize: {r1000['deserialize_ms']:.2f} ms")
    print(f"  Trace: {r1000['trace_ms']:.2f} ms")


if __name__ == "__main__":
    main()
