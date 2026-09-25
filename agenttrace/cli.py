#!/usr/bin/env python3
"""
Command-line interface for AgentTrace.
"""

import argparse
import sys
from pathlib import Path


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="AgentTrace - Multi-Agent Causal Tracing System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Start the debugging UI server
    agenttrace serve

    # Run the demo
    agenttrace demo

    # Analyze a saved trace
    agenttrace analyze trace.json
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # serve command
    serve_parser = subparsers.add_parser("serve", help="Start the debugging UI server")
    serve_parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host to bind to (default: 127.0.0.1)",
    )
    serve_parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind to (default: 8000)",
    )
    serve_parser.add_argument(
        "--storage-path",
        type=str,
        help="Path for persistent storage (default: in-memory)",
    )

    # demo command
    demo_parser = subparsers.add_parser("demo", help="Run the interactive demo")
    demo_parser.add_argument(
        "--orders",
        type=int,
        default=15,
        help="Number of orders to simulate (default: 15)",
    )

    # analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Analyze a saved trace file")
    analyze_parser.add_argument("trace_file", help="Path to the trace JSON file")
    analyze_parser.add_argument(
        "--errors",
        action="store_true",
        help="Show error analysis",
    )
    analyze_parser.add_argument(
        "--stats",
        action="store_true",
        help="Show statistics",
    )

    # version command
    subparsers.add_parser("version", help="Show version information")

    args = parser.parse_args()

    if args.command == "serve":
        cmd_serve(args)
    elif args.command == "demo":
        cmd_demo(args)
    elif args.command == "analyze":
        cmd_analyze(args)
    elif args.command == "version":
        cmd_version()
    else:
        parser.print_help()
        sys.exit(1)


def cmd_serve(args):
    """Start the debugging UI server."""
    from agenttrace.api.server import create_app, set_storage
    from agenttrace.storage.memory import MemoryStorage
    from agenttrace.storage.temporal_db import TemporalStorage

    import uvicorn

    # Choose storage backend
    if args.storage_path:
        storage = TemporalStorage(args.storage_path)
        print(f"Using persistent storage at: {args.storage_path}")
    else:
        storage = MemoryStorage()
        print("Using in-memory storage")

    set_storage(storage)

    # Create app
    app = create_app(storage=storage)

    # Mount UI static files
    from fastapi.staticfiles import StaticFiles
    ui_path = Path(__file__).parent.parent / "ui"
    if ui_path.exists():
        app.mount("/ui", StaticFiles(directory=str(ui_path)), name="ui")

    print(f"\nStarting AgentTrace server at http://{args.host}:{args.port}")
    print("Press Ctrl+C to stop\n")

    uvicorn.run(app, host=args.host, port=args.port)


def cmd_demo(args):
    """Run the interactive demo."""
    # Import demo module
    import importlib.util
    demo_path = Path(__file__).parent.parent / "examples" / "debug_demo.py"

    if demo_path.exists():
        spec = importlib.util.spec_from_file_location("debug_demo", demo_path)
        demo = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(demo)
        demo.main()
    else:
        print(f"Demo file not found: {demo_path}")
        print("Running simple demo instead...")

        from agenttrace.examples.simple_agents import run_demo
        run_demo()


def cmd_analyze(args):
    """Analyze a saved trace file."""
    from agenttrace.tracer import Tracer
    from agenttrace.query.interface import QueryInterface
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel

    console = Console()

    # Load trace
    try:
        tracer = Tracer.load(args.trace_file)
    except Exception as e:
        console.print(f"[red]Error loading trace: {e}[/red]")
        sys.exit(1)

    query = QueryInterface(tracer.graph)
    stats = tracer.get_statistics()

    console.print(Panel(f"[bold]AgentTrace Analysis[/bold]\nRun ID: {stats['run_id']}"))

    # Show statistics
    if args.stats or not args.errors:
        table = Table(title="Statistics")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Nodes", str(stats["node_count"]))
        table.add_row("Edges", str(stats["edge_count"]))
        table.add_row("Agents", ", ".join(stats["agents"]))
        table.add_row("Is DAG", str(stats["is_dag"]))

        console.print(table)

        # Node types
        type_table = Table(title="Node Types")
        type_table.add_column("Type", style="cyan")
        type_table.add_column("Count", style="green")

        for node_type, count in stats["node_types"].items():
            type_table.add_row(node_type, str(count))

        console.print(type_table)

    # Show errors
    if args.errors:
        errors = tracer.find_errors()

        if not errors:
            console.print("[green]No errors found in trace[/green]")
        else:
            console.print(f"\n[red]Found {len(errors)} errors:[/red]")

            for i, error in enumerate(errors, 1):
                explanation = query.explain_error(error.id)

                console.print(Panel(
                    f"[bold]Error {i}[/bold]\n"
                    f"Type: {explanation['error_info']['type']}\n"
                    f"Message: {explanation['error_info']['message']}\n"
                    f"Agent: {explanation['error_info']['agent']}\n"
                    f"Root causes: {len(explanation['root_causes'])}",
                    title=f"Error {i}",
                    border_style="red",
                ))

                if explanation["likely_causes"]:
                    console.print("[bold]Likely causes:[/bold]")
                    for cause in explanation["likely_causes"][:3]:
                        console.print(f"  - Score: {cause['score']:.2f}")
                        for reason in cause["reasons"]:
                            console.print(f"    {reason}")


def cmd_version():
    """Show version information."""
    from agenttrace import __version__
    print(f"AgentTrace version {__version__}")


if __name__ == "__main__":
    main()
