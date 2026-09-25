"""Tests for the Tracer class."""

import pytest
import time

from agenttrace.tracer import Tracer, get_tracer
from agenttrace.core.node import NodeType
from agenttrace.core.edge import EdgeType
from agenttrace.hooks.decorators import trace_agent, trace_tool


class TestTracer:
    """Tests for Tracer class."""

    def test_create_tracer(self):
        tracer = Tracer(run_id="test")
        assert tracer.run_id == "test"
        assert not tracer.is_active

    def test_start_stop(self):
        tracer = Tracer()
        tracer.start()

        assert tracer.is_active
        assert get_tracer() is tracer

        tracer.stop()
        assert not tracer.is_active
        assert get_tracer() is None

    def test_context_manager(self):
        with Tracer() as tracer:
            assert tracer.is_active
            assert get_tracer() is tracer

        assert not tracer.is_active

    def test_record_node(self):
        with Tracer() as tracer:
            node = tracer.record(
                node_type=NodeType.AGENT_INPUT,
                agent_id="test_agent",
                data={"message": "hello"},
            )

            assert node.type == NodeType.AGENT_INPUT
            assert node.agent_id == "test_agent"
            assert node.run_id == tracer.run_id
            assert tracer.graph.node_count == 1

    def test_link_nodes(self):
        with Tracer() as tracer:
            n1 = tracer.record(NodeType.AGENT_INPUT, "agent", "input")
            n2 = tracer.record(NodeType.AGENT_OUTPUT, "agent", "output")

            edge = tracer.link(n1.id, n2.id, EdgeType.INPUT_OUTPUT)

            assert edge.source_id == n1.id
            assert edge.target_id == n2.id
            assert tracer.graph.edge_count == 1

    def test_checkpoint(self):
        with Tracer() as tracer:
            cp = tracer.checkpoint("agent", "important_point", {"state": "saved"})

            assert cp.type == NodeType.CHECKPOINT
            assert cp.metadata["checkpoint_name"] == "important_point"

    def test_span(self):
        with Tracer() as tracer:
            with tracer.span("agent", "operation", "input_data") as span:
                # Simulate work
                time.sleep(0.01)

            # Should have created input node
            assert span.type == NodeType.AGENT_INPUT
            assert tracer.graph.node_count >= 1

    def test_span_with_error(self):
        with Tracer() as tracer:
            with pytest.raises(ValueError):
                with tracer.span("agent", "failing_op"):
                    raise ValueError("Test error")

            # Should have error node
            errors = tracer.find_errors()
            assert len(errors) == 1
            assert "Test error" in errors[0].data["error_message"]

    def test_trace_forward(self):
        with Tracer() as tracer:
            n1 = tracer.record(NodeType.AGENT_INPUT, "a", "1")
            n2 = tracer.record(NodeType.AGENT_OUTPUT, "a", "2", parent_ids=[n1.id])
            n3 = tracer.record(NodeType.AGENT_OUTPUT, "a", "3", parent_ids=[n2.id])

            affected = tracer.trace_forward(n1.id)

            assert len(affected) == 2
            assert n2 in affected
            assert n3 in affected

    def test_trace_backward(self):
        with Tracer() as tracer:
            n1 = tracer.record(NodeType.AGENT_INPUT, "a", "1")
            n2 = tracer.record(NodeType.AGENT_OUTPUT, "a", "2", parent_ids=[n1.id])
            n3 = tracer.record(NodeType.AGENT_OUTPUT, "a", "3", parent_ids=[n2.id])

            causes = tracer.trace_backward(n3.id)

            assert len(causes) == 2
            assert n1 in causes
            assert n2 in causes

    def test_find_root_cause(self):
        with Tracer() as tracer:
            root = tracer.record(NodeType.AGENT_INPUT, "a", "root")
            mid = tracer.record(NodeType.AGENT_OUTPUT, "a", "mid", parent_ids=[root.id])
            end = tracer.record(NodeType.ERROR, "a", {"error": "oops"}, parent_ids=[mid.id])

            roots = tracer.find_root_cause(end.id)

            assert len(roots) == 1
            assert roots[0].id == root.id

    def test_analyze_error(self):
        with Tracer() as tracer:
            n1 = tracer.record(NodeType.AGENT_INPUT, "a", "input")
            n2 = tracer.record(NodeType.TOOL_CALL, "a", "call", parent_ids=[n1.id])
            err = tracer.record(
                NodeType.ERROR,
                "a",
                {"error_type": "RuntimeError", "error_message": "failed"},
                parent_ids=[n2.id],
            )

            analysis = tracer.analyze_error(err.id)

            assert "error_node" in analysis
            assert "root_causes" in analysis
            assert "causal_chains" in analysis

    def test_save_load(self, tmp_path):
        path = tmp_path / "trace.json"

        with Tracer(run_id="save_test") as tracer:
            tracer.record(NodeType.AGENT_INPUT, "a", "data1")
            tracer.record(NodeType.AGENT_OUTPUT, "a", "data2")
            tracer.save(str(path))

        loaded = Tracer.load(str(path))

        assert loaded.graph.run_id == "save_test"
        assert loaded.graph.node_count == 2

    def test_statistics(self):
        with Tracer() as tracer:
            tracer.record(NodeType.AGENT_INPUT, "agent1", "1")
            tracer.record(NodeType.AGENT_OUTPUT, "agent1", "2")
            tracer.record(NodeType.TOOL_CALL, "agent2", "3")

            stats = tracer.get_statistics()

            assert stats["node_count"] == 3
            assert "agent1" in stats["agents"]
            assert "agent2" in stats["agents"]


class TestDecorators:
    """Tests for tracing decorators."""

    def test_trace_agent_decorator(self):
        @trace_agent(agent_id="decorated_agent")
        def my_function(x: int) -> int:
            return x * 2

        with Tracer() as tracer:
            result = my_function(5)

            assert result == 10
            assert tracer.graph.node_count >= 2  # Input and output

    def test_trace_agent_with_error(self):
        @trace_agent(agent_id="error_agent")
        def failing_function():
            raise ValueError("Intentional error")

        with Tracer() as tracer:
            with pytest.raises(ValueError):
                failing_function()

            errors = tracer.find_errors()
            assert len(errors) == 1

    def test_trace_tool_decorator(self):
        @trace_tool(tool_name="my_tool")
        def tool_function(query: str) -> list:
            return [f"result for {query}"]

        with Tracer() as tracer:
            result = tool_function("test")

            assert result == ["result for test"]

            tool_calls = tracer.graph.get_nodes_by_type(NodeType.TOOL_CALL)
            tool_results = tracer.graph.get_nodes_by_type(NodeType.TOOL_RESULT)

            assert len(tool_calls) == 1
            assert len(tool_results) == 1

    def test_nested_tracing(self):
        @trace_agent(agent_id="outer")
        def outer_function():
            return inner_function()

        @trace_agent(agent_id="inner")
        def inner_function():
            return "result"

        with Tracer() as tracer:
            result = outer_function()

            assert result == "result"

            outer_nodes = tracer.graph.get_nodes_by_agent("outer")
            inner_nodes = tracer.graph.get_nodes_by_agent("inner")

            assert len(outer_nodes) >= 2
            assert len(inner_nodes) >= 2

    def test_auto_detect_agent_id(self):
        class MyAgent:
            agent_id = "auto_detected"

            @trace_agent()
            def process(self, data):
                return f"processed: {data}"

        with Tracer() as tracer:
            agent = MyAgent()
            result = agent.process("input")

            assert result == "processed: input"

            nodes = tracer.graph.get_nodes_by_agent("auto_detected")
            assert len(nodes) >= 2
