"""Tests for the inference engine."""

import pytest
from datetime import datetime, timedelta

from agenttrace.core.graph import CausalGraph
from agenttrace.core.node import Node, NodeType
from agenttrace.core.edge import EdgeType
from agenttrace.inference.dataflow import DataFlowAnalyzer
from agenttrace.inference.temporal import TemporalAnalyzer
from agenttrace.inference.engine import InferenceEngine


class TestDataFlowAnalyzer:
    """Tests for DataFlowAnalyzer."""

    @pytest.fixture
    def analyzer(self):
        return DataFlowAnalyzer(similarity_threshold=0.5)

    def test_exact_match(self, analyzer):
        graph = CausalGraph()

        n1 = Node(
            type=NodeType.AGENT_OUTPUT,
            agent_id="a",
            data={"result": "hello world"},
        )
        # Add small delay for ordering
        n2 = Node(
            type=NodeType.AGENT_INPUT,
            agent_id="b",
            data={"result": "hello world"},
            timestamp=datetime.now() + timedelta(milliseconds=10),
        )

        graph.add_node(n1)
        graph.add_node(n2)

        edges = analyzer.analyze(graph)

        assert len(edges) == 1
        assert edges[0].confidence >= 0.9

    def test_partial_match(self, analyzer):
        graph = CausalGraph()

        n1 = Node(
            type=NodeType.AGENT_OUTPUT,
            agent_id="a",
            data="The quick brown fox",
        )
        n2 = Node(
            type=NodeType.AGENT_INPUT,
            agent_id="b",
            data="The quick brown fox jumps",
            timestamp=datetime.now() + timedelta(milliseconds=10),
        )

        graph.add_node(n1)
        graph.add_node(n2)

        edges = analyzer.analyze(graph)

        # Should find a match due to containment
        assert len(edges) >= 1

    def test_no_match(self, analyzer):
        graph = CausalGraph()

        n1 = Node(
            type=NodeType.AGENT_OUTPUT,
            agent_id="a",
            data="completely different",
        )
        n2 = Node(
            type=NodeType.AGENT_INPUT,
            agent_id="b",
            data="unrelated content xyz",
            timestamp=datetime.now() + timedelta(milliseconds=10),
        )

        graph.add_node(n1)
        graph.add_node(n2)

        edges = analyzer.analyze(graph)

        # Should not find a strong match
        assert all(e.confidence < 0.5 for e in edges)

    def test_find_data_sources(self, analyzer):
        graph = CausalGraph()

        source1 = Node(
            type=NodeType.AGENT_OUTPUT,
            agent_id="a",
            data={"key": "value123"},
        )
        source2 = Node(
            type=NodeType.AGENT_OUTPUT,
            agent_id="b",
            data={"key": "different"},
            timestamp=datetime.now() + timedelta(milliseconds=5),
        )
        target = Node(
            type=NodeType.AGENT_INPUT,
            agent_id="c",
            data={"key": "value123"},
            timestamp=datetime.now() + timedelta(milliseconds=10),
        )

        graph.add_node(source1)
        graph.add_node(source2)
        graph.add_node(target)

        sources = analyzer.find_data_sources(graph, target.id)

        # Should find source1 as the best match
        assert len(sources) >= 1
        assert sources[0][0].id == source1.id


class TestTemporalAnalyzer:
    """Tests for TemporalAnalyzer."""

    @pytest.fixture
    def analyzer(self):
        return TemporalAnalyzer(max_time_delta=timedelta(seconds=1))

    def test_temporal_sequence(self, analyzer):
        graph = CausalGraph()
        now = datetime.now()

        n1 = Node(
            type=NodeType.AGENT_INPUT,
            agent_id="a",
            data="1",
            timestamp=now,
        )
        n2 = Node(
            type=NodeType.AGENT_OUTPUT,
            agent_id="a",
            data="2",
            timestamp=now + timedelta(milliseconds=100),
        )
        n3 = Node(
            type=NodeType.TOOL_CALL,
            agent_id="a",
            data="3",
            timestamp=now + timedelta(milliseconds=200),
        )

        graph.add_node(n1)
        graph.add_node(n2)
        graph.add_node(n3)

        edges = analyzer.analyze(graph)

        # Should find temporal relationships
        assert len(edges) >= 1

    def test_inter_agent_trigger(self, analyzer):
        graph = CausalGraph()
        now = datetime.now()

        output = Node(
            type=NodeType.AGENT_OUTPUT,
            agent_id="agent_a",
            data="trigger",
            timestamp=now,
        )
        response = Node(
            type=NodeType.AGENT_INPUT,
            agent_id="agent_b",
            data="response",
            timestamp=now + timedelta(milliseconds=50),
        )

        graph.add_node(output)
        graph.add_node(response)

        edges = analyzer.analyze(graph)

        # Should detect trigger-response pattern
        trigger_edges = [e for e in edges if e.type == EdgeType.TRIGGER_RESPONSE]
        assert len(trigger_edges) >= 1

    def test_time_gap_too_large(self, analyzer):
        graph = CausalGraph()
        now = datetime.now()

        n1 = Node(
            type=NodeType.AGENT_OUTPUT,
            agent_id="a",
            data="1",
            timestamp=now,
        )
        n2 = Node(
            type=NodeType.AGENT_INPUT,
            agent_id="b",
            data="2",
            timestamp=now + timedelta(seconds=5),  # Too far apart
        )

        graph.add_node(n1)
        graph.add_node(n2)

        edges = analyzer.analyze(graph)

        # Should not find edges for large time gap
        assert len(edges) == 0

    def test_detect_anomalies(self, analyzer):
        graph = CausalGraph()
        now = datetime.now()

        nodes = [
            Node(type=NodeType.AGENT_INPUT, agent_id="a", data="1", timestamp=now),
            Node(type=NodeType.AGENT_OUTPUT, agent_id="a", data="2",
                 timestamp=now + timedelta(milliseconds=100)),
            # Long gap here
            Node(type=NodeType.AGENT_INPUT, agent_id="a", data="3",
                 timestamp=now + timedelta(seconds=5)),
        ]

        for n in nodes:
            graph.add_node(n)

        anomalies = analyzer.detect_anomalies(graph)

        # Should detect the long gap
        assert len(anomalies) >= 1
        assert any(a["type"] == "long_gap" for a in anomalies)


class TestInferenceEngine:
    """Tests for the combined inference engine."""

    @pytest.fixture
    def engine(self):
        return InferenceEngine(min_confidence=0.4)

    def test_combined_inference(self, engine):
        graph = CausalGraph()
        now = datetime.now()

        # Create nodes with both data flow and temporal relationships
        n1 = Node(
            type=NodeType.AGENT_OUTPUT,
            agent_id="producer",
            data={"result": "data_payload"},
            timestamp=now,
        )
        n2 = Node(
            type=NodeType.AGENT_INPUT,
            agent_id="consumer",
            data={"result": "data_payload"},  # Same data = strong data flow
            timestamp=now + timedelta(milliseconds=50),  # Close in time
        )

        graph.add_node(n1)
        graph.add_node(n2)

        edges = engine.infer_relationships(graph, apply_to_graph=True)

        # Should have high confidence due to both signals
        assert len(edges) >= 1
        assert edges[0].confidence >= 0.6

    def test_analyze_causality(self, engine):
        graph = CausalGraph()
        now = datetime.now()

        nodes = [
            Node(type=NodeType.AGENT_INPUT, agent_id="a", data="start", timestamp=now),
            Node(type=NodeType.AGENT_OUTPUT, agent_id="a", data="mid",
                 timestamp=now + timedelta(milliseconds=50)),
            Node(type=NodeType.AGENT_OUTPUT, agent_id="a", data="end",
                 timestamp=now + timedelta(milliseconds=100)),
        ]

        for n in nodes:
            graph.add_node(n)

        # Add edges
        from agenttrace.core.edge import Edge
        graph.add_edge(Edge(
            source_id=nodes[0].id,
            target_id=nodes[1].id,
            type=EdgeType.INPUT_OUTPUT,
        ))
        graph.add_edge(Edge(
            source_id=nodes[1].id,
            target_id=nodes[2].id,
            type=EdgeType.DATA_FLOW,
        ))

        analysis = engine.analyze_causality(graph, nodes[2].id)

        assert "target_node" in analysis
        assert "all_causes" in analysis

    def test_suggest_likely_causes(self, engine):
        graph = CausalGraph()
        now = datetime.now()

        # Create a chain leading to an error
        n1 = Node(type=NodeType.AGENT_INPUT, agent_id="a", data="input", timestamp=now)
        n2 = Node(type=NodeType.DECISION, agent_id="a", data="decision",
                  timestamp=now + timedelta(milliseconds=50))
        n3 = Node(type=NodeType.TOOL_CALL, agent_id="a", data="call",
                  timestamp=now + timedelta(milliseconds=100))
        error = Node(
            type=NodeType.ERROR,
            agent_id="a",
            data={"error_type": "RuntimeError", "error_message": "Failed"},
            timestamp=now + timedelta(milliseconds=150),
        )

        for n in [n1, n2, n3, error]:
            graph.add_node(n)

        # Add edges
        from agenttrace.core.edge import Edge
        graph.add_edge(Edge(source_id=n1.id, target_id=n2.id, type=EdgeType.INPUT_OUTPUT))
        graph.add_edge(Edge(source_id=n2.id, target_id=n3.id, type=EdgeType.DATA_FLOW))
        graph.add_edge(Edge(source_id=n3.id, target_id=error.id, type=EdgeType.ERROR_PROPAGATION))

        suggestions = engine.suggest_likely_causes(graph, error.id)

        assert len(suggestions) >= 1
        # Decision nodes should score higher
        assert any(s["node"]["type"] == "decision" for s in suggestions)
