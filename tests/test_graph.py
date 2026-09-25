"""Tests for the causal graph implementation."""

import pytest
from datetime import datetime, timedelta

from agenttrace.core.graph import CausalGraph
from agenttrace.core.node import Node, NodeType
from agenttrace.core.edge import Edge, EdgeType


class TestNode:
    """Tests for Node class."""

    def test_create_node(self):
        node = Node(
            type=NodeType.AGENT_INPUT,
            agent_id="test_agent",
            data={"message": "hello"},
        )
        assert node.type == NodeType.AGENT_INPUT
        assert node.agent_id == "test_agent"
        assert node.data == {"message": "hello"}
        assert node.id is not None

    def test_node_serialization(self):
        node = Node(
            type=NodeType.TOOL_CALL,
            agent_id="test",
            data={"tool": "search", "query": "test"},
            metadata={"source": "user"},
        )
        d = node.to_dict()
        restored = Node.from_dict(d)

        assert restored.type == node.type
        assert restored.agent_id == node.agent_id
        assert restored.data == node.data
        assert restored.metadata == node.metadata

    def test_node_equality(self):
        node1 = Node(type=NodeType.AGENT_INPUT, agent_id="a", data={})
        node2 = Node(type=NodeType.AGENT_INPUT, agent_id="a", data={})

        # Different IDs means not equal
        assert node1 != node2

        # Same ID means equal
        node2.id = node1.id
        assert node1 == node2


class TestEdge:
    """Tests for Edge class."""

    def test_create_edge(self):
        edge = Edge(
            source_id="node1",
            target_id="node2",
            type=EdgeType.INPUT_OUTPUT,
        )
        assert edge.source_id == "node1"
        assert edge.target_id == "node2"
        assert edge.confidence == 1.0

    def test_edge_confidence_validation(self):
        with pytest.raises(ValueError):
            Edge(
                source_id="a",
                target_id="b",
                type=EdgeType.DATA_FLOW,
                confidence=1.5,  # Invalid
            )

    def test_edge_serialization(self):
        edge = Edge(
            source_id="src",
            target_id="tgt",
            type=EdgeType.TRIGGER_RESPONSE,
            confidence=0.8,
            metadata={"inferred": True},
        )
        d = edge.to_dict()
        restored = Edge.from_dict(d)

        assert restored.source_id == edge.source_id
        assert restored.target_id == edge.target_id
        assert restored.type == edge.type
        assert restored.confidence == edge.confidence


class TestCausalGraph:
    """Tests for CausalGraph class."""

    @pytest.fixture
    def graph(self):
        return CausalGraph(run_id="test_run")

    @pytest.fixture
    def populated_graph(self, graph):
        """Create a graph with nodes and edges."""
        # Create a simple chain: input -> processing -> output
        n1 = Node(type=NodeType.AGENT_INPUT, agent_id="agent1", data="input")
        n2 = Node(type=NodeType.AGENT_OUTPUT, agent_id="agent1", data="processed", parent_ids=[n1.id])
        n3 = Node(type=NodeType.TOOL_CALL, agent_id="agent1", data="tool_call", parent_ids=[n2.id])
        n4 = Node(type=NodeType.TOOL_RESULT, agent_id="agent1", data="result", parent_ids=[n3.id])
        n5 = Node(type=NodeType.AGENT_OUTPUT, agent_id="agent1", data="final", parent_ids=[n4.id])

        for node in [n1, n2, n3, n4, n5]:
            graph.add_node(node)

        return graph

    def test_add_node(self, graph):
        node = Node(type=NodeType.AGENT_INPUT, agent_id="test", data="data")
        added = graph.add_node(node)

        assert added.run_id == graph.run_id
        assert node.id in graph
        assert len(graph) == 1

    def test_add_edge(self, graph):
        n1 = Node(type=NodeType.AGENT_INPUT, agent_id="a", data="1")
        n2 = Node(type=NodeType.AGENT_OUTPUT, agent_id="a", data="2")
        graph.add_node(n1)
        graph.add_node(n2)

        edge = Edge(source_id=n1.id, target_id=n2.id, type=EdgeType.INPUT_OUTPUT)
        graph.add_edge(edge)

        assert graph.edge_count == 1

    def test_add_edge_invalid_node(self, graph):
        n1 = Node(type=NodeType.AGENT_INPUT, agent_id="a", data="1")
        graph.add_node(n1)

        edge = Edge(source_id=n1.id, target_id="nonexistent", type=EdgeType.DATA_FLOW)

        with pytest.raises(ValueError):
            graph.add_edge(edge)

    def test_trace_forward(self, populated_graph):
        # Get the first node
        nodes = list(populated_graph)
        first_node = min(nodes, key=lambda n: n.timestamp)

        affected = populated_graph.trace_forward(first_node.id)

        # Should find all subsequent nodes
        assert len(affected) == 4  # All except the first

    def test_trace_backward(self, populated_graph):
        # Get the last node
        nodes = list(populated_graph)
        last_node = max(nodes, key=lambda n: n.timestamp)

        causes = populated_graph.trace_backward(last_node.id)

        # Should find all preceding nodes
        assert len(causes) == 4  # All except the last

    def test_trace_forward_with_depth(self, populated_graph):
        nodes = list(populated_graph)
        first_node = min(nodes, key=lambda n: n.timestamp)

        affected = populated_graph.trace_forward(first_node.id, max_depth=2)

        # Should only find immediate successors within depth
        assert len(affected) <= 2

    def test_find_root_causes(self, populated_graph):
        nodes = list(populated_graph)
        last_node = max(nodes, key=lambda n: n.timestamp)

        roots = populated_graph.find_root_causes(last_node.id)

        # Should find the initial input node
        assert len(roots) == 1
        assert roots[0].type == NodeType.AGENT_INPUT

    def test_find_path(self, populated_graph):
        nodes = list(populated_graph)
        first_node = min(nodes, key=lambda n: n.timestamp)
        last_node = max(nodes, key=lambda n: n.timestamp)

        path = populated_graph.find_path(first_node.id, last_node.id)

        assert path is not None
        assert path[0].id == first_node.id
        assert path[-1].id == last_node.id
        assert len(path) == 5

    def test_no_path(self, graph):
        n1 = Node(type=NodeType.AGENT_INPUT, agent_id="a", data="1")
        n2 = Node(type=NodeType.AGENT_OUTPUT, agent_id="b", data="2")
        graph.add_node(n1)
        graph.add_node(n2)

        path = graph.find_path(n1.id, n2.id)
        assert path is None

    def test_get_nodes_by_agent(self, graph):
        n1 = Node(type=NodeType.AGENT_INPUT, agent_id="agent_a", data="1")
        n2 = Node(type=NodeType.AGENT_OUTPUT, agent_id="agent_a", data="2")
        n3 = Node(type=NodeType.AGENT_INPUT, agent_id="agent_b", data="3")

        graph.add_node(n1)
        graph.add_node(n2)
        graph.add_node(n3)

        agent_a_nodes = graph.get_nodes_by_agent("agent_a")
        assert len(agent_a_nodes) == 2

        agent_b_nodes = graph.get_nodes_by_agent("agent_b")
        assert len(agent_b_nodes) == 1

    def test_get_nodes_by_type(self, populated_graph):
        inputs = populated_graph.get_nodes_by_type(NodeType.AGENT_INPUT)
        outputs = populated_graph.get_nodes_by_type(NodeType.AGENT_OUTPUT)

        assert len(inputs) == 1
        assert len(outputs) == 2

    def test_serialization(self, populated_graph):
        json_str = populated_graph.to_json()
        restored = CausalGraph.from_json(json_str)

        assert restored.run_id == populated_graph.run_id
        assert restored.node_count == populated_graph.node_count
        assert restored.edge_count == populated_graph.edge_count

    def test_counterfactual(self, populated_graph):
        nodes = list(populated_graph)
        target = nodes[2]  # Middle node

        cf = populated_graph.create_counterfactual(target.id, "alternative_data")

        # Should have same structure
        assert cf.node_count == populated_graph.node_count
        assert cf.edge_count == populated_graph.edge_count

        # But different data for target node
        cf_node = cf.get_node(target.id)
        assert cf_node.data == "alternative_data"
        assert cf_node.metadata.get("counterfactual") is True

    def test_compare_graphs(self, populated_graph):
        cf = populated_graph.create_counterfactual(
            list(populated_graph)[0].id,
            "modified",
        )

        comparison = populated_graph.compare_with(cf)

        assert comparison["common_nodes"] == populated_graph.node_count
        assert len(comparison["different_data"]) == 1

    def test_statistics(self, populated_graph):
        stats = populated_graph.get_statistics()

        assert stats["node_count"] == 5
        assert stats["edge_count"] == 4
        assert "agent1" in stats["agents"]
        assert stats["is_dag"] is True
