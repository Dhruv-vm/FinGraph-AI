from datetime import datetime, timezone

from src.data.graph.schema import GraphEdge, GraphNode, TemporalGraph
from src.data.graph.snapshot import build_temporal_snapshot


def test_graph_node_preserves_temporal_metadata():
    node = GraphNode(
        node_id="company:AAPL",
        node_type="company",
        properties={"ticker": "AAPL"},
        event_time=None,
        available_time="2026-01-01T00:00:00+00:00",
    )

    data = node.to_dict()

    assert data["node_id"] == "company:AAPL"
    assert data["node_type"] == "company"
    assert data["properties"]["ticker"] == "AAPL"
    assert data["available_time"] == "2026-01-01T00:00:00+00:00"


def test_graph_edge_preserves_temporal_metadata():
    edge = GraphEdge(
        edge_id="company:AAPL->SUPPLIES->company:TSM",
        source="company:AAPL",
        target="supplier:TSM",
        relationship="SUPPLIES",
        event_time="2026-01-01T00:00:00+00:00",
        available_time="2026-01-02T00:00:00+00:00",
        properties={"confidence": 0.95},
    )

    data = edge.to_dict()

    assert data["relationship"] == "SUPPLIES"
    assert data["event_time"] == "2026-01-01T00:00:00+00:00"
    assert data["available_time"] == "2026-01-02T00:00:00+00:00"
    assert data["properties"]["confidence"] == 0.95


def test_temporal_snapshot_excludes_future_nodes():
    graph = TemporalGraph(
        nodes=[
            GraphNode(
                node_id="company:AAPL",
                node_type="company",
            ),
            GraphNode(
                node_id="doc:past",
                node_type="Document",
                available_time="2026-01-01T00:00:00+00:00",
            ),
            GraphNode(
                node_id="doc:future",
                node_type="Document",
                available_time="2026-08-01T00:00:00+00:00",
            ),
        ]
    )

    snapshot = build_temporal_snapshot(
        graph,
        "2026-07-01T00:00:00+00:00",
    )

    node_ids = {node.node_id for node in snapshot.nodes}

    assert "company:AAPL" in node_ids
    assert "doc:past" in node_ids
    assert "doc:future" not in node_ids


def test_temporal_snapshot_excludes_future_edges():
    graph = TemporalGraph(
        nodes=[
            GraphNode(
                node_id="company:AAPL",
                node_type="company",
            ),
            GraphNode(
                node_id="company:TSM",
                node_type="company",
            ),
            GraphNode(
                node_id="company:AMD",
                node_type="company",
            ),
        ],
        edges=[
            GraphEdge(
                edge_id="edge:past",
                source="company:AAPL",
                target="company:TSM",
                relationship="SUPPLIES",
                available_time="2026-01-01T00:00:00+00:00",
            ),
            GraphEdge(
                edge_id="edge:future",
                source="company:AAPL",
                target="company:AMD",
                relationship="SUPPLIES",
                available_time="2026-08-01T00:00:00+00:00",
            ),
        ],
    )

    snapshot = build_temporal_snapshot(
        graph,
        "2026-07-01T00:00:00+00:00",
    )

    edge_ids = {edge.edge_id for edge in snapshot.edges}

    assert "edge:past" in edge_ids
    assert "edge:future" not in edge_ids


def test_temporal_snapshot_includes_exact_boundary():
    graph = TemporalGraph(
        nodes=[
            GraphNode(
                node_id="company:AAPL",
                node_type="company",
            ),
            GraphNode(
                node_id="company:TSM",
                node_type="company",
                available_time="2026-07-01T00:00:00+00:00",
            ),
        ],
        edges=[
            GraphEdge(
                edge_id="edge:boundary",
                source="company:AAPL",
                target="company:TSM",
                relationship="SUPPLIES",
                available_time="2026-07-01T00:00:00+00:00",
            )
        ],
    )

    reference_time = datetime(
        2026,
        7,
        1,
        tzinfo=timezone.utc,
    )

    snapshot = build_temporal_snapshot(
        graph,
        reference_time,
    )

    node_ids = {node.node_id for node in snapshot.nodes}
    edge_ids = {edge.edge_id for edge in snapshot.edges}

    assert "company:TSM" in node_ids
    assert "edge:boundary" in edge_ids


def test_temporal_snapshot_rejects_unknown_availability():
    graph = TemporalGraph(
        nodes=[
            GraphNode(
                node_id="company:AAPL",
                node_type="company",
            ),
            GraphNode(
                node_id="doc:unknown",
                node_type="Document",
                available_time=None,
            ),
        ]
    )

    snapshot = build_temporal_snapshot(
        graph,
        "2026-07-01T00:00:00+00:00",
    )

    node_ids = {node.node_id for node in snapshot.nodes}

    assert "company:AAPL" in node_ids
    assert "doc:unknown" not in node_ids


def test_snapshot_requires_both_edge_endpoints_to_be_available():
    graph = TemporalGraph(
        nodes=[
            GraphNode(
                node_id="company:AAPL",
                node_type="company",
            ),
            GraphNode(
                node_id="supplier:TSM",
                node_type="Supplier",
                available_time="2026-08-01T00:00:00+00:00",
            ),
        ],
        edges=[
            GraphEdge(
                edge_id="edge:future-target",
                source="company:AAPL",
                target="supplier:TSM",
                relationship="SUPPLIES",
                available_time="2026-01-01T00:00:00+00:00",
            )
        ],
    )

    snapshot = build_temporal_snapshot(
        graph,
        "2026-07-01T00:00:00+00:00",
    )

    assert snapshot.edges == []
