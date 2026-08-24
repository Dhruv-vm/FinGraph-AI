from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from src.data.graph.schema import GraphEdge, GraphNode, TemporalGraph


def parse_datetime(value: str | None) -> datetime | None:
    """Parse ISO/date strings into timezone-aware datetimes."""

    if not value:
        return None

    value = value.strip()

    if len(value) == 10:
        value = f"{value}T00:00:00+00:00"

    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)

    return dt


def is_available(
    available_time: str | None,
    as_of: datetime,
) -> bool:
    """Return whether information was available by the snapshot time."""

    if not available_time:
        return False

    available = parse_datetime(available_time)

    if available is None:
        return False

    return available <= as_of


def build_temporal_snapshot(
    graph: TemporalGraph | dict[str, Any],
    as_of: str | datetime,
) -> TemporalGraph:
    """
    Build a point-in-time graph snapshot.

    Only nodes and edges available by `as_of` are included.

    This prevents future information from leaking into
    historical analysis or prediction.
    """

    if isinstance(as_of, str):
        snapshot_time = parse_datetime(as_of)

        if snapshot_time is None:
            raise ValueError(f"Invalid snapshot time: {as_of}")
    else:
        snapshot_time = as_of

        if snapshot_time.tzinfo is None:
            snapshot_time = snapshot_time.replace(
                tzinfo=timezone.utc
            )

    if isinstance(graph, dict):
        nodes = [
            GraphNode(
                node_id=node["node_id"],
                node_type=node["node_type"],
                properties=node.get("properties", {}),
                event_time=node.get("event_time"),
                available_time=node.get("available_time"),
            )
            for node in graph["nodes"]
        ]

        edges = [
            GraphEdge(
                edge_id=edge["edge_id"],
                source=edge["source"],
                target=edge["target"],
                relationship=edge["relationship"],
                event_time=edge.get("event_time"),
                available_time=edge.get("available_time"),
                properties=edge.get("properties", {}),
            )
            for edge in graph["edges"]
        ]

        graph = TemporalGraph(
            nodes=nodes,
            edges=edges,
        )

    # Company/entity nodes are always available.
    snapshot_nodes = [
        node
        for node in graph.nodes
        if node.node_type == "company"
        or is_available(
            node.available_time,
            snapshot_time,
        )
    ]

    node_ids = {
        node.node_id
        for node in snapshot_nodes
    }

    snapshot_edges = [
        edge
        for edge in graph.edges
        if edge.source in node_ids
        and edge.target in node_ids
        and is_available(
            edge.available_time,
            snapshot_time,
        )
    ]

    return TemporalGraph(
        nodes=snapshot_nodes,
        edges=snapshot_edges,
        as_of=snapshot_time.isoformat(),
    )