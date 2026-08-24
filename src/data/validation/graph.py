from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


def parse_datetime(value: str | None) -> datetime | None:
    """Parse dates/datetimes into timezone-aware UTC datetimes."""

    if not value:
        return None

    try:
        parsed = datetime.fromisoformat(
            value.replace("Z", "+00:00")
        )

        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)

        return parsed.astimezone(timezone.utc)

    except ValueError:
        return None


def validate_graph_temporality(
    graph: dict[str, Any],
) -> dict[str, Any]:
    """
    Validate temporal consistency of graph nodes and edges.

    Core rule:
        available_time must never precede event_time.

    This prevents future information from becoming
    available before the event actually occurred.
    """

    node_errors = []
    edge_errors = []

    for node in graph.get("nodes", []):
        event_time = parse_datetime(node.get("event_time"))
        available_time = parse_datetime(node.get("available_time"))

        if (
            event_time is not None
            and available_time is not None
            and available_time < event_time
        ):
            node_errors.append(
                {
                    "node_id": node.get("node_id"),
                    "error": "available_time_before_event_time",
                }
            )

    for edge in graph.get("edges", []):
        event_time = parse_datetime(edge.get("event_time"))
        available_time = parse_datetime(edge.get("available_time"))

        if (
            event_time is not None
            and available_time is not None
            and available_time < event_time
        ):
            edge_errors.append(
                {
                    "edge_id": edge.get("edge_id"),
                    "error": "available_time_before_event_time",
                }
            )

    return {
        "nodes": len(graph.get("nodes", [])),
        "edges": len(graph.get("edges", [])),
        "node_errors": len(node_errors),
        "edge_errors": len(edge_errors),
        "temporal_valid": not node_errors and not edge_errors,
        "node_error_details": node_errors,
        "edge_error_details": edge_errors,
    }