from __future__ import annotations

from collections import Counter
from typing import Any


def _node_dict(node: Any) -> dict[str, Any]:
    """Convert a GraphNode or dictionary into a dictionary."""

    if isinstance(node, dict):
        return node

    return node.to_dict()


def _edge_dict(edge: Any) -> dict[str, Any]:
    """Convert a GraphEdge or dictionary into a dictionary."""

    if isinstance(edge, dict):
        return edge

    return edge.to_dict()


def company_event_history(
    snapshot: Any,
    entity_id: str,
) -> list[dict[str, Any]]:
    """
    Return all events connected to a company in a temporal snapshot.

    Only events already present in the supplied snapshot are considered.
    """

    if isinstance(snapshot, dict):
        nodes = snapshot.get("nodes", [])
        edges = snapshot.get("edges", [])
    else:
        nodes = snapshot.nodes
        edges = snapshot.edges

    related_node_ids: set[str] = set()

    for edge in edges:
        edge_data = _edge_dict(edge)

        if edge_data["source"] == entity_id:
            related_node_ids.add(edge_data["target"])

        elif edge_data["target"] == entity_id:
            related_node_ids.add(edge_data["source"])

    history = []

    for node in nodes:
        node_data = _node_dict(node)

        if node_data["node_id"] in related_node_ids:
            history.append(node_data)

    history.sort(
        key=lambda x: (
            x.get("event_time") or "",
            x.get("available_time") or "",
        )
    )

    return history


def company_event_counts(
    snapshot: Any,
    entity_id: str,
) -> dict[str, int]:
    """Count events connected to a company by event type."""

    history = company_event_history(
        snapshot,
        entity_id,
    )

    counts = Counter(
        node.get("node_type")
        for node in history
    )

    return {
        "market": counts.get("market", 0),
        "news": counts.get("news", 0),
        "sec_filing": counts.get("sec_filing", 0),
        "macro": counts.get("macro", 0),
        "total": len(history),
    }


def company_activity_summary(
    snapshot: Any,
    entity_id: str,
) -> dict[str, Any]:
    """
    Build a deterministic activity summary for a company.

    All calculations operate only on the supplied temporal snapshot.
    """

    if isinstance(snapshot, dict):
        nodes = snapshot.get("nodes", [])
    else:
        nodes = snapshot.nodes

    company = None

    for node in nodes:
        node_data = _node_dict(node)

        if node_data["node_id"] == entity_id:
            company = node_data
            break

    if company is None:
        raise ValueError(
            f"Company entity not found: {entity_id}"
        )

    counts = company_event_counts(
        snapshot,
        entity_id,
    )

    history = company_event_history(
        snapshot,
        entity_id,
    )

    event_dates = [
        node.get("event_time")
        for node in history
        if node.get("event_time")
    ]

    available_dates = [
        node.get("available_time")
        for node in history
        if node.get("available_time")
    ]

    return {
        "entity_id": entity_id,
        "ticker": company.get("properties", {}).get("ticker"),
        "company": company.get("properties", {}).get("company"),
        "sector": company.get("properties", {}).get("sector"),
        "as_of": _get_snapshot_as_of(snapshot),
        "total_events": counts["total"],
        "market_events": counts["market"],
        "news_events": counts["news"],
        "sec_filings": counts["sec_filing"],
        "macro_events": counts["macro"],
        "first_event_time": min(event_dates) if event_dates else None,
        "last_event_time": max(event_dates) if event_dates else None,
        "first_available_time": (
            min(available_dates)
            if available_dates
            else None
        ),
        "last_available_time": (
            max(available_dates)
            if available_dates
            else None
        ),
    }


def _get_snapshot_as_of(snapshot: Any) -> str | None:
    """Return the snapshot's as-of date when available."""

    if isinstance(snapshot, dict):
        return snapshot.get("as_of")

    return getattr(snapshot, "as_of", None)
