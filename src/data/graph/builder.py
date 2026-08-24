from __future__ import annotations
from src.data.graph.relationships import relationship_for_event
import hashlib
import json
from typing import Any

from src.data.entities.resolution import (
    build_entity_indexes,
    load_company_universe,
    resolve_event_company,
)
from src.data.graph.relationships import build_relationship
from src.data.graph.schema import (
    GraphEdge,
    GraphNode,
    TemporalGraph,
)


def _stable_hash(value: dict[str, Any]) -> str:
    """Create a deterministic short hash for an event."""

    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )

    return hashlib.sha256(
        payload.encode("utf-8")
    ).hexdigest()[:16]

def build_relationship(
    company_id: str,
    event: dict[str, Any],
) -> dict[str, Any] | None:
    """Build a company-to-event relationship."""

    event_type = event.get("event_type")

    relationship_map = {
        "market": "HAS_MARKET_EVENT",
        "news": "HAS_NEWS",
        "sec_filing": "FILED",
        "macro": "AFFECTED_BY",
    }

    relationship = relationship_map.get(event_type)

    if relationship is None:
        return None

    event_id = build_event_id(event)

    return {
        "edge_id": (
            f"{company_id}->{relationship}->{event_id}"
        ),
        "source": company_id,
        "target": event_id,
        "relationship": relationship,
        "event_time": event.get("event_time"),
        "available_time": event.get("available_time"),
        "properties": {},
    }

def build_event_id(event: dict[str, Any]) -> str:
    """Build a deterministic identifier for a unified event."""

    event_type = event.get("event_type", "unknown")
    ticker = event.get("ticker") or event.get("data", {}).get("ticker", "")

    data = event.get("data", {})

    # SEC filings have a naturally stable identifier.
    if event_type == "sec_filing":
        accession = data.get("accession_number")

        if accession:
            return f"sec:{ticker}:{accession}"

    # Market data is naturally identified by ticker + date.
    if event_type == "market":
        date = event.get("event_time") or data.get("date")

        if date:
            return f"market:{ticker}:{date}"

    # Macro data is identified by series + date.
    if event_type == "macro":
        series_id = data.get("series_id", "unknown")
        date = event.get("event_time") or data.get("date")

        if date:
            return f"macro:{series_id}:{date}"

    # News needs a stable hash because URLs/titles can vary.
    if event_type == "news":
        identity = {
            "ticker": ticker,
            "title": data.get("title"),
            "url": data.get("url"),
            "published_at": data.get("published_at"),
        }

        return f"news:{ticker}:{_stable_hash(identity)}"

    # Safe fallback for future event types.
    return f"{event_type}:{ticker}:{_stable_hash(event)}"


def build_event_node(event: dict[str, Any]) -> GraphNode:
    """Convert a unified event into a graph node."""

    event_id = build_event_id(event)
    event_type = event["event_type"]
    data = event.get("data", {})

    event_time = event.get("event_time")
    available_time = event.get("available_time")

    return GraphNode(
        node_id=event_id,
        node_type=event_type,
        properties=data,
        event_time=event_time,
        available_time=available_time,
    )

def build_company_node(
    company: dict[str, str],
) -> GraphNode:
    """Convert a company universe record into a graph node."""

    return GraphNode(
        node_id=company["entity_id"],
        node_type="company",
        properties={
            "ticker": company["ticker"],
            "company": company["company"],
            "sector": company["sector"],
            "cik": company["cik"],
        },
    )


def build_graph(
    events: list[dict[str, Any]],
    companies: list[dict[str, str]] | None = None,
) -> TemporalGraph:
    """
    Build a temporal knowledge graph from unified events.

    Every resolved event receives:
        company → event

    relationship with event and availability timestamps preserved.
    """

    if companies is None:
        companies = load_company_universe()

    indexes = build_entity_indexes(companies)

    graph = TemporalGraph()

    # Add canonical company nodes first.
    company_nodes: dict[str, GraphNode] = {}

    for company in companies:
        node = build_company_node(company)

        company_nodes[node.node_id] = node
        graph.nodes.append(node)

    seen_event_ids: set[str] = set()
    seen_edge_ids: set[str] = set()

    for event in events:
        resolved_event = resolve_event_company(
            event,
            indexes,
        )

        event_id = build_event_id(resolved_event)

        # Prevent duplicate event nodes.
        if event_id not in seen_event_ids:
            graph.nodes.append(
                build_event_node(resolved_event)
            )
            seen_event_ids.add(event_id)

        if not resolved_event.get("entity_resolved"):
            continue

        company_id = resolved_event.get("entity_id")

        if company_id not in company_nodes:
            continue

        relationship = build_relationship(
            company_id,
            {
                **resolved_event,
                "event_id": event_id,
            },
        )

        if relationship is None:
            continue

        edge_id = (
            f"{relationship['source']}"
            f"->{relationship['relationship']}"
            f"->{relationship['target']}"
        )

        if edge_id in seen_edge_ids:
            continue

        graph.edges.append(
            GraphEdge(
                edge_id=edge_id,
                source=relationship["source"],
                target=relationship["target"],
                relationship=relationship["relationship"],
                event_time=relationship["event_time"],
                available_time=relationship["available_time"],
            )
        )

        seen_edge_ids.add(edge_id)

    return graph