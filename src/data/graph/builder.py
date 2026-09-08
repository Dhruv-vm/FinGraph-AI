from __future__ import annotations

import hashlib
import json
from typing import Any, Iterable

from src.data.entities.resolution import (
    build_entity_indexes,
    load_company_universe,
    resolve_event_company,
)
from src.data.graph.relationships import (
    build_relationship,
    build_semantic_relationship,
    is_valid_node_type,
)
from src.data.graph.schema import GraphEdge, GraphNode, TemporalGraph


def _stable_hash(value: Any) -> str:
    """Create a deterministic short hash."""

    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )

    return hashlib.sha256(
        payload.encode("utf-8")
    ).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Semantic KG identifiers
# ---------------------------------------------------------------------------


def build_semantic_node_id(
    node_type: str,
    canonical_name: str,
) -> str:
    """
    Build a deterministic identifier for a semantic KG node.

    Example:
        Company + Apple -> company:apple
        Supplier + TSMC -> supplier:tsmc
    """

    if not is_valid_node_type(node_type):
        raise ValueError(
            f"Unsupported FinGraph node type: {node_type}"
        )

    normalized = " ".join(
        str(canonical_name).strip().lower().split()
    )

    if not normalized:
        raise ValueError(
            "Semantic node requires a non-empty canonical name."
        )

    return f"{node_type.lower()}:{normalized}"


def build_semantic_node(
    node_type: str,
    canonical_name: str,
    *,
    properties: dict[str, Any] | None = None,
    event_time: str | None = None,
    available_time: str | None = None,
) -> GraphNode:
    """Build a semantic KG node with temporal metadata."""

    node_id = build_semantic_node_id(
        node_type,
        canonical_name,
    )

    node_properties = dict(properties or {})
    node_properties.setdefault(
        "canonical_name",
        canonical_name,
    )

    return GraphNode(
        node_id=node_id,
        node_type=node_type,
        properties=node_properties,
        event_time=event_time,
        available_time=available_time,
    )


def build_semantic_edge(
    source_id: str,
    target_id: str,
    relationship: str,
    *,
    event_time: str | None = None,
    available_time: str | None = None,
    properties: dict[str, Any] | None = None,
) -> GraphEdge:
    """Build a deterministic semantic KG edge."""

    edge_properties = dict(properties or {})

    edge_identity = {
        "source": source_id,
        "target": target_id,
        "relationship": relationship,
        "event_time": event_time,
        "available_time": available_time,
        "properties": edge_properties,
    }

    edge_id = (
        f"{source_id}"
        f"->{relationship}"
        f"->{target_id}"
        f":{_stable_hash(edge_identity)}"
    )

    relationship_data = build_semantic_relationship(
        source_id=source_id,
        target_id=target_id,
        relationship=relationship,
        event_time=event_time,
        available_time=available_time,
        properties=edge_properties,
    )

    return GraphEdge(
        edge_id=edge_id,
        source=relationship_data["source"],
        target=relationship_data["target"],
        relationship=relationship_data["relationship"],
        event_time=relationship_data["event_time"],
        available_time=relationship_data["available_time"],
        properties=relationship_data["properties"],
    )


# ---------------------------------------------------------------------------
# Semantic graph builder
# ---------------------------------------------------------------------------


def build_semantic_graph(
    entities: Iterable[dict[str, Any]],
    relations: Iterable[dict[str, Any]],
) -> TemporalGraph:
    """
    Build a temporal semantic knowledge graph from extracted entities
    and relations.

    Expected entity format:

        {
            "node_type": "Supplier",
            "canonical_name": "TSMC",
            "properties": {...},
            "event_time": "...",
            "available_time": "..."
        }

    Expected relation format:

        {
            "source": "supplier:tsmc",
            "target": "company:nvidia",
            "relationship": "SUPPLIES",
            "event_time": "...",
            "available_time": "...",
            "properties": {
                "source_document_id": "...",
                "evidence_chunk_id": "...",
                "confidence": 0.95
            }
        }

    Relations reference already-created canonical node IDs.
    """

    graph = TemporalGraph()

    node_ids: set[str] = set()
    edge_ids: set[str] = set()

    # ---------------------------------------------------------------
    # Nodes
    # ---------------------------------------------------------------

    for entity in entities:
        node_type = entity["node_type"]
        canonical_name = entity["canonical_name"]

        node = build_semantic_node(
            node_type=node_type,
            canonical_name=canonical_name,
            properties=entity.get("properties"),
            event_time=entity.get("event_time"),
            available_time=entity.get("available_time"),
        )

        if node.node_id in node_ids:
            continue

        graph.nodes.append(node)
        node_ids.add(node.node_id)

    # ---------------------------------------------------------------
    # Relationships
    # ---------------------------------------------------------------

    for relation in relations:
        source_id = relation["source"]
        target_id = relation["target"]

        # Never create dangling edges.
        if source_id not in node_ids:
            continue

        if target_id not in node_ids:
            continue

        edge = build_semantic_edge(
            source_id=source_id,
            target_id=target_id,
            relationship=relation["relationship"],
            event_time=relation.get("event_time"),
            available_time=relation.get("available_time"),
            properties=relation.get("properties"),
        )

        if edge.edge_id in edge_ids:
            continue

        graph.edges.append(edge)
        edge_ids.add(edge.edge_id)

    return graph


# ---------------------------------------------------------------------------
# Existing temporal event graph
#
# Kept intact for compatibility while the semantic KG pipeline is introduced.
# ---------------------------------------------------------------------------


def build_event_id(event: dict[str, Any]) -> str:
    """Build a deterministic identifier for a unified event."""

    event_type = event.get("event_type", "unknown")
    ticker = event.get("ticker") or event.get("data", {}).get("ticker", "")

    data = event.get("data", {})

    if event_type == "sec_filing":
        accession = data.get("accession_number")

        if accession:
            return f"sec:{ticker}:{accession}"

    if event_type == "market":
        date = event.get("event_time") or data.get("date")

        if date:
            return f"market:{ticker}:{date}"

    if event_type == "macro":
        series_id = data.get("series_id", "unknown")
        date = event.get("event_time") or data.get("date")

        if date:
            return f"macro:{series_id}:{date}"

    if event_type == "news":
        identity = {
            "ticker": ticker,
            "title": data.get("title"),
            "url": data.get("url"),
            "published_at": data.get("published_at"),
        }

        return f"news:{ticker}:{_stable_hash(identity)}"

    return f"{event_type}:{ticker}:{_stable_hash(event)}"


def build_event_node(event: dict[str, Any]) -> GraphNode:
    """Convert a unified event into a graph node."""

    event_id = build_event_id(event)
    event_type = event["event_type"]
    data = event.get("data", {})

    return GraphNode(
        node_id=event_id,
        node_type=event_type,
        properties=data,
        event_time=event.get("event_time"),
        available_time=event.get("available_time"),
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
    Build the existing temporal event graph.

    This compatibility path remains available while FinGraph AI migrates
    to the semantic KG builder above.
    """

    if companies is None:
        companies = load_company_universe()

    indexes = build_entity_indexes(companies)

    graph = TemporalGraph()

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
                properties=relationship.get("properties", {}),
            )
        )

        seen_edge_ids.add(edge_id)

    return graph
