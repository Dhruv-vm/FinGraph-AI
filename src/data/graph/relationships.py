from __future__ import annotations

from typing import Any


# ---------------------------------------------------------------------------
# FinGraph AI semantic ontology
# ---------------------------------------------------------------------------

NODE_TYPES = {
    "Company",
    "Person",
    "Product",
    "Supplier",
    "Competitor",
    "Event",
    "Risk",
    "Country",
    "FinancialMetric",
    "Document",
    "NewsArticle",
}


SEMANTIC_RELATIONSHIPS = {
    "SUPPLIES",
    "MANUFACTURES",
    "DEPENDS_ON",
    "COMPETES_WITH",
    "PARTNERS_WITH",
    "INVESTED_IN",
    "ACQUIRED",
    "LOCATED_IN",
    "AFFECTED_BY",
    "CAUSED",
    "ANNOUNCED",
    "HAS_RISK",
    "MENTIONED_IN",
}


# ---------------------------------------------------------------------------
# Legacy event mappings
#
# Kept temporarily so the existing event ingestion pipeline continues to
# function while the semantic KG builder is introduced.
# ---------------------------------------------------------------------------

EVENT_RELATIONSHIPS = {
    "market": "HAS_MARKET_EVENT",
    "news": "HAS_NEWS",
    "sec_filing": "FILED",
    "macro": "AFFECTED_BY",
}


def is_valid_node_type(node_type: str) -> bool:
    """Return True when node_type belongs to the FinGraph ontology."""

    return node_type in NODE_TYPES


def is_valid_relationship(relationship: str) -> bool:
    """Return True when relationship belongs to the semantic ontology."""

    return relationship in SEMANTIC_RELATIONSHIPS


def relationship_for_event(event_type: str) -> str | None:
    """
    Return the legacy graph relationship associated with an event type.

    This compatibility function will be removed once the semantic extraction
    pipeline fully replaces the old event graph.
    """

    return EVENT_RELATIONSHIPS.get(event_type)


def build_relationship(
    company_id: str,
    event: dict[str, Any],
) -> dict[str, Any] | None:
    """
    Build a deterministic legacy company → event relationship.

    Kept for compatibility with the existing temporal event graph pipeline.
    New semantic relationships should be constructed from extracted
    entities and relations rather than event_type mappings.
    """

    event_type = event.get("event_type")
    relationship = relationship_for_event(event_type)

    if relationship is None:
        return None

    if not event.get("entity_resolved"):
        return None

    event_id = event.get("event_id")

    if not event_id:
        raise ValueError(
            "Unified event must contain event_id before "
            "building graph relationships."
        )

    return {
        "source": company_id,
        "target": event_id,
        "relationship": relationship,
        "event_time": event.get("event_time"),
        "available_time": event.get("available_time"),
    }


def build_semantic_relationship(
    source_id: str,
    target_id: str,
    relationship: str,
    *,
    event_time: str | None = None,
    available_time: str | None = None,
    properties: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Build a semantic KG relationship with temporal and provenance metadata.

    The caller is responsible for ensuring that source_id and target_id
    correspond to valid ontology nodes.
    """

    if not is_valid_relationship(relationship):
        raise ValueError(
            f"Unsupported FinGraph relationship: {relationship}"
        )

    return {
        "source": source_id,
        "target": target_id,
        "relationship": relationship,
        "event_time": event_time,
        "available_time": available_time,
        "properties": properties or {},
    }
