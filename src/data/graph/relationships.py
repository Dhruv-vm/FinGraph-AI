from __future__ import annotations

from typing import Any


RELATIONSHIPS = {
    "market": "HAS_MARKET_EVENT",
    "news": "HAS_NEWS",
    "sec_filing": "FILED",
    "macro": "AFFECTED_BY",
}


def relationship_for_event(event_type: str) -> str | None:
    """Return the graph relationship associated with an event type."""

    return RELATIONSHIPS.get(event_type)


def build_relationship(
    company_id: str,
    event: dict[str, Any],
) -> dict[str, Any] | None:
    """
    Build a deterministic company → event relationship.

    Returns None when the event type is unsupported or
    the event has not been resolved to a company.
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