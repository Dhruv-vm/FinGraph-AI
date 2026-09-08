from __future__ import annotations

import hashlib
from datetime import datetime
from typing import Any, Iterable

from .schema import ExtractedRelation
from src.data.graph.relationships import is_valid_relationship


def _parse_datetime(value: str | None) -> datetime | None:
    if value is None:
        return None

    text = str(value).strip()

    if not text:
        return None

    if text.endswith("Z"):
        text = text[:-1] + "+00:00"

    try:
        return datetime.fromisoformat(text)
    except ValueError:
        pass

    # Accept common date-only formats returned by LLM extraction.
    for fmt in ("%B %d, %Y", "%b %d, %Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue

    raise ValueError(f"Invalid datetime: {value}")


def validate_temporal_metadata(
    event_time: str | None,
    available_time: str | None,
) -> None:
    """Validate temporal metadata when both timestamps are available."""
    event = _parse_datetime(event_time)
    available = _parse_datetime(available_time)

    if event is not None and available is not None:
        if available < event:
            raise ValueError(
                "available_time cannot be earlier than event_time."
            )


def build_relation_id(
    source_entity_id: str,
    relationship: str,
    target_entity_id: str,
) -> str:
    """Build a deterministic relationship identifier."""
    value = (
        f"{source_entity_id}:"
        f"{relationship.strip().upper()}:"
        f"{target_entity_id}"
    )

    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]

    return f"relation:{digest}"


def build_relation(
    source_entity_id: str,
    target_entity_id: str,
    relationship: str,
    confidence: float = 1.0,
    event_time: str | None = None,
    available_time: str | None = None,
    properties: dict[str, Any] | None = None,
) -> ExtractedRelation:
    """Build a validated extracted relation."""
    source = str(source_entity_id).strip()
    target = str(target_entity_id).strip()
    relationship = str(relationship).strip().upper()

    if not source:
        raise ValueError("Source entity ID cannot be empty.")

    if not target:
        raise ValueError("Target entity ID cannot be empty.")

    if source == target:
        raise ValueError("Self-referential relations are not allowed.")

    if not is_valid_relationship(relationship):
        raise ValueError(
            f"Unsupported relationship type: {relationship}"
        )

    confidence = float(confidence)

    if not 0.0 <= confidence <= 1.0:
        raise ValueError("Relation confidence must be between 0 and 1.")

    validate_temporal_metadata(event_time, available_time)

    relation_id = build_relation_id(
        source_entity_id=source,
        relationship=relationship,
        target_entity_id=target,
    )

    return ExtractedRelation(
        relation_id=relation_id,
        source_entity_id=source,
        target_entity_id=target,
        relationship=relationship,
        confidence=confidence,
        event_time=event_time,
        available_time=available_time,
        properties=dict(properties or {}),
    )


def deduplicate_relations(
    relations: Iterable[ExtractedRelation],
) -> list[ExtractedRelation]:
    """Deduplicate relations using their deterministic relation IDs."""
    unique: dict[str, ExtractedRelation] = {}

    for relation in relations:
        existing = unique.get(relation.relation_id)

        if existing is None:
            unique[relation.relation_id] = relation
            continue

        # Prefer the higher-confidence extraction.
        if relation.confidence > existing.confidence:
            primary = relation
            secondary = existing
        else:
            primary = existing
            secondary = relation

        merged_properties = {
            **secondary.properties,
            **primary.properties,
        }

        unique[relation.relation_id] = ExtractedRelation(
            relation_id=primary.relation_id,
            source_entity_id=primary.source_entity_id,
            target_entity_id=primary.target_entity_id,
            relationship=primary.relationship,
            confidence=max(
                primary.confidence,
                secondary.confidence,
            ),
            event_time=primary.event_time or secondary.event_time,
            available_time=(
                primary.available_time
                or secondary.available_time
            ),
            properties=merged_properties,
        )

    return list(unique.values())


def relations_from_dicts(
    records: Iterable[dict[str, Any]],
) -> list[ExtractedRelation]:
    """Convert schema-compatible dictionaries into relations."""
    relations: list[ExtractedRelation] = []

    for record in records:
        relations.append(
            build_relation(
                source_entity_id=record["source_entity_id"],
                target_entity_id=record["target_entity_id"],
                relationship=record["relationship"],
                confidence=record.get("confidence", 1.0),
                event_time=record.get("event_time"),
                available_time=record.get("available_time"),
                properties=record.get("properties"),
            )
        )

    return deduplicate_relations(relations)
