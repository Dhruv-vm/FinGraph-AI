from __future__ import annotations

import hashlib
import re
from typing import Any, Iterable

from .schema import ExtractedEntity


ENTITY_TYPES = {
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


def normalize_entity_name(name: str) -> str:
    """Normalize an entity name for deterministic matching."""
    value = re.sub(r"\s+", " ", str(name).strip())
    return value.casefold()


def build_entity_id(entity_type: str, canonical_name: str) -> str:
    """Build a deterministic entity identifier."""
    normalized_type = str(entity_type).strip().lower()
    normalized_name = normalize_entity_name(canonical_name)

    digest = hashlib.sha256(
        f"{normalized_type}:{normalized_name}".encode("utf-8")
    ).hexdigest()[:16]

    return f"{normalized_type}:{digest}"


def validate_entity_type(entity_type: str) -> bool:
    """Return whether an entity type belongs to the FinGraph schema."""
    return str(entity_type).strip() in ENTITY_TYPES


def build_entity(
    entity_type: str,
    name: str,
    canonical_name: str | None = None,
    confidence: float = 1.0,
    properties: dict[str, Any] | None = None,
) -> ExtractedEntity:
    """Build a validated, deterministic extracted entity."""
    entity_type = str(entity_type).strip()
    name = str(name).strip()

    if not name:
        raise ValueError("Entity name cannot be empty.")

    if not validate_entity_type(entity_type):
        raise ValueError(f"Unsupported entity type: {entity_type}")

    canonical = (
        str(canonical_name).strip()
        if canonical_name is not None
        else name
    )

    if not canonical:
        raise ValueError("Canonical entity name cannot be empty.")

    confidence = float(confidence)

    if not 0.0 <= confidence <= 1.0:
        raise ValueError("Entity confidence must be between 0 and 1.")

    entity_id = build_entity_id(entity_type, canonical)

    return ExtractedEntity(
        entity_id=entity_id,
        entity_type=entity_type,
        name=name,
        canonical_name=canonical,
        confidence=confidence,
        properties=dict(properties or {}),
    )


def deduplicate_entities(
    entities: Iterable[ExtractedEntity],
) -> list[ExtractedEntity]:
    """Deduplicate entities by deterministic entity ID.

    When duplicates occur, retain the highest-confidence instance and
    merge properties from the other instances.
    """
    unique: dict[str, ExtractedEntity] = {}

    for entity in entities:
        existing = unique.get(entity.entity_id)

        if existing is None:
            unique[entity.entity_id] = entity
            continue

        if entity.confidence > existing.confidence:
            primary = entity
            secondary = existing
        else:
            primary = existing
            secondary = entity

        merged_properties = {
            **secondary.properties,
            **primary.properties,
        }

        unique[entity.entity_id] = ExtractedEntity(
            entity_id=primary.entity_id,
            entity_type=primary.entity_type,
            name=primary.name,
            canonical_name=primary.canonical_name,
            confidence=max(primary.confidence, secondary.confidence),
            properties=merged_properties,
        )

    return list(unique.values())


def entities_from_dicts(
    records: Iterable[dict[str, Any]],
) -> list[ExtractedEntity]:
    """Convert schema-compatible dictionaries into extracted entities."""
    entities: list[ExtractedEntity] = []

    for record in records:
        entities.append(
            build_entity(
                entity_type=record["entity_type"],
                name=record["name"],
                canonical_name=record.get("canonical_name"),
                confidence=record.get("confidence", 1.0),
                properties=record.get("properties"),
            )
        )

    return deduplicate_entities(entities)
