from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ExtractedEntity:
    entity_id: str
    entity_type: str
    name: str
    canonical_name: str
    confidence: float = 1.0
    properties: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "entity_id": self.entity_id,
            "entity_type": self.entity_type,
            "name": self.name,
            "canonical_name": self.canonical_name,
            "confidence": self.confidence,
            "properties": self.properties,
        }


@dataclass
class ExtractedRelation:
    relation_id: str
    source_entity_id: str
    target_entity_id: str
    relationship: str
    confidence: float = 1.0
    event_time: str | None = None
    available_time: str | None = None
    properties: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "relation_id": self.relation_id,
            "source_entity_id": self.source_entity_id,
            "target_entity_id": self.target_entity_id,
            "relationship": self.relationship,
            "confidence": self.confidence,
            "event_time": self.event_time,
            "available_time": self.available_time,
            "properties": self.properties,
        }


@dataclass
class ExtractionResult:
    document_id: str
    entities: list[ExtractedEntity] = field(default_factory=list)
    relations: list[ExtractedRelation] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "document_id": self.document_id,
            "entities": [entity.to_dict() for entity in self.entities],
            "relations": [relation.to_dict() for relation in self.relations],
            "metadata": self.metadata,
        }
