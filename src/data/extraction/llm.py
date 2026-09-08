from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Protocol

from .schema import ExtractionResult
from .entities import entities_from_dicts
from .relations import relations_from_dicts


SYSTEM_PROMPT = """
You are a financial knowledge graph extraction system.

Extract only information explicitly supported by the supplied document text.

Allowed entity types:
Company, Person, Product, Supplier, Competitor, Event, Risk,
Country, FinancialMetric, Document, NewsArticle.

Allowed relationships:
SUPPLIES, MANUFACTURES, DEPENDS_ON, COMPETES_WITH, PARTNERS_WITH,
INVESTED_IN, ACQUIRED, LOCATED_IN, AFFECTED_BY, CAUSED, ANNOUNCED,
HAS_RISK, MENTIONED_IN.

Do not infer unsupported facts.
Do not invent entities or relationships.
Return valid JSON only.
""".strip()


@dataclass
class ExtractionRequest:
    document_id: str
    text: str
    available_time: str | None = None
    publication_date: str | None = None
    metadata: dict[str, Any] | None = None


class LLMClient(Protocol):
    """Provider-neutral interface for structured extraction."""

    def extract(self, request: ExtractionRequest) -> str:
        ...


def parse_extraction_response(
    response: str,
    request: ExtractionRequest,
) -> ExtractionResult:
    """Parse and validate an LLM JSON response."""
    if not response or not response.strip():
        raise ValueError("LLM response is empty.")

    try:
        payload = json.loads(response)
    except json.JSONDecodeError as exc:
        raise ValueError("LLM response is not valid JSON.") from exc

    if not isinstance(payload, dict):
        raise ValueError("LLM response must be a JSON object.")

    raw_entities = payload.get("entities", [])
    raw_relations = payload.get("relations", [])

    if not isinstance(raw_entities, list):
        raise ValueError("'entities' must be a JSON list.")

    if not isinstance(raw_relations, list):
        raise ValueError("'relations' must be a JSON list.")

    entities = entities_from_dicts(raw_entities)
    relations = relations_from_dicts(raw_relations)

    # Ensure every relation points to an entity extracted from
    # this document. This prevents dangling graph relationships.
    entity_ids = {entity.entity_id for entity in entities}

    for relation in relations:
        if relation.source_entity_id not in entity_ids:
            raise ValueError(
                f"Relation references unknown source entity: "
                f"{relation.source_entity_id}"
            )

        if relation.target_entity_id not in entity_ids:
            raise ValueError(
                f"Relation references unknown target entity: "
                f"{relation.target_entity_id}"
            )

    # Apply document-level temporal metadata when the model did not
    # explicitly provide relation availability.
    normalized_relations = []

    for relation in relations:
        if relation.available_time is None:
            relation.available_time = request.available_time

        normalized_relations.append(relation)

    metadata = dict(request.metadata or {})
    metadata["publication_date"] = request.publication_date
    metadata["available_time"] = request.available_time

    return ExtractionResult(
        document_id=request.document_id,
        entities=entities,
        relations=normalized_relations,
        metadata=metadata,
    )


class MockLLMClient:
    """Deterministic client used for tests and pipeline development.

    This deliberately performs no external API calls.
    """

    def __init__(self, response: str | None = None) -> None:
        self.response = response or '{"entities": [], "relations": []}'

    def extract(self, request: ExtractionRequest) -> str:
        return self.response


def extract_document(
    client: LLMClient,
    request: ExtractionRequest,
) -> ExtractionResult:
    """Run extraction through any compatible LLM client."""
    response = client.extract(request)

    return parse_extraction_response(
        response=response,
        request=request,
    )
