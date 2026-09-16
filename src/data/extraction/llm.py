from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Protocol

from dotenv import load_dotenv
from openai import OpenAI
from ollama import Client as OllamaClient

from .entities import entities_from_dicts, validate_entity_type
from .relations import is_valid_relationship, relations_from_dicts
from .schema import ExtractionResult


SYSTEM_PROMPT = """
You are a financial information extraction system.

Extract entities and semantic relationships from the supplied financial
document.

Return ONLY valid JSON.
Do not provide explanations, reasoning, markdown, code fences, or commentary.

The JSON must have this structure:

{
  "entities": [
    {
      "entity_type": "Company",
      "name": "<COMPANY>",
      "canonical_name": "<COMPANY>",
      "confidence": 0.95,
      "properties": {}
    }
  ],
  "relations": [
    {
      "source_entity": "<COMPANY>",
      "target_entity": "<PRODUCT>",
      "relationship": "MANUFACTURES",
      "confidence": 0.90,
      "event_time": null,
      "available_time": null,
      "properties": {}
    }
  ]
}

Allowed entity types:
- Company
- Person
- Product
- Supplier
- Competitor
- Event
- Risk
- Country
- FinancialMetric
- Document
- NewsArticle

Allowed relationships:
- SUPPLIES
- MANUFACTURES
- DEPENDS_ON
- COMPETES_WITH
- PARTNERS_WITH
- INVESTED_IN
- ACQUIRED
- LOCATED_IN
- AFFECTED_BY
- CAUSED
- ANNOUNCED
- HAS_RISK
- MENTIONED_IN

Important:
- Use entity names in source_entity and target_entity.
- Do NOT invent internal entity IDs.
- Only extract relationships supported by the supplied text.
- Do not infer unsupported facts.
- Use ONLY the exact relationship names listed above.
- NEVER invent, rename, paraphrase, or translate a relationship type.
- NEVER use relationship names such as "tradedOn", "hasClassStock",
  "representsInterestIn", "owns", "hasStock", or "listedOn".
- If a relationship cannot be represented by one of the allowed
  relationship types, omit that relationship.
- Confidence must be between 0 and 1.
- Use null for unknown temporal values.

Extraction scope:
- Extract only semantically important entities needed for financial
  question answering and graph reasoning.
- Prefer a small set of high-value entities over exhaustive extraction.
- Do NOT create an Event for every sentence, financial value, accounting
  line item, or numerical observation.
- Represent financial quantities as FinancialMetric entities when they
  are meaningful to the document's financial context.
- Avoid duplicate entities that express the same fact.
- Do not extract individual table rows as separate entities unless they
  are independently useful for reasoning.
- Extract only relationships that connect meaningful entities and are
  explicitly supported by the text.
- Prefer high-confidence relationships and omit weak or redundant ones.
- For each chunk, return at most 15 entities and 9 relationships.
- Prefer zero relationships over weak or repetitive relationships.
- Do NOT copy or reuse example entities, names, values, or relationships
  from this system prompt. "Example Corp" and "Example Product" are
  placeholders only and must never appear unless they are actually
  present in the supplied document text.
- Do not create MANUFACTURES, SUPPLIES, DEPENDS_ON, or other business
  relationships merely because a financial table contains an accounting
  line item.
- FinancialMetric entities normally do not need relationships unless the
  supplied text explicitly describes one of the allowed relationships.
- Do not create relationships from a company to every financial metric,
  debt category, expense, balance-sheet line, or table row.
- If a chunk contains mostly financial tables with no meaningful semantic
  relationships, return an empty relations array.
- Keep the JSON compact and complete. Never stop in the middle of an
  object, array, string, or JSON structure.
"""


class LLMClient(Protocol):
    def extract(self, text: str) -> str:
        ...


@dataclass
class ExtractionRequest:
    document_id: str
    text: str
    available_time: str | None = None
    publication_date: str | None = None
    metadata: dict[str, Any] | None = None


class OpenRouterLLMClient:
    def __init__(
        self,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> None:
        load_dotenv(".env")

        api_key = os.getenv("LLM_API_KEY")

        if not api_key:
            raise ValueError(
                "LLM_API_KEY is required for OpenRouter extraction."
            )

        self.model = model or os.getenv(
            "LLM_MODEL",
            "openrouter/free",
        )

        self.temperature = (
            temperature
            if temperature is not None
            else float(os.getenv("LLM_TEMPERATURE", "0"))
        )

        self.max_tokens = (
            max_tokens
            if max_tokens is not None
            else int(os.getenv("LLM_MAX_TOKENS", "3000"))
        )

        self.client = OpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
        )

    def extract(self, text: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": (
                        "Extract the financial entities and relationships "
                        "from this document text:\n\n"
                        f"{text}"
                    ),
                },
            ],
        )

        if not response.choices:
            raise ValueError("OpenRouter returned no choices.")

        message = response.choices[0].message
        content = message.content

        if content and content.strip():
            return content.strip()

        raise ValueError(
            "OpenRouter returned an empty response "
            f"(finish_reason={response.choices[0].finish_reason})."
        )


class MockLLMClient:
    def __init__(self, response: str | None = None) -> None:
        self.response = response or json.dumps(
            {
                "entities": [],
                "relations": [],
            }
        )

    def extract(self, text: str) -> str:
        return self.response


def _normalize_lookup_name(value: str) -> str:
    return " ".join(str(value).strip().split()).casefold()


def _resolve_relation_entity(
    value: str,
    entity_lookup: dict[str, str],
    *,
    role: str,
) -> str:
    normalized = _normalize_lookup_name(value)

    if normalized not in entity_lookup:
        raise ValueError(
            f"unknown {role} entity: {value!r}"
        )

    return entity_lookup[normalized]


def _prepare_relations(
    raw_relations: list[dict[str, Any]],
    entity_lookup: dict[str, str],
    *,
    default_available_time: str | None = None,
) -> list[dict[str, Any]]:
    prepared: list[dict[str, Any]] = []

    for raw in raw_relations:
        item = dict(raw)

        # New LLM-facing schema.
        source_name = item.pop("source_entity", None)
        target_name = item.pop("target_entity", None)

        # Backward-compatible aliases.
        if source_name is None:
            source_name = item.pop("source_entity_name", None)

        if target_name is None:
            target_name = item.pop("target_entity_name", None)

        # Existing deterministic-ID schema remains supported.
        source_id = item.pop("source_entity_id", None)
        target_id = item.pop("target_entity_id", None)

        if source_id is None:
            if source_name is None:
                continue

            normalized_source = _normalize_lookup_name(source_name)

            if normalized_source not in entity_lookup:
                continue

            source_id = entity_lookup[normalized_source]

        if target_id is None:
            if target_name is None:
                continue

            normalized_target = _normalize_lookup_name(target_name)

            if normalized_target not in entity_lookup:
                continue

            target_id = entity_lookup[normalized_target]

        item["source_entity_id"] = source_id
        item["target_entity_id"] = target_id

        if (
            not item.get("available_time")
            and default_available_time is not None
        ):
            item["available_time"] = default_available_time

        # LLMs may occasionally generate relationship types outside the
        # controlled KG schema. Skip unsupported relations while preserving
        # valid extracted entities and relations.
        if not is_valid_relationship(item.get("relationship")):
            continue

        prepared.append(item)

    return prepared


def parse_extraction_response(
    response: str,
    request: ExtractionRequest,
) -> ExtractionResult:
    try:
        payload = json.loads(response)
    except json.JSONDecodeError as exc:
        raise ValueError("LLM response is not valid JSON") from exc

    if not isinstance(payload, dict):
        raise ValueError("LLM response must be a JSON object.")

    raw_entities = payload.get("entities", [])
    raw_relations = payload.get("relations", [])

    if not isinstance(raw_entities, list):
        raise ValueError("'entities' must be a list.")

    if not isinstance(raw_relations, list):
        raise ValueError("'relations' must be a list.")

    # LLMs may occasionally generate entity types outside the
    # controlled KG schema. Skip unsupported entities while preserving
    # valid extracted entities.
    supported_entities = [
        entity
        for entity in raw_entities
        if validate_entity_type(entity.get("entity_type"))
    ]

    entities = entities_from_dicts(supported_entities)

    entity_lookup: dict[str, str] = {}

    for entity in entities:
        entity_lookup[
            _normalize_lookup_name(entity.name)
        ] = entity.entity_id

        entity_lookup[
            _normalize_lookup_name(entity.canonical_name)
        ] = entity.entity_id

    prepared_relations = _prepare_relations(
        raw_relations,
        entity_lookup,
        default_available_time=request.available_time,
    )

    relations = relations_from_dicts(prepared_relations)

    entity_ids = {
        entity.entity_id
        for entity in entities
    }

    for relation in relations:
        if relation.source_entity_id not in entity_ids:
            raise ValueError(
                "unknown source entity: "
                f"{relation.source_entity_id}"
            )

        if relation.target_entity_id not in entity_ids:
            raise ValueError(
                "unknown target entity: "
                f"{relation.target_entity_id}"
            )

    return ExtractionResult(
        document_id=request.document_id,
        entities=entities,
        relations=relations,
        metadata={
            "parser": "fingraph-extraction-parser",
            "relation_resolution": (
                "entity_name_to_deterministic_id"
            ),
        },
    )


OLLAMA_SYSTEM_PROMPT = """
You are a financial information extraction system.

Extract high-value financial entities and explicit semantic relationships
from the supplied financial document.

Return ONLY valid JSON:
{
  "entities": [
    {
      "entity_type": "Company",
      "name": "NVIDIA",
      "confidence": 0.98
    }
  ],
  "relations": [
    {
      "source_entity": "NVIDIA",
      "target_entity": "TSMC",
      "relationship": "SUPPLIES",
      "confidence": 0.95
    }
  ]
}

Allowed entity types:
Company, Person, Product, Supplier, Competitor, Event, Risk,
Country, FinancialMetric, Document, NewsArticle

Allowed relationships:
SUPPLIES, MANUFACTURES, DEPENDS_ON, COMPETES_WITH, PARTNERS_WITH,
INVESTED_IN, ACQUIRED, LOCATED_IN, AFFECTED_BY, CAUSED, ANNOUNCED,
HAS_RISK, MENTIONED_IN

Rules:
- Extract only important entities useful for financial question answering.
- Maximum 12 entities and 5 relationships.
- Only create relationships explicitly supported by the text.
- Never invent facts.
- Do not create relationships from financial table rows.
- FinancialMetric entities normally have no relationships.
- Prefer zero relationships over weak relationships.
- Use exact allowed relationship names.
- Entity names in relations must exactly match entity names.
- Confidence must be between 0 and 1.
- Do not output canonical_name, properties, event_time, available_time,
  entity IDs, or relation IDs.
- Return complete valid JSON with no explanation.
"""


class OllamaLLMClient:
    def __init__(
        self,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> None:
        load_dotenv(".env")

        self.model = model or os.getenv(
            "LLM_MODEL",
            "qwen3.5:4b",
        )

        self.temperature = (
            temperature
            if temperature is not None
            else float(os.getenv("LLM_TEMPERATURE", "0"))
        )

        self.max_tokens = (
            max_tokens
            if max_tokens is not None
            else int(os.getenv("LLM_MAX_TOKENS", "6000"))
        )

        self.client = OllamaClient(
            host=os.getenv("OLLAMA_HOST", "http://localhost:11434")
        )

    def extract(self, text: str) -> str:
        def call_model(
            system_prompt: str,
            user_prompt: str,
        ) -> str:
            response = self.client.chat(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt,
                    },
                    {
                        "role": "user",
                        "content": user_prompt,
                    },
                ],
                options={
                    "temperature": self.temperature,
                    "num_predict": self.max_tokens,
                },
                format="json",
                think=False,
            )

            message = response.get("message", {})
            content = message.get("content")

            if content and content.strip():
                return content.strip()

            raise ValueError("Ollama returned an empty response.")

        normal_prompt = (
            "Extract the financial entities and relationships "
            "from this document text:\n\n"
            f"{text}"
        )

        response = call_model(
            OLLAMA_SYSTEM_PROMPT,
            normal_prompt,
        )

        try:
            json.loads(response)
            return response
        except json.JSONDecodeError:
            compact_system_prompt = (
                "You extract financial knowledge from SEC text. "
                "Return ONLY valid JSON. "
                "Be concise and extract only high-value information."
            )

            compact_prompt = (
                "Extract only the most important financial entities "
                "and relationships from this text.\n\n"
                "Return ONLY valid JSON with exactly two keys: "
                "entities and relations.\n"
                "Each entity must contain: entity_type, name, "
                "canonical_name, confidence, properties.\n"
                "Each relation must contain: relationship, "
                "source_entity, target_entity, confidence, properties.\n"
                "Maximum 8 entities. Maximum 2 relationships.\n"
                "Prefer the company and important FinancialMetric entities.\n"
                "Do not extract every table row.\n"
                "Prefer zero relationships over weak relationships.\n"
                "Do not invent information.\n"
                "Do not include explanations or Markdown.\n"
                "Make sure the JSON is complete and valid.\n\n"
                f"Document text:\n{text}"
            )

            compact_response = call_model(
                compact_system_prompt,
                compact_prompt,
            )

            json.loads(compact_response)

            return compact_response



def extract_document(
    client: LLMClient,
    request: ExtractionRequest,
) -> ExtractionResult:
    response = client.extract(request.text)

    return parse_extraction_response(
        response,
        request,
    )