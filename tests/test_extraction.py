import pytest

from src.data.extraction.entities import build_entity
from src.data.extraction.llm import (
    ExtractionRequest,
    MockLLMClient,
    extract_document,
)
from src.data.extraction.relations import build_relation


def test_entity_is_deterministic():
    first = build_entity(
        entity_type="Company",
        name="NVIDIA Corporation",
    )
    second = build_entity(
        entity_type="Company",
        name="NVIDIA Corporation",
    )

    assert first.entity_id == second.entity_id


def test_entity_rejects_unknown_type():
    with pytest.raises(ValueError):
        build_entity(
            entity_type="UnknownType",
            name="Example",
        )


def test_relation_is_deterministic():
    first = build_relation(
        source_entity_id="company:nvidia",
        target_entity_id="company:tsmc",
        relationship="SUPPLIES",
    )
    second = build_relation(
        source_entity_id="company:nvidia",
        target_entity_id="company:tsmc",
        relationship="SUPPLIES",
    )

    assert first.relation_id == second.relation_id


def test_relation_rejects_unknown_relationship():
    with pytest.raises(ValueError):
        build_relation(
            source_entity_id="company:nvidia",
            target_entity_id="company:tsmc",
            relationship="UNKNOWN_RELATION",
        )


def test_relation_rejects_invalid_temporal_order():
    with pytest.raises(ValueError):
        build_relation(
            source_entity_id="company:nvidia",
            target_entity_id="company:tsmc",
            relationship="SUPPLIES",
            event_time="2026-08-01T12:00:00+00:00",
            available_time="2026-07-01T12:00:00+00:00",
        )


def test_llm_extraction_returns_structured_result():
    response = """
    {
      "entities": [
        {
          "entity_type": "Company",
          "name": "NVIDIA",
          "canonical_name": "NVIDIA",
          "confidence": 0.98
        },
        {
          "entity_type": "Supplier",
          "name": "TSMC",
          "canonical_name": "TSMC",
          "confidence": 0.96
        }
      ],
      "relations": [
        {
          "source_entity_id": "company:placeholder",
          "target_entity_id": "supplier:placeholder",
          "relationship": "SUPPLIES",
          "confidence": 0.95
        }
      ]
    }
    """

    # Entity IDs are deterministic, so construct a valid response first.
    nvidia = build_entity("Company", "NVIDIA")
    tsmc = build_entity("Supplier", "TSMC")

    response = f"""
    {{
      "entities": [
        {{
          "entity_type": "Company",
          "name": "NVIDIA",
          "canonical_name": "NVIDIA",
          "confidence": 0.98
        }},
        {{
          "entity_type": "Supplier",
          "name": "TSMC",
          "canonical_name": "TSMC",
          "confidence": 0.96
        }}
      ],
      "relations": [
        {{
          "source_entity_id": "{nvidia.entity_id}",
          "target_entity_id": "{tsmc.entity_id}",
          "relationship": "SUPPLIES",
          "confidence": 0.95
        }}
      ]
    }}
    """

    request = ExtractionRequest(
        document_id="doc:test:001",
        text="NVIDIA uses TSMC as a manufacturing supplier.",
        available_time="2026-08-01T12:00:00+00:00",
    )

    result = extract_document(
        MockLLMClient(response),
        request,
    )

    assert result.document_id == "doc:test:001"
    assert len(result.entities) == 2
    assert len(result.relations) == 1
    assert result.relations[0].available_time == (
        "2026-08-01T12:00:00+00:00"
    )


def test_llm_rejects_invalid_json():
    request = ExtractionRequest(
        document_id="doc:test:002",
        text="Example financial document.",
    )

    with pytest.raises(ValueError, match="valid JSON"):
        extract_document(
            MockLLMClient("not valid json"),
            request,
        )


def test_llm_rejects_dangling_relation():
    entity = build_entity("Company", "NVIDIA")

    response = f"""
    {{
      "entities": [
        {{
          "entity_type": "Company",
          "name": "NVIDIA"
        }}
      ],
      "relations": [
        {{
          "source_entity_id": "{entity.entity_id}",
          "target_entity_id": "company:does-not-exist",
          "relationship": "SUPPLIES"
        }}
      ]
    }}
    """

    request = ExtractionRequest(
        document_id="doc:test:003",
        text="Example financial document.",
    )

    with pytest.raises(ValueError, match="unknown target entity"):
        extract_document(
            MockLLMClient(response),
            request,
        )


def test_llm_skips_unsupported_relationship_types():
    response = """
    {
      "entities": [
        {
          "entity_type": "Company",
          "name": "NVIDIA",
          "canonical_name": "NVIDIA"
        },
        {
          "entity_type": "Company",
          "name": "Microsoft",
          "canonical_name": "Microsoft"
        }
      ],
      "relations": [
        {
          "source_entity": "NVIDIA",
          "target_entity": "Microsoft",
          "relationship": "PART_OF"
        },
        {
          "source_entity": "NVIDIA",
          "target_entity": "Microsoft",
          "relationship": "PARTNERS_WITH"
        }
      ]
    }
    """

    request = ExtractionRequest(
        document_id="doc:test:004",
        text="Example financial document.",
    )

    result = extract_document(
        MockLLMClient(response),
        request,
    )

    assert len(result.relations) == 1
    assert result.relations[0].relationship == "PARTNERS_WITH"


def test_llm_skips_unsupported_entity_types():
    response = """
    {
      "entities": [
        {
          "entity_type": "Location",
          "name": "California",
          "canonical_name": "California"
        },
        {
          "entity_type": "Company",
          "name": "NVIDIA",
          "canonical_name": "NVIDIA"
        }
      ],
      "relations": []
    }
    """

    request = ExtractionRequest(
        document_id="doc:test:005",
        text="NVIDIA is headquartered in California.",
    )

    result = extract_document(
        MockLLMClient(response),
        request,
    )

    assert len(result.entities) == 1
    assert result.entities[0].name == "NVIDIA"
