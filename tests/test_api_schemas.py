"""
Unit tests for FastAPI Pydantic Schemas.
"""

import pytest
from pydantic import ValidationError

from api.schemas.query import (
    CitationItem,
    EvidenceItem,
    QueryRequest,
    QueryResponse,
    SufficiencyDetail,
    TemporalValidationDetail,
)
from api.schemas.health import ComponentHealth, HealthResponse
from api.schemas.benchmark import (
    CachedBenchmarkResult,
    DemoQuestionItem,
    DemoQuestionsResponse,
)


def test_query_request_validation():
    # Valid request
    req = QueryRequest(query="What is Microsoft's cloud revenue?", top_k=5)
    assert req.query == "What is Microsoft's cloud revenue?"
    assert req.top_k == 5
    assert req.reference_time is None

    # Too short query (< 2 chars). Two-character inputs such as "hi" are accepted so the
    # assist chatbot can reply to greetings.
    with pytest.raises(ValidationError):
        QueryRequest(query="h")
    assert QueryRequest(query="hi").query == "hi"

    # Invalid top_k (< 1)
    with pytest.raises(ValidationError):
        QueryRequest(query="Valid query", top_k=0)

    # Invalid top_k (> 50)
    with pytest.raises(ValidationError):
        QueryRequest(query="Valid query", top_k=51)


def test_evidence_item_serialization():
    item = EvidenceItem(
        evidence_id="hybrid:doc:sec:123:chunk:1",
        source_type="hybrid",
        score=0.88,
        rank=1,
        text="Sample financial text disclosure",
        document_id="doc:sec:123",
        chunk_id="doc:sec:123:chunk:1",
        provenance_verified=True,
    )
    assert item.evidence_id == "hybrid:doc:sec:123:chunk:1"
    assert item.source_type == "hybrid"
    assert item.provenance_verified is True


def test_query_response_serialization():
    resp = QueryResponse(
        query="What is Microsoft's cloud revenue?",
        decision="answer",
        answer="Microsoft cloud revenue reached $38.9B.",
        insufficient_evidence=False,
        confidence=0.95,
        sufficiency=SufficiencyDetail(
            sufficient=True,
            reason="All constraints met",
            evidence_count=5,
        ),
        temporal_validation=TemporalValidationDetail(
            temporal_valid=True,
            context_items_count=5,
        ),
        latency_ms=150.2,
        timestamp="2026-10-31T12:00:00Z",
    )
    assert resp.decision == "answer"
    assert resp.answer == "Microsoft cloud revenue reached $38.9B."
    assert resp.insufficient_evidence is False
    assert resp.sufficiency.sufficient is True


def test_health_response_schema():
    health = HealthResponse(
        status="healthy",
        version="0.1.0",
        timestamp="2026-10-31T12:00:00Z",
        pipeline=ComponentHealth(status="healthy", message="Pipeline ready"),
        qdrant=ComponentHealth(status="healthy", message="Vector store connected"),
        knowledge_graph=ComponentHealth(status="healthy", message="10054 nodes"),
        llm=ComponentHealth(status="healthy", message="Ollama reachable"),
    )
    assert health.status == "healthy"
    assert health.pipeline.status == "healthy"
    assert health.llm.status == "healthy"


def test_benchmark_schemas():
    demo = DemoQuestionsResponse(
        total_count=1,
        questions=[
            DemoQuestionItem(
                question_id="FSH-07",
                question="What major video gaming company did Microsoft acquire?",
                category="factual_single_hop",
                answerable=True,
                expected_hops=1,
                ground_truth_answer="Microsoft acquired Activision Blizzard.",
                cached_phase7_result=CachedBenchmarkResult(
                    decision="answer",
                    token_f1=1.0,
                    generated_answer="Microsoft acquired Activision Blizzard.",
                    latency_sec=12.3,
                    outcome_status="TN_correct_decision_to_answer",
                ),
            )
        ],
    )
    assert demo.total_count == 1
    assert demo.questions[0].question_id == "FSH-07"
    assert demo.questions[0].cached_phase7_result.token_f1 == 1.0
