"""
FastAPI Route Unit Tests with Mocked FinGraphPipeline.
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch
import pytest
from fastapi.testclient import TestClient

from api.main import create_app
from src.qa.agent import QAEvidencePackage
from src.qa.evidence_sufficiency import SufficiencyResult
from src.qa.generator import QAAnswer
from src.retrieval.evidence import RetrievalEvidence


@pytest.fixture
def mock_pipeline():
    mock = MagicMock()
    mock.kg_path = "/mock/kg.json"
    mock.vector_path = "/mock/vector"
    mock.hop_mode = "adaptive"
    mock.balanced_fusion = True
    mock.min_per_entity = 3
    mock.regulatory_linking = True

    # Vector store mock
    mock_vs = MagicMock()
    mock_vs.collection_name = "fingraph_chunks"
    mock_vs.vector_size = 384
    mock_col = MagicMock()
    mock_col.points_count = 8181
    mock_vs.client.get_collection.return_value = mock_col
    mock.vector_store = mock_vs

    # KG mock
    mock_kg = MagicMock()
    mock_node = MagicMock()
    mock_node.node_id = "company:MSFT"
    mock_node.node_type = "company"
    mock_node.properties = {
        "name": "Microsoft Corporation",
        "canonical_name": "Microsoft Corporation",
        "ticker": "MSFT",
    }
    other_nodes = [
        MagicMock(node_id=f"node_{i}", node_type="company", properties={"name": f"Entity {i}"})
        for i in range(10053)
    ]
    mock_kg.nodes = [mock_node] + other_nodes
    mock_edge = MagicMock(
        edge_id="edge_msft_1",
        source="company:MSFT",
        target="node_1",
        relationship="PARTNERS_WITH",
        available_time=None,
        event_time=None,
        properties={},
        confidence=0.9,
    )
    mock_kg.edges = [mock_edge] + [
        MagicMock(
            edge_id=f"edge_{i}",
            source=f"node_{i}",
            target=f"node_{i+1}",
            relationship="RELATED",
            available_time=None,
            event_time=None,
            properties={},
            confidence=0.8,
        )
        for i in range(9821)
    ]
    mock_gr = MagicMock()
    mock_gr.graph = mock_kg
    mock_gr.regulatory_linking = True
    mock.graph_retriever = mock_gr

    return mock


@pytest.fixture
def client(mock_pipeline):
    app = create_app()
    app.state.pipeline = mock_pipeline
    app.state.pipeline_lock = asyncio.Lock()
    with TestClient(app) as test_client:
        yield test_client


def test_root_endpoint(client):
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "FinGraph AI Research API"
    assert "api_v1" in data
    assert data["status"] == "online"


def test_openapi_docs(client):
    response = client.get("/openapi.json")
    assert response.status_code == 200
    data = response.json()
    paths = data.get("paths", {})
    assert "/api/v1/query" in paths
    assert "/api/v1/health" in paths
    assert "/api/v1/benchmark/demo-questions" in paths


@patch("api.routes.health._check_ollama")
def test_health_endpoint_healthy(mock_ollama, client):
    from api.schemas.health import ComponentHealth
    mock_ollama.return_value = ComponentHealth(
        status="healthy",
        message="Ollama reachable",
        details={"host": "http://localhost:11434", "model": "qwen3.5:4b", "model_present": True},
    )

    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["pipeline"]["status"] == "healthy"
    assert data["qdrant"]["status"] == "healthy"
    assert data["knowledge_graph"]["status"] == "healthy"
    assert data["llm"]["status"] == "healthy"
    assert data["qdrant"]["details"]["points_count"] == 8181
    assert data["knowledge_graph"]["details"]["node_count"] == 10054


@patch("api.routes.health._check_ollama")
def test_health_endpoint_unavailable_llm(mock_ollama, client):
    from api.schemas.health import ComponentHealth
    mock_ollama.return_value = ComponentHealth(
        status="unavailable",
        message="Cannot connect to Ollama",
        details={},
    )

    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "unavailable"
    assert data["llm"]["status"] == "unavailable"


def test_demo_questions_endpoint(client):
    response = client.get("/api/v1/benchmark/demo-questions")
    assert response.status_code == 200
    data = response.json()
    assert data["total_count"] == 5
    assert len(data["questions"]) == 5

    q_ids = [q["question_id"] for q in data["questions"]]
    assert "FSH-07" in q_ids
    assert "CDC-04" in q_ids
    assert "RMH-07" in q_ids
    assert "TPT-01" in q_ids
    assert "TPT-02" in q_ids

    # Check cached results present
    fsh = next(q for q in data["questions"] if q["question_id"] == "FSH-07")
    assert fsh["cached_phase7_result"]["decision"] == "answer"
    assert fsh["cached_phase7_result"]["token_f1"] == 1.0

    tpt2 = next(q for q in data["questions"] if q["question_id"] == "TPT-02")
    assert tpt2["cached_phase7_result"]["decision"] == "refuse"
    assert tpt2["cached_phase7_result"]["token_f1"] == 1.0


def test_query_endpoint_answerable(mock_pipeline, client):
    # Mock evidence
    ev = RetrievalEvidence(
        evidence_id="hybrid:doc:sec:1:chunk:1",
        source_type="hybrid",
        score=0.92,
        rank=1,
        text="Microsoft acquired Activision Blizzard in October 2023.",
        document_id="doc:sec:1",
        chunk_id="chunk:1",
        available_time="2023-10-15T00:00:00Z",
        metadata={"provenance_verified": True},
    )

    pkg = QAEvidencePackage(
        query="What company did Microsoft acquire?",
        reference_time=None,
        question_type="factual_single_hop",
        expected_hops=1,
        evidence=[ev],
        graph_paths=[["company:MSFT", "ACQUIRED", "Activision Blizzard"]],
        sources=["doc:sec:1"],
        retrieval_rounds=1,
        temporal_valid=True,
        sufficient=True,
        primary_evidence=[ev],
        supporting_evidence=[],
        sufficiency_result=SufficiencyResult(
            sufficient=True,
            reason="All constraints met",
            evidence_count=1,
            temporal_valid=True,
            provenance_complete=True,
        ),
    )

    ans = QAAnswer(
        answer="Microsoft acquired Activision Blizzard.",
        confidence=0.98,
        evidence_ids=["hybrid:doc:sec:1:chunk:1"],
        citations=[{
            "evidence_id": "hybrid:doc:sec:1:chunk:1",
            "document_id": "doc:sec:1",
            "chunk_id": "chunk:1",
            "source_type": "hybrid",
            "text_snippet": "Microsoft acquired Activision Blizzard",
        }],
        insufficient_evidence=False,
        reasoning_summary="Direct acquisition evidence found in 10-K.",
        sources=["doc:sec:1"],
    )

    mock_pipeline.answer.return_value = (pkg, ans)

    response = client.post(
        "/api/v1/query",
        json={"query": "What company did Microsoft acquire?", "top_k": 5},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["decision"] == "answer"
    assert data["answer"] == "Microsoft acquired Activision Blizzard."
    assert data["insufficient_evidence"] is False
    assert len(data["evidence"]) == 1
    assert len(data["citations"]) == 1
    assert data["sufficiency"]["sufficient"] is True
    assert data["temporal_validation"]["temporal_valid"] is True
    assert data["latency_ms"] > 0
    assert "X-Request-ID" in response.headers
    assert "X-Response-Time-Ms" in response.headers


def test_query_endpoint_refusal(mock_pipeline, client):
    pkg = QAEvidencePackage(
        query="What risks affected NVIDIA as of June 1, 2025?",
        reference_time="2025-06-01T00:00:00+00:00",
        question_type="temporal_point_in_time",
        expected_hops=1,
        evidence=[],
        graph_paths=[],
        sources=[],
        retrieval_rounds=1,
        temporal_valid=True,
        sufficient=False,
        primary_evidence=[],
        supporting_evidence=[],
        sufficiency_result=SufficiencyResult(
            sufficient=False,
            reason="No filings available before reference date",
            evidence_count=0,
            temporal_valid=True,
            provenance_complete=True,
        ),
    )

    ans = QAAnswer(
        answer="Insufficient evidence in the retrieved corpus.",
        confidence=0.0,
        evidence_ids=[],
        citations=[],
        insufficient_evidence=True,
        reasoning_summary="No pre-cutoff evidence.",
        sources=[],
    )

    mock_pipeline.answer.return_value = (pkg, ans)

    response = client.post(
        "/api/v1/query",
        json={
            "query": "What risks affected NVIDIA as of June 1, 2025?",
            "reference_time": "2025-06-01T00:00:00+00:00",
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["decision"] == "refuse"
    assert data["insufficient_evidence"] is True
    assert data["answer"] == "Insufficient evidence in the retrieved corpus."
    assert data["sufficiency"]["sufficient"] is False


def test_query_validation_error(client):
    # Query length < 3
    response = client.post("/api/v1/query", json={"query": "a"})
    assert response.status_code == 422
    data = response.json()
    assert data["error"] == "ValidationError"


def test_query_internal_error(mock_pipeline, client):
    mock_pipeline.answer.side_effect = RuntimeError("Simulated internal pipeline failure")
    response = client.post(
        "/api/v1/query",
        json={"query": "Valid query that triggers error"},
    )
    assert response.status_code == 500
    data = response.json()
    assert "Simulated internal pipeline failure" in data["detail"]


def test_graph_endpoints(client):
    # Test companies listing
    res_comp = client.get("/api/v1/graph/companies?limit=10")
    assert res_comp.status_code == 200
    data_comp = res_comp.json()
    assert "companies" in data_comp
    assert "total_companies" in data_comp

    # Test search
    res_search = client.get("/api/v1/graph/search?q=MSFT&limit=5")
    assert res_search.status_code == 200
    data_search = res_search.json()
    assert isinstance(data_search, list)
    assert len(data_search) > 0
    assert data_search[0]["id"] == "company:MSFT"

    # Test subgraph
    res_subgraph = client.get("/api/v1/graph/subgraph?node_id=company:MSFT")
    assert res_subgraph.status_code == 200
    data_subgraph = res_subgraph.json()
    assert "nodes" in data_subgraph
    assert "edges" in data_subgraph
    assert "center_node" in data_subgraph

