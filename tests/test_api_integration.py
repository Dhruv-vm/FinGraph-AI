"""
FastAPI Live Pipeline Integration Tests.

Validates end-to-end execution of FinGraphPipeline via FastAPI TestClient.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from api.main import app


@pytest.fixture(scope="module")
def live_client():
    """
    Module-scoped TestClient that triggers the actual lifespan event,
    loading the singleton FinGraphPipeline (Qdrant, KG, LLM generator).
    """
    with TestClient(app) as client:
        yield client


def test_live_root_and_openapi(live_client):
    res_root = live_client.get("/")
    assert res_root.status_code == 200
    assert res_root.json()["status"] == "online"

    res_spec = live_client.get("/openapi.json")
    assert res_spec.status_code == 200
    assert "FinGraph AI Research API" in res_spec.json()["info"]["title"]


def test_live_health(live_client):
    response = live_client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ("healthy", "degraded")
    assert data["pipeline"]["status"] == "healthy"
    assert data["knowledge_graph"]["status"] == "healthy"
    assert data["knowledge_graph"]["details"]["node_count"] > 10000
    assert data["qdrant"]["status"] == "healthy"
    assert data["qdrant"]["details"]["points_count"] > 8000


def test_live_demo_questions(live_client):
    response = live_client.get("/api/v1/benchmark/demo-questions")
    assert response.status_code == 200
    data = response.json()
    assert data["total_count"] == 5
    question_ids = [q["question_id"] for q in data["questions"]]
    assert question_ids == ["FSH-07", "CDC-04", "RMH-07", "TPT-01", "TPT-02"]


def test_live_query_temporal_refusal(live_client):
    """
    Verifies live pipeline temporal point-in-time refusal:
    TPT-02 asks for NVIDIA risks as of June 1, 2025.
    Must refuse fail-closed with zero future leakage.
    """
    payload = {
        "query": "What risks affected NVIDIA as of June 1, 2025?",
        "reference_time": "2025-06-01T00:00:00+00:00",
        "top_k": 5,
    }
    response = live_client.post("/api/v1/query", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["decision"] == "refuse"
    assert data["insufficient_evidence"] is True
    assert "Insufficient evidence" in data["answer"]
    assert data["temporal_validation"]["temporal_valid"] is True
    assert data["temporal_validation"]["future_items_count"] == 0
    assert data["latency_ms"] > 0
    assert "X-Request-ID" in response.headers
    assert "X-Response-Time-Ms" in response.headers


def test_live_query_factual_answer(live_client):
    """
    Verifies live pipeline factual retrieval and answer generation:
    FSH-07 asks for Microsoft acquisition of video gaming company.
    """
    payload = {
        "query": "What major video gaming and entertainment company did Microsoft acquire?",
        "top_k": 5,
    }
    response = live_client.post("/api/v1/query", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["decision"] == "answer"
    assert data["insufficient_evidence"] is False
    assert "Activision" in data["answer"]
    assert len(data["evidence"]) > 0
    assert len(data["citations"]) > 0
    assert data["sufficiency"]["sufficient"] is True
    assert data["latency_ms"] > 0
