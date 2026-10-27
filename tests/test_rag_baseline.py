from __future__ import annotations

from unittest.mock import MagicMock, patch
import pytest

from scripts.run_rag_baseline import VectorOnlyRAG
from src.retrieval.vector import RetrievedChunk


@pytest.fixture
def mock_vector_store():
    store = MagicMock()
    # Mock search returning 2 chunks
    store.search.return_value = [
        RetrievedChunk(
            chunk_id="doc:sec:a:chunk:1",
            score=0.88,
            text="Apple manufactures Personal Computers and smartphones.",
            payload={
                "document_id": "doc:sec:a",
                "available_time": "2025-08-01T00:00:00+00:00",
                "company": "AAPL",
            },
        ),
        RetrievedChunk(
            chunk_id="doc:sec:a:chunk:2",
            score=0.75,
            text="Apple products include tablets and watches.",
            payload={
                "document_id": "doc:sec:a",
                "available_time": "2025-08-01T00:00:00+00:00",
                "company": "AAPL",
            },
        ),
    ]
    return store


def test_baseline_does_not_call_graph_or_hybrid_components(mock_vector_store):
    """
    CRUCIAL SCIENTIFIC ISOLATION TEST:
    Verify that VectorOnlyRAG does NOT instantiate or invoke GraphRetriever,
    HybridRetriever, DeterministicReranker, or RetrievalAgent.
    """
    with patch("src.retrieval.graph.GraphRetriever") as mock_gr, \
         patch("src.retrieval.hybrid.HybridRetriever") as mock_hr, \
         patch("src.retrieval.reranker.DeterministicReranker") as mock_rr, \
         patch("src.qa.agent.RetrievalAgent") as mock_agent:

        # Mock LLM callable
        def mock_llm(prompt: str, sys_prompt: str) -> str:
            return (
                '{"answer": "Apple manufactures Personal Computers and smartphones.", '
                '"confidence": 0.95, "evidence_ids": ["vector:doc:sec:a:chunk:1"], '
                '"reasoning_summary": "Direct statement."}'
            )

        rag = VectorOnlyRAG(
            vector_store=mock_vector_store,
            llm_client=mock_llm,
        )

        ans = rag.answer("What products does Apple manufacture?")

        # Assert vector store was searched
        mock_vector_store.search.assert_called_once()

        # Assert Graph and Hybrid components were NEVER called
        mock_gr.assert_not_called()
        mock_hr.assert_not_called()
        mock_rr.assert_not_called()
        mock_agent.assert_not_called()

        assert "Apple manufactures" in ans.answer
        assert len(ans.evidence_ids) == 1
        assert ans.citations[0]["chunk_id"] == "doc:sec:a:chunk:1"


def test_baseline_temporal_filter_passing(mock_vector_store):
    rag = VectorOnlyRAG(vector_store=mock_vector_store)
    rag.retrieve("Query", reference_time="2025-06-01T00:00:00+00:00")

    mock_vector_store.search.assert_called_once()
    _, kwargs = mock_vector_store.search.call_args
    assert kwargs.get("reference_time") is not None


def test_baseline_refuses_when_vector_store_empty():
    empty_store = MagicMock()
    empty_store.search.return_value = []

    rag = VectorOnlyRAG(vector_store=empty_store)
    ans = rag.answer("What products does Mystery Corp manufacture?")

    assert ans.insufficient_evidence is True
    assert "Insufficient evidence" in ans.answer
    assert len(ans.evidence_ids) == 0
    assert len(ans.citations) == 0
