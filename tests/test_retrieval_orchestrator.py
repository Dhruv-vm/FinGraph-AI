from __future__ import annotations

from unittest.mock import MagicMock
import pytest

from src.qa.query_analysis import QueryAnalysis
from src.retrieval.evidence import RetrievalEvidence
from src.retrieval.orchestrator import RetrievalOrchestrator


def test_initial_hybrid_retrieval_called_with_reference_time():
    mock_hybrid = MagicMock()
    mock_hybrid.retrieve.return_value = [
        RetrievalEvidence(
            evidence_id="ev_1",
            source_type="hybrid",
            score=0.9,
            rank=1,
            text="Evidence 1",
            document_id="doc_1",
            chunk_id="chunk_1",
            available_time="2024-01-01T00:00:00+00:00",
            metadata={"hop_count": 1},
        )
    ]

    orchestrator = RetrievalOrchestrator(hybrid_retriever=mock_hybrid)
    cutoff = "2024-03-31T00:00:00+00:00"

    results, analysis, sufficiency = orchestrator.orchestrate(
        query="What companies does NVIDIA depend on as of 2024-03-31?",
    )

    assert mock_hybrid.retrieve.called
    kwargs = mock_hybrid.retrieve.call_args[1]
    # Check reference time was detected and passed down
    assert kwargs["reference_time"] == cutoff


def test_reranking_preserves_provenance():
    mock_hybrid = MagicMock()
    mock_hybrid.retrieve.return_value = [
        RetrievalEvidence(
            evidence_id="ev_prov",
            source_type="hybrid",
            score=0.88,
            rank=1,
            text="Detailed relation",
            document_id="doc_sec_10k_2024",
            chunk_id="chunk_sec_42",
            publication_date="2024-02-15",
            available_time="2024-02-15T10:00:00+00:00",
            source_url="https://sec.gov/filing/10k",
            retrieval_sources=["vector", "graph"],
            metadata={"hop_count": 1},
        )
    ]

    orchestrator = RetrievalOrchestrator(hybrid_retriever=mock_hybrid)
    results, _, _ = orchestrator.orchestrate("NVIDIA dependencies")

    assert len(results) == 1
    res = results[0]
    assert res.document_id == "doc_sec_10k_2024"
    assert res.chunk_id == "chunk_sec_42"
    assert res.publication_date == "2024-02-15"
    assert res.available_time == "2024-02-15T10:00:00+00:00"
    assert res.source_url == "https://sec.gov/filing/10k"
    assert res.retrieval_sources == ["vector", "graph"]


def test_temporal_cutoff_survives_orchestration():
    mock_hybrid = MagicMock()
    # Hybrid returns one historical and one future
    mock_hybrid.retrieve.return_value = [
        RetrievalEvidence(
            evidence_id="ev_historical",
            source_type="graph",
            score=0.75,
            rank=1,
            text="Historical event",
            document_id="doc_h",
            chunk_id="chunk_h",
            available_time="2024-01-01T00:00:00+00:00",
            metadata={"hop_count": 1},
        ),
        RetrievalEvidence(
            evidence_id="ev_future",
            source_type="graph",
            score=0.99,
            rank=2,
            text="Future leak",
            document_id="doc_f",
            chunk_id="chunk_f",
            available_time="2025-01-01T00:00:00+00:00",
            metadata={"hop_count": 1},
        ),
    ]

    orchestrator = RetrievalOrchestrator(hybrid_retriever=mock_hybrid)
    cutoff = "2024-06-01T00:00:00+00:00"

    results, _, sufficiency = orchestrator.orchestrate(
        query="Historical query",
        reference_time=cutoff,
    )

    # Future evidence must be filtered out by orchestrator reranking
    assert len(results) == 1
    assert results[0].evidence_id == "ev_historical"
    assert not any(r.evidence_id == "ev_future" for r in results)
    assert sufficiency.temporal_valid is True
