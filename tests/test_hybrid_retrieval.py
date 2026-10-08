from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock
import pytest

from src.data.graph.schema import GraphEdge, GraphNode, TemporalGraph
from src.retrieval.evidence import GraphEvidence, RetrievalEvidence
from src.retrieval.graph import GraphRetriever
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.vector import RetrievedChunk


@pytest.fixture
def mock_vector_chunks() -> list[RetrievedChunk]:
    return [
        RetrievedChunk(
            chunk_id="chunk_shared_01",
            score=0.88,
            text="TSMC manufactures advanced accelerated compute chips for NVIDIA.",
            payload={
                "document_id": "doc_nvda_10k",
                "available_time": "2024-02-01T10:00:00+00:00",
                "publication_date": "2024-02-01",
            },
        ),
        RetrievedChunk(
            chunk_id="chunk_vector_only_02",
            score=0.75,
            text="General risks include supply chain disruptions in East Asia.",
            payload={
                "document_id": "doc_nvda_10k",
                "available_time": "2024-02-01T10:00:00+00:00",
                "publication_date": "2024-02-01",
            },
        ),
    ]


@pytest.fixture
def mock_graph_evidence() -> list[GraphEvidence]:
    return [
        GraphEvidence(
            evidence_id="edge:e_tsm_supplies_nvda",
            source_node="company:TSM",
            target_node="company:NVDA",
            relationship="SUPPLIES",
            path=["company:TSM", "company:NVDA"],
            hop_count=1,
            text="TSMC (Company) --[SUPPLIES]--> NVIDIA (Company)",
            available_time="2024-02-01T10:00:00+00:00",
            confidence=0.95,
            document_id="doc_nvda_10k",
            chunk_id="chunk_shared_01",  # Overlaps with vector chunk
        ),
        GraphEvidence(
            evidence_id="edge:e_asml_supplies_tsm",
            source_node="company:ASML",
            target_node="company:TSM",
            relationship="SUPPLIES",
            path=["company:ASML", "company:TSM"],
            hop_count=1,
            text="ASML (Company) --[SUPPLIES]--> TSMC (Company)",
            available_time="2024-02-01T10:00:00+00:00",
            confidence=0.90,
            document_id="doc_tsm_20f",
            chunk_id="chunk_graph_only_03",  # Graph only
        ),
    ]


def test_hybrid_fuses_overlapping_evidence(mock_vector_chunks, mock_graph_evidence):
    mock_vector_store = MagicMock()
    mock_vector_store.search.return_value = mock_vector_chunks

    mock_graph_retriever = MagicMock()
    mock_graph_retriever.retrieve.return_value = mock_graph_evidence

    hybrid = HybridRetriever(
        vector_store=mock_vector_store,
        graph_retriever=mock_graph_retriever,
        rrf_k=60,
    )

    results = hybrid.retrieve("Who supplies Nvidia?", top_k=5)

    assert len(results) == 3

    # The overlapping item chunk_shared_01 should be top-ranked with source_type 'hybrid'
    top_item = results[0]
    assert top_item.chunk_id == "chunk_shared_01"
    assert top_item.source_type == "hybrid"
    assert set(top_item.retrieval_sources) == {"vector", "graph"}
    assert top_item.rank == 1

    # Overlapping item should have combined RRF score: 1/(60+1) + 1/(60+1) = 2/61 ~ 0.03278
    expected_rrf = (1.0 / (60 + 1)) + (1.0 / (60 + 1))
    assert top_item.score == pytest.approx(expected_rrf)


def test_hybrid_vector_only(mock_vector_chunks):
    mock_vector_store = MagicMock()
    mock_vector_store.search.return_value = mock_vector_chunks

    # No graph retriever
    hybrid = HybridRetriever(vector_store=mock_vector_store, graph_retriever=None)
    results = hybrid.retrieve("General risks", top_k=5)

    assert len(results) == 2
    assert all(r.source_type == "vector" for r in results)
    assert all(r.retrieval_sources == ["vector"] for r in results)


def test_hybrid_graph_only(mock_graph_evidence):
    mock_graph_retriever = MagicMock()
    mock_graph_retriever.retrieve.return_value = mock_graph_evidence

    # No vector store
    hybrid = HybridRetriever(vector_store=None, graph_retriever=mock_graph_retriever)
    results = hybrid.retrieve("Suppliers", top_k=5)

    assert len(results) == 2
    assert all(r.source_type == "graph" for r in results)
    assert all(r.retrieval_sources == ["graph"] for r in results)


def test_hybrid_temporal_filtering_passed_to_subretrievers(mock_vector_chunks, mock_graph_evidence):
    mock_vector_store = MagicMock()
    mock_vector_store.search.return_value = []

    mock_graph_retriever = MagicMock()
    mock_graph_retriever.retrieve.return_value = []

    hybrid = HybridRetriever(
        vector_store=mock_vector_store,
        graph_retriever=mock_graph_retriever,
    )

    cutoff = "2024-03-01T00:00:00+00:00"
    hybrid.retrieve("Query", reference_time=cutoff)

    # Check vector search received datetime
    assert mock_vector_store.search.called
    v_call_kwargs = mock_vector_store.search.call_args[1]
    assert isinstance(v_call_kwargs["reference_time"], datetime)
    assert v_call_kwargs["reference_time"].year == 2024
    assert v_call_kwargs["reference_time"].month == 3

    # Check graph retriever received reference_time
    assert mock_graph_retriever.retrieve.called
    g_call_kwargs = mock_graph_retriever.retrieve.call_args[1]
    assert isinstance(g_call_kwargs["reference_time"], datetime)
    assert g_call_kwargs["reference_time"].year == 2024
    assert g_call_kwargs["reference_time"].month == 3


def test_hybrid_determinism(mock_vector_chunks, mock_graph_evidence):
    mock_vector_store = MagicMock()
    mock_vector_store.search.return_value = mock_vector_chunks

    mock_graph_retriever = MagicMock()
    mock_graph_retriever.retrieve.return_value = mock_graph_evidence

    hybrid = HybridRetriever(
        vector_store=mock_vector_store,
        graph_retriever=mock_graph_retriever,
    )

    run1 = hybrid.retrieve("Who supplies Nvidia?", top_k=5)
    run2 = hybrid.retrieve("Who supplies Nvidia?", top_k=5)

    assert len(run1) == len(run2)
    assert [r.evidence_id for r in run1] == [r.evidence_id for r in run2]
    assert [r.score for r in run1] == [r.score for r in run2]


def test_future_graph_evidence_cannot_enter_rrf(mock_vector_chunks):
    """Future graph evidence must be filtered out before RRF rank fusion."""
    mock_vector_store = MagicMock()
    mock_vector_store.search.return_value = mock_vector_chunks

    future_graph_ev = [
        GraphEvidence(
            evidence_id="edge:future_graph_edge",
            source_node="company:NVDA",
            target_node="product:future_gpu",
            relationship="MANUFACTURES",
            path=["company:NVDA", "product:future_gpu"],
            hop_count=1,
            text="NVDA manufactures Future GPU",
            available_time="2026-08-01T00:00:00+00:00",  # Future!
            confidence=0.99,
        )
    ]
    mock_graph_retriever = MagicMock()
    mock_graph_retriever.retrieve.return_value = future_graph_ev

    hybrid = HybridRetriever(vector_store=mock_vector_store, graph_retriever=mock_graph_retriever)
    cutoff = "2024-03-01T00:00:00+00:00"
    results = hybrid.retrieve("Query", reference_time=cutoff)

    # Future graph evidence must NOT be present in results
    assert not any(r.evidence_id == "graph:edge:future_graph_edge" for r in results)
    assert not any("future_gpu" in r.text for r in results)
    assert all(r.source_type == "vector" for r in results)


def test_future_vector_evidence_cannot_enter_rrf(mock_graph_evidence):
    """Future vector chunks must be filtered out before RRF rank fusion."""
    future_vector_chunks = [
        RetrievedChunk(
            chunk_id="chunk_future_vector",
            score=0.95,
            text="Future earnings report leaked.",
            payload={"available_time": "2026-08-01T00:00:00+00:00"},
        )
    ]
    mock_vector_store = MagicMock()
    mock_vector_store.search.return_value = future_vector_chunks

    mock_graph_retriever = MagicMock()
    mock_graph_retriever.retrieve.return_value = mock_graph_evidence

    hybrid = HybridRetriever(vector_store=mock_vector_store, graph_retriever=mock_graph_retriever)
    cutoff = "2024-03-01T00:00:00+00:00"
    results = hybrid.retrieve("Query", reference_time=cutoff)

    # Future vector chunk must NOT be present in results
    assert not any(r.chunk_id == "chunk_future_vector" for r in results)
    assert not any("Future earnings" in r.text for r in results)


def test_rrf_receives_only_temporally_valid_candidates():
    """In mixed sets, only historical chunks/edges participate in RRF ranks."""
    mixed_vector = [
        RetrievedChunk(
            chunk_id="v_valid",
            score=0.85,
            text="Valid historical chunk",
            payload={"available_time": "2024-01-01T00:00:00+00:00"},
        ),
        RetrievedChunk(
            chunk_id="v_future",
            score=0.95,
            text="Future invalid chunk",
            payload={"available_time": "2025-01-01T00:00:00+00:00"},
        ),
    ]
    mixed_graph = [
        GraphEvidence(
            evidence_id="g_valid",
            source_node="company:A",
            target_node="company:B",
            relationship="SUPPLIES",
            path=["company:A", "company:B"],
            hop_count=1,
            text="A supplies B",
            available_time="2024-01-01T00:00:00+00:00",
            chunk_id="v_valid",  # Should fuse with v_valid
        ),
        GraphEvidence(
            evidence_id="g_future",
            source_node="company:A",
            target_node="company:C",
            relationship="ACQUIRED",
            path=["company:A", "company:C"],
            hop_count=1,
            text="A acquired C in future",
            available_time="2025-01-01T00:00:00+00:00",
            chunk_id="v_future",
        ),
    ]

    mock_v = MagicMock()
    mock_v.search.return_value = mixed_vector
    mock_g = MagicMock()
    mock_g.retrieve.return_value = mixed_graph

    hybrid = HybridRetriever(vector_store=mock_v, graph_retriever=mock_g)
    results = hybrid.retrieve("Query", reference_time="2024-06-01T00:00:00+00:00")

    # Exactly 1 item should survive (the valid one which is fused)
    assert len(results) == 1
    assert results[0].chunk_id == "v_valid"
    assert results[0].source_type == "hybrid"
    assert results[0].available_time <= "2024-06-01T00:00:00+00:00"
