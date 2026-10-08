from __future__ import annotations

import pytest

from src.retrieval.evidence import RetrievalEvidence
from src.retrieval.reranker import DeterministicReranker


@pytest.fixture
def reranker() -> DeterministicReranker:
    return DeterministicReranker()


def test_relationship_intent_match_ranks_above_unrelated(reranker):
    candidates = [
        RetrievalEvidence(
            evidence_id="ev_manufacture",
            source_type="graph",
            score=0.80,
            rank=1,
            text="NVIDIA manufactures Hopper GPUs",
            available_time="2024-01-01T00:00:00+00:00",
            metadata={"relationship": "MANUFACTURES", "hop_count": 1},
        ),
        RetrievalEvidence(
            evidence_id="ev_risk",
            source_type="graph",
            score=0.75,
            rank=2,
            text="NVIDIA faces regulatory scrutiny under the DMA",
            available_time="2024-01-01T00:00:00+00:00",
            metadata={"relationship": "AFFECTED_BY", "hop_count": 1},
        ),
    ]

    # Query asking specifically for risks
    reranked = reranker.rerank("What risks affect NVIDIA?", candidates)

    assert len(reranked) == 2
    # ev_risk should be promoted to rank 1 because AFFECTED_BY matches risk intent
    assert reranked[0].evidence_id == "ev_risk"
    assert reranked[1].evidence_id == "ev_manufacture"


def test_higher_confidence_ranks_appropriately(reranker):
    candidates = [
        RetrievalEvidence(
            evidence_id="ev_low_conf",
            source_type="graph",
            score=0.50,
            rank=1,
            text="Apple competes with Microsoft",
            available_time="2024-01-01T00:00:00+00:00",
            original_scores={"graph": 0.40},
            metadata={"relationship": "COMPETES_WITH", "hop_count": 1},
        ),
        RetrievalEvidence(
            evidence_id="ev_high_conf",
            source_type="graph",
            score=0.50,
            rank=2,
            text="Apple competes with Google",
            available_time="2024-01-01T00:00:00+00:00",
            original_scores={"graph": 0.99},
            metadata={"relationship": "COMPETES_WITH", "hop_count": 1},
        ),
    ]

    reranked = reranker.rerank("Who competes with Apple?", candidates)
    assert reranked[0].evidence_id == "ev_high_conf"


def test_hop_depth_penalty(reranker):
    candidates = [
        RetrievalEvidence(
            evidence_id="ev_3_hop",
            source_type="graph",
            score=0.60,
            rank=1,
            text="A -> B -> C -> TSMC",
            available_time="2024-01-01T00:00:00+00:00",
            metadata={"relationship": "SUPPLIES -> SUPPLIES -> SUPPLIES", "hop_count": 3},
        ),
        RetrievalEvidence(
            evidence_id="ev_1_hop",
            source_type="graph",
            score=0.60,
            rank=2,
            text="TSMC supplies NVIDIA",
            available_time="2024-01-01T00:00:00+00:00",
            metadata={"relationship": "SUPPLIES", "hop_count": 1},
        ),
    ]

    reranked = reranker.rerank("Who supplies NVIDIA?", candidates)
    # Direct 1-hop should be preferred over distant 3-hop
    assert reranked[0].evidence_id == "ev_1_hop"


def test_temporal_invalid_evidence_is_strictly_rejected(reranker):
    candidates = [
        RetrievalEvidence(
            evidence_id="ev_historical",
            source_type="graph",
            score=0.70,
            rank=1,
            text="Historical filing info",
            available_time="2024-01-15T00:00:00+00:00",
            metadata={"hop_count": 1},
        ),
        RetrievalEvidence(
            evidence_id="ev_future",
            source_type="graph",
            score=0.99,  # High score should NOT save future evidence!
            rank=2,
            text="Future earnings leaked in late 2025",
            available_time="2025-11-01T00:00:00+00:00",
            metadata={"hop_count": 1},
        ),
    ]

    cutoff = "2024-06-01T00:00:00+00:00"
    reranked = reranker.rerank("Info query", candidates, reference_time=cutoff)

    # Future evidence must be completely excluded from the result list
    assert len(reranked) == 1
    assert reranked[0].evidence_id == "ev_historical"
    assert not any(r.evidence_id == "ev_future" for r in reranked)
