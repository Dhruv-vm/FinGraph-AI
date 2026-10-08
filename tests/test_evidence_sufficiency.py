from __future__ import annotations

import pytest

from src.qa.evidence_sufficiency import EvidenceSufficiencyChecker
from src.qa.query_analysis import QueryAnalysis
from src.retrieval.evidence import RetrievalEvidence


@pytest.fixture
def checker() -> EvidenceSufficiencyChecker:
    return EvidenceSufficiencyChecker()


def test_sufficient_evidence_returns_true(checker):
    analysis = QueryAnalysis(
        original_query="What company supplies NVIDIA?",
        entities=["company:NVDA"],
        relationship_intent=["SUPPLIES"],
        expected_hop_depth=1,
        is_multi_hop=False,
    )

    evidence = [
        RetrievalEvidence(
            evidence_id="ev_tsm",
            source_type="graph",
            score=0.85,
            rank=1,
            text="TSMC (Company) --[SUPPLIES]--> NVIDIA (Company)",
            document_id="doc_123",
            chunk_id="chunk_456",
            available_time="2024-01-01T00:00:00+00:00",
            metadata={"relationship": "SUPPLIES", "hop_count": 1},
        )
    ]

    result = checker.check(analysis, evidence)
    assert result.sufficient is True
    assert len(result.missing_entities) == 0
    assert len(result.missing_relationships) == 0


def test_missing_entity_returns_insufficient(checker):
    analysis = QueryAnalysis(
        original_query="Does TSMC supply Apple and Microsoft?",
        entities=["company:AAPL", "company:MSFT"],
        relationship_intent=["SUPPLIES"],
        expected_hop_depth=1,
        is_multi_hop=False,
    )

    # Evidence only mentions Apple, completely missing Microsoft
    evidence = [
        RetrievalEvidence(
            evidence_id="ev_apple",
            source_type="graph",
            score=0.80,
            rank=1,
            text="TSMC supplies Apple Inc.",
            document_id="doc_123",
            chunk_id="chunk_1",
            available_time="2024-01-01T00:00:00+00:00",
            metadata={"relationship": "SUPPLIES", "hop_count": 1},
        )
    ]

    result = checker.check(analysis, evidence)
    assert result.sufficient is False
    assert "company:MSFT" in result.missing_entities


def test_missing_relationship_returns_insufficient(checker):
    analysis = QueryAnalysis(
        original_query="What risks affect NVIDIA?",
        entities=["company:NVDA"],
        relationship_intent=["HAS_RISK", "AFFECTED_BY"],
        expected_hop_depth=1,
        is_multi_hop=False,
    )

    # Evidence only contains MANUFACTURES, no risk relationships
    evidence = [
        RetrievalEvidence(
            evidence_id="ev_mfg",
            source_type="graph",
            score=0.90,
            rank=1,
            text="NVIDIA manufactures GeForce GPUs",
            document_id="doc_123",
            chunk_id="chunk_2",
            available_time="2024-01-01T00:00:00+00:00",
            metadata={"relationship": "MANUFACTURES", "hop_count": 1},
        )
    ]

    result = checker.check(analysis, evidence)
    assert result.sufficient is False
    assert "HAS_RISK" in result.missing_relationships or "AFFECTED_BY" in result.missing_relationships


def test_missing_hop_coverage_returns_insufficient(checker):
    analysis = QueryAnalysis(
        original_query="Which supplier connects ASML to NVIDIA indirectly?",
        entities=["company:NVDA"],
        relationship_intent=["SUPPLIES"],
        expected_hop_depth=2,
        is_multi_hop=True,
    )

    # Evidence only contains 1-hop direct relationship
    evidence = [
        RetrievalEvidence(
            evidence_id="ev_direct",
            source_type="graph",
            score=0.75,
            rank=1,
            text="TSMC supplies NVIDIA",
            document_id="doc_123",
            chunk_id="chunk_3",
            available_time="2024-01-01T00:00:00+00:00",
            metadata={"relationship": "SUPPLIES", "hop_count": 1},
        )
    ]

    result = checker.check(analysis, evidence)
    assert result.sufficient is False
    assert result.observed_hops == 1
    assert result.required_hops == 2
    assert "Insufficient multi-hop reasoning depth" in result.reason
