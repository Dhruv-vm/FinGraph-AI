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


def test_soft_hop_mode_allows_single_hop():
    soft_checker = EvidenceSufficiencyChecker(hop_mode="soft")
    analysis = QueryAnalysis(
        original_query="Which supplier connects ASML to NVIDIA indirectly?",
        entities=["company:NVDA"],
        relationship_intent=["SUPPLIES"],
        expected_hop_depth=2,
        is_multi_hop=True,
    )
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
    result = soft_checker.check(analysis, evidence)
    assert result.sufficient is True


def test_partial_entity_mode_allows_single_entity_coverage():
    partial_checker = EvidenceSufficiencyChecker(entity_mode="partial")
    analysis = QueryAnalysis(
        original_query="Does TSMC supply Apple and Microsoft?",
        entities=["company:AAPL", "company:MSFT"],
        relationship_intent=["SUPPLIES"],
        expected_hop_depth=1,
        is_multi_hop=False,
    )
    # Evidence only covers Apple, not Microsoft
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
    result = partial_checker.check(analysis, evidence)
    assert result.sufficient is True


def test_disabled_gate_always_sufficient():
    disabled_checker = EvidenceSufficiencyChecker(gate_enabled=False)
    analysis = QueryAnalysis(
        original_query="What unknown relationship connects unrelated entities?",
        entities=["company:UNKNOWN1", "company:UNKNOWN2"],
        relationship_intent=["UNRELATED_REL"],
        expected_hop_depth=3,
        is_multi_hop=True,
    )
    evidence = [
        RetrievalEvidence(
            evidence_id="ev_generic",
            source_type="vector",
            score=0.50,
            rank=1,
            text="Generic SEC disclosure snippet",
            document_id="doc_123",
            chunk_id="chunk_1",
            available_time="2024-01-01T00:00:00+00:00",
            metadata={},
        )
    ]
    result = disabled_checker.check(analysis, evidence)
    assert result.sufficient is True


def test_disabled_gate_preserves_temporal_safety():
    """Disabled gate must still reject evidence violating point-in-time constraints."""
    disabled_checker = EvidenceSufficiencyChecker(gate_enabled=False)
    analysis = QueryAnalysis(
        original_query="What occurred before 2024?",
        entities=["company:NVDA"],
        relationship_intent=[],
        expected_hop_depth=1,
        is_multi_hop=False,
        reference_time="2024-01-01T00:00:00Z",
    )
    # Evidence available in 2025 (future leakage)
    evidence = [
        RetrievalEvidence(
            evidence_id="ev_future",
            source_type="vector",
            score=0.90,
            rank=1,
            text="NVIDIA 2025 future disclosure",
            document_id="doc_123",
            chunk_id="chunk_1",
            available_time="2025-06-01T00:00:00Z",
            metadata={},
        )
    ]
    result = disabled_checker.check(analysis, evidence)
    assert result.sufficient is False
    assert result.temporal_valid is False
    assert "temporal violations" in result.reason.lower()


def test_disabled_gate_preserves_provenance_validation():
    """Disabled gate must still reject evidence lacking document provenance."""
    disabled_checker = EvidenceSufficiencyChecker(gate_enabled=False)
    analysis = QueryAnalysis(
        original_query="What happened?",
        entities=["company:NVDA"],
        relationship_intent=[],
        expected_hop_depth=1,
        is_multi_hop=False,
    )
    evidence = [
        RetrievalEvidence(
            evidence_id="ev_no_prov",
            source_type="vector",
            score=0.90,
            rank=1,
            text="Disembodied text snippet with no SEC provenance",
            document_id=None,
            chunk_id=None,
            available_time=None,
            metadata={},
        )
    ]
    result = disabled_checker.check(analysis, evidence)
    assert result.sufficient is False
    assert result.provenance_complete is False
    assert "provenance" in result.reason.lower()


def test_formal_edge_only_evidence_passes_strict_and_textual():
    """When formal graph edge metadata is present, both strict and textual modes succeed."""
    strict_checker = EvidenceSufficiencyChecker(relationship_mode="strict")
    textual_checker = EvidenceSufficiencyChecker(relationship_mode="textual")
    analysis = QueryAnalysis(
        original_query="What products does NVIDIA manufacture?",
        entities=["company:NVDA"],
        relationship_intent=["MANUFACTURES"],
        expected_hop_depth=1,
        is_multi_hop=False,
    )
    # Text does not contain the word 'manufacture', but metadata contains formal graph edge
    evidence = [
        RetrievalEvidence(
            evidence_id="ev_formal",
            source_type="graph",
            score=0.88,
            rank=1,
            text="NVIDIA GeForce RTX 4090 hardware specifications and compute units.",
            document_id="doc_123",
            chunk_id="chunk_1",
            available_time="2024-01-01T00:00:00Z",
            metadata={"relationship": "MANUFACTURES", "hop_count": 1, "path": ["company:NVDA", "GeForce RTX 4090"]},
        )
    ]
    res_strict = strict_checker.check(analysis, evidence)
    res_textual = textual_checker.check(analysis, evidence)
    assert res_strict.sufficient is True
    assert res_textual.sufficient is True


def test_text_only_evidence_fails_strict_but_passes_textual():
    """When relation is only described in text without formal graph edge metadata:
    Strict mode fails (rejects substring match), textual mode passes."""
    strict_checker = EvidenceSufficiencyChecker(relationship_mode="strict")
    textual_checker = EvidenceSufficiencyChecker(relationship_mode="textual")
    analysis = QueryAnalysis(
        original_query="Who supplies key parts to NVIDIA?",
        entities=["company:NVDA"],
        relationship_intent=["SUPPLIES"],
        expected_hop_depth=1,
        is_multi_hop=False,
    )
    # Text describes supplier relation, but metadata has NO formal edge
    evidence = [
        RetrievalEvidence(
            evidence_id="ev_text_only",
            source_type="vector",
            score=0.85,
            rank=1,
            text="TSMC serves as the primary foundry vendor supplying semiconductor wafers to NVIDIA.",
            document_id="doc_123",
            chunk_id="chunk_1",
            available_time="2024-01-01T00:00:00Z",
            metadata={"hop_count": 1},
        )
    ]
    res_strict = strict_checker.check(analysis, evidence)
    res_textual = textual_checker.check(analysis, evidence)
    assert res_strict.sufficient is False
    assert "SUPPLIES" in res_strict.missing_relationships
    assert res_textual.sufficient is True
    assert len(res_textual.missing_relationships) == 0


def test_missing_relationships_fail_both_modes():
    """When neither formal edges nor textual keywords match, both modes reject."""
    strict_checker = EvidenceSufficiencyChecker(relationship_mode="strict")
    textual_checker = EvidenceSufficiencyChecker(relationship_mode="textual")
    analysis = QueryAnalysis(
        original_query="What risks affect NVIDIA?",
        entities=["company:NVDA"],
        relationship_intent=["HAS_RISK"],
        expected_hop_depth=1,
        is_multi_hop=False,
    )
    evidence = [
        RetrievalEvidence(
            evidence_id="ev_unrelated",
            source_type="vector",
            score=0.70,
            rank=1,
            text="NVIDIA is headquartered in Santa Clara, California and employs thousands of engineers.",
            document_id="doc_123",
            chunk_id="chunk_1",
            available_time="2024-01-01T00:00:00Z",
            metadata={"relationship": "LOCATED_IN", "hop_count": 1},
        )
    ]
    res_strict = strict_checker.check(analysis, evidence)
    res_textual = textual_checker.check(analysis, evidence)
    assert res_strict.sufficient is False
    assert "HAS_RISK" in res_strict.missing_relationships
    assert res_textual.sufficient is False
    assert "HAS_RISK" in res_textual.missing_relationships


def test_multi_hop_supplier_risk_compound_chain_strict_vs_textual():
    """In multi-hop supplier-risk queries, strict mode requires formal graph edges for both
    SUPPLIES and risk (HAS_RISK / AFFECTED_BY), whereas textual mode allows textual mentions."""
    strict_checker = EvidenceSufficiencyChecker(
        relationship_mode="strict", hop_mode="soft", entity_mode="strict"
    )
    textual_checker = EvidenceSufficiencyChecker(
        relationship_mode="textual", hop_mode="soft", entity_mode="strict"
    )
    analysis = QueryAnalysis(
        original_query="What internal manufacturing risks affect the suppliers that supply Apple?",
        entities=["company:AAPL"],
        relationship_intent=["SUPPLIES", "HAS_RISK"],
        expected_hop_depth=2,
        is_multi_hop=True,
    )
    # Evidence has formal SUPPLIES edge, but risk is only in text (no formal HAS_RISK / AFFECTED_BY edge)
    evidence = [
        RetrievalEvidence(
            evidence_id="ev_supp",
            source_type="graph",
            score=0.85,
            rank=1,
            text="Foxconn supplies assembly services to Apple Inc. Operations face severe equipment risk and flood disruptions.",
            document_id="doc_123",
            chunk_id="chunk_1",
            available_time="2024-01-01T00:00:00Z",
            metadata={"relationship": "SUPPLIES", "hop_count": 1},
        )
    ]
    res_strict = strict_checker.check(analysis, evidence)
    res_textual = textual_checker.check(analysis, evidence)
    assert res_strict.sufficient is False
    assert "compound relationship" in res_strict.reason.lower()
    assert res_textual.sufficient is True


def test_compound_chain_valid_connected_accepted():
    """Scenario 1: Correct directed supplier-to-company relationship and supplier's risk
    relationship are present and connected. Strict mode accepts the evidence."""
    checker = EvidenceSufficiencyChecker(relationship_mode="strict", hop_mode="soft", entity_mode="strict")
    analysis = QueryAnalysis(
        original_query="What internal manufacturing risks affect the suppliers that supply Apple?",
        entities=["company:AAPL"],
        relationship_intent=["SUPPLIES", "HAS_RISK"],
        expected_hop_depth=2,
        is_multi_hop=True,
    )
    evidence = [
        RetrievalEvidence(
            evidence_id="ev_supp",
            source_type="graph",
            score=0.90,
            rank=1,
            text="Foxconn supplies assembly components to Apple Inc.",
            document_id="doc_sec_01",
            chunk_id="chunk_01",
            available_time="2024-01-01T00:00:00Z",
            metadata={
                "graph_edges": [
                    {"source": "company:FOXCONN", "relationship": "SUPPLIES", "target": "company:AAPL"}
                ],
                "hop_count": 1,
            },
        ),
        RetrievalEvidence(
            evidence_id="ev_risk",
            source_type="graph",
            score=0.85,
            rank=2,
            text="Foxconn operations are exposed to internal facility fire and equipment risk.",
            document_id="doc_sec_02",
            chunk_id="chunk_02",
            available_time="2024-01-01T00:00:00Z",
            metadata={
                "graph_edges": [
                    {"source": "company:FOXCONN", "relationship": "HAS_RISK", "target": "risk:EQUIPMENT_FAILURE"}
                ],
                "hop_count": 1,
            },
        ),
    ]
    res = checker.check(analysis, evidence)
    assert res.sufficient is True


def test_compound_chain_disconnected_entities_rejected():
    """Scenario 2: Valid SUPPLIES edge and valid risk edge exist, but refer to unrelated entities.
    Strict mode rejects them."""
    checker = EvidenceSufficiencyChecker(relationship_mode="strict", hop_mode="soft", entity_mode="strict")
    analysis = QueryAnalysis(
        original_query="What internal manufacturing risks affect the suppliers that supply Apple?",
        entities=["company:AAPL"],
        relationship_intent=["SUPPLIES", "HAS_RISK"],
        expected_hop_depth=2,
        is_multi_hop=True,
    )
    evidence = [
        RetrievalEvidence(
            evidence_id="ev_supp",
            source_type="graph",
            score=0.90,
            rank=1,
            text="Foxconn supplies assembly components to Apple Inc.",
            document_id="doc_sec_01",
            chunk_id="chunk_01",
            available_time="2024-01-01T00:00:00Z",
            metadata={
                "graph_edges": [
                    {"source": "company:FOXCONN", "relationship": "SUPPLIES", "target": "company:AAPL"}
                ],
            },
        ),
        RetrievalEvidence(
            evidence_id="ev_risk",
            source_type="graph",
            score=0.85,
            rank=2,
            text="Tesla faces battery manufacturing fire risk.",
            document_id="doc_sec_02",
            chunk_id="chunk_02",
            available_time="2024-01-01T00:00:00Z",
            metadata={
                "graph_edges": [
                    {"source": "company:TSLA", "relationship": "HAS_RISK", "target": "risk:BATTERY_FIRE"}
                ],
            },
        ),
    ]
    res = checker.check(analysis, evidence)
    assert res.sufficient is False
    assert "compound relationship" in res.reason.lower()
    assert "disconnected" in res.reason.lower()


def test_compound_chain_wrong_direction_rejected():
    """Scenario 3: Relationships exist but direction is reversed (Apple supplies Foxconn).
    Strict mode rejects it."""
    checker = EvidenceSufficiencyChecker(relationship_mode="strict", hop_mode="soft", entity_mode="strict")
    analysis = QueryAnalysis(
        original_query="What internal manufacturing risks affect the suppliers that supply Apple?",
        entities=["company:AAPL"],
        relationship_intent=["SUPPLIES", "HAS_RISK"],
        expected_hop_depth=2,
        is_multi_hop=True,
    )
    evidence = [
        RetrievalEvidence(
            evidence_id="ev_rev_supp",
            source_type="graph",
            score=0.90,
            rank=1,
            text="Apple Inc. supplies custom silicon designs to Foxconn.",
            document_id="doc_sec_01",
            chunk_id="chunk_01",
            available_time="2024-01-01T00:00:00Z",
            metadata={
                "graph_edges": [
                    {"source": "company:AAPL", "relationship": "SUPPLIES", "target": "company:FOXCONN"}
                ],
            },
        ),
        RetrievalEvidence(
            evidence_id="ev_risk",
            source_type="graph",
            score=0.85,
            rank=2,
            text="Foxconn operations are exposed to facility risks.",
            document_id="doc_sec_02",
            chunk_id="chunk_02",
            available_time="2024-01-01T00:00:00Z",
            metadata={
                "graph_edges": [
                    {"source": "company:FOXCONN", "relationship": "HAS_RISK", "target": "risk:FLOOD"}
                ],
            },
        ),
    ]
    res = checker.check(analysis, evidence)
    assert res.sufficient is False
    assert "compound relationship" in res.reason.lower()


def test_compound_chain_missing_endpoints_rejected():
    """Scenario 4: Relationship labels exist without sufficient structured endpoints.
    Strict mode rejects them."""
    checker = EvidenceSufficiencyChecker(relationship_mode="strict", hop_mode="soft", entity_mode="strict")
    analysis = QueryAnalysis(
        original_query="What internal manufacturing risks affect the suppliers that supply Apple?",
        entities=["company:AAPL"],
        relationship_intent=["SUPPLIES", "HAS_RISK"],
        expected_hop_depth=2,
        is_multi_hop=True,
    )
    evidence = [
        RetrievalEvidence(
            evidence_id="ev_malformed",
            source_type="graph",
            score=0.80,
            rank=1,
            text="Supply contracts and associated business risks for Apple Inc.",
            document_id="doc_sec_01",
            chunk_id="chunk_01",
            available_time="2024-01-01T00:00:00Z",
            metadata={
                "relationship": "SUPPLIES",
                "graph_relations": ["SUPPLIES", "HAS_RISK"],
            },
        )
    ]
    res = checker.check(analysis, evidence)
    assert res.sufficient is False
    assert "compound relationship" in res.reason.lower()


def test_compound_chain_unrelated_evidence_records_rejected():
    """Scenario 6: SUPPLIES is in one result and risk relationship is in another unrelated result.
    Strict mode rejects them."""
    checker = EvidenceSufficiencyChecker(relationship_mode="strict", hop_mode="soft", entity_mode="strict")
    analysis = QueryAnalysis(
        original_query="What internal manufacturing risks affect the suppliers that supply Apple?",
        entities=["company:AAPL"],
        relationship_intent=["SUPPLIES", "HAS_RISK"],
        expected_hop_depth=2,
        is_multi_hop=True,
    )
    evidence = [
        RetrievalEvidence(
            evidence_id="ev_supp_aapl",
            source_type="graph",
            score=0.90,
            rank=1,
            text="Foxconn supplies Apple.",
            document_id="doc_sec_01",
            chunk_id="chunk_01",
            available_time="2024-01-01T00:00:00Z",
            metadata={
                "graph_edges": [
                    {"source": "company:FOXCONN", "relationship": "SUPPLIES", "target": "company:AAPL"}
                ],
            },
        ),
        RetrievalEvidence(
            evidence_id="ev_risk_boeing",
            source_type="graph",
            score=0.80,
            rank=2,
            text="Boeing suffers from labor strikes.",
            document_id="doc_sec_02",
            chunk_id="chunk_02",
            available_time="2024-01-01T00:00:00Z",
            metadata={
                "graph_edges": [
                    {"source": "company:BA", "relationship": "HAS_RISK", "target": "risk:STRIKE"}
                ],
            },
        ),
    ]
    res = checker.check(analysis, evidence)
    assert res.sufficient is False
    assert "compound relationship" in res.reason.lower()
    assert "disconnected" in res.reason.lower()


def test_compound_chain_connected_multi_edge_path_accepted():
    """Scenario 7: A valid multi-edge graph path is preserved and evaluated correctly."""
    checker = EvidenceSufficiencyChecker(relationship_mode="strict", hop_mode="soft", entity_mode="strict")
    analysis = QueryAnalysis(
        original_query="What internal manufacturing risks affect the suppliers that supply Apple?",
        entities=["company:AAPL"],
        relationship_intent=["SUPPLIES", "HAS_RISK"],
        expected_hop_depth=2,
        is_multi_hop=True,
    )
    evidence = [
        RetrievalEvidence(
            evidence_id="ev_path",
            source_type="graph",
            score=0.92,
            rank=1,
            text="Foxconn supplies Apple Inc. -> Foxconn faces severe facility flood risk.",
            document_id="doc_sec_01",
            chunk_id="chunk_01",
            available_time="2024-01-01T00:00:00Z",
            metadata={
                "graph_edge_paths": [
                    [
                        {"source": "company:FOXCONN", "relationship": "SUPPLIES", "target": "company:AAPL"},
                        {"source": "company:FOXCONN", "relationship": "HAS_RISK", "target": "risk:FLOOD"},
                    ]
                ],
                "hop_count": 2,
            },
        )
    ]
    res = checker.check(analysis, evidence)
    assert res.sufficient is True


def test_backward_compatibility_fixtures_without_graph_edges():
    """Scenario 10: Manually constructed legacy fixtures without graph_edges work for ordinary
    queries but cannot falsely pass the strict compound gate."""
    strict_checker = EvidenceSufficiencyChecker(relationship_mode="strict", hop_mode="soft", entity_mode="strict")
    # 1. Non-compound single-hop query: works with legacy metadata
    single_hop_analysis = QueryAnalysis(
        original_query="Who supplies Apple?",
        entities=["company:AAPL"],
        relationship_intent=["SUPPLIES"],
        expected_hop_depth=1,
        is_multi_hop=False,
    )
    legacy_evidence = [
        RetrievalEvidence(
            evidence_id="ev_legacy",
            source_type="graph",
            score=0.88,
            rank=1,
            text="Foxconn supplies Apple.",
            document_id="doc_sec_01",
            chunk_id="chunk_01",
            available_time="2024-01-01T00:00:00Z",
            metadata={"relationship": "SUPPLIES", "hop_count": 1},
        )
    ]
    single_res = strict_checker.check(single_hop_analysis, legacy_evidence)
    assert single_res.sufficient is True

    # 2. Compound query: legacy fixture lacking graph_edges fails closed
    compound_analysis = QueryAnalysis(
        original_query="What internal manufacturing risks affect the suppliers that supply Apple?",
        entities=["company:AAPL"],
        relationship_intent=["SUPPLIES", "HAS_RISK"],
        expected_hop_depth=2,
        is_multi_hop=True,
    )
    compound_res = strict_checker.check(compound_analysis, legacy_evidence)
    assert compound_res.sufficient is False
    assert "compound relationship" in compound_res.reason.lower()


def test_ordinary_single_relation_query_unchanged():
    """Scenario 12: Ordinary single-relation queries remain completely unaffected."""
    checker = EvidenceSufficiencyChecker(relationship_mode="strict", hop_mode="strict", entity_mode="strict")
    analysis = QueryAnalysis(
        original_query="What products does Apple manufacture?",
        entities=["company:AAPL"],
        relationship_intent=["MANUFACTURES"],
        expected_hop_depth=1,
        is_multi_hop=False,
    )
    evidence = [
        RetrievalEvidence(
            evidence_id="ev_mfg",
            source_type="graph",
            score=0.95,
            rank=1,
            text="Apple Inc. manufactures iPhone and Mac computers.",
            document_id="doc_sec_01",
            chunk_id="chunk_01",
            available_time="2024-01-01T00:00:00Z",
            metadata={
                "relationship": "MANUFACTURES",
                "hop_count": 1,
                "graph_edges": [
                    {"source": "company:AAPL", "relationship": "MANUFACTURES", "target": "product:IPHONE"}
                ],
            },
        )
    ]
    res = checker.check(analysis, evidence)
    assert res.sufficient is True
