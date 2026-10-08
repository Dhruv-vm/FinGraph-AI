from __future__ import annotations

import pytest

from src.qa.agent import QAEvidencePackage
from src.qa.evidence_sufficiency import SufficiencyResult
from src.qa.validation import AnswerValidator
from src.retrieval.evidence import RetrievalEvidence


def make_evidence(eid: str, time: str = "2024-01-01T00:00:00+00:00") -> RetrievalEvidence:
    return RetrievalEvidence(
        evidence_id=eid,
        source_type="graph",
        score=0.9,
        rank=1,
        text=f"Text for {eid}",
        document_id="doc_1",
        chunk_id="chunk_1",
        available_time=time,
        metadata={"hop_count": 1},
    )


def make_package(
    evidence: list[RetrievalEvidence],
    ref_time: str | None = "2024-06-01T00:00:00+00:00",
    sufficient: bool = True,
) -> QAEvidencePackage:
    return QAEvidencePackage(
        query="What chip did NVIDIA release?",
        reference_time=ref_time,
        question_type="factual",
        expected_hops=1,
        evidence=evidence,
        graph_paths=[],
        sources=["doc_1"],
        retrieval_rounds=1,
        temporal_valid=True,
        sufficient=sufficient,
        sufficiency_result=SufficiencyResult(sufficient=sufficient, reason="Test"),
    )


def test_validator_accepts_valid_answer():
    ev = make_evidence("ev_1")
    pkg = make_package([ev])
    validator = AnswerValidator()

    payload = {
        "answer": "NVIDIA released the chip.",
        "confidence": 0.95,
        "evidence_ids": ["ev_1"],
        "reasoning_summary": "Supported by evidence ev_1.",
    }
    result = validator.validate(payload, pkg)
    assert result.is_valid is True
    assert len(result.errors) == 0
    assert result.temporal_valid is True
    assert result.provenance_complete is True


def test_validator_rejects_unknown_evidence_id():
    ev = make_evidence("ev_1")
    pkg = make_package([ev])
    validator = AnswerValidator()

    payload = {
        "answer": "Hallucinated claim.",
        "confidence": 0.9,
        "evidence_ids": ["ev_unknown_999"],
    }
    result = validator.validate(payload, pkg)
    assert result.is_valid is False
    assert any("Unknown evidence IDs" in err for err in result.errors)


def test_validator_rejects_missing_citation_for_factual_claims():
    ev = make_evidence("ev_1")
    pkg = make_package([ev])
    validator = AnswerValidator()

    payload = {
        "answer": "Factual answer without evidence citations.",
        "confidence": 0.85,
        "evidence_ids": [],
    }
    result = validator.validate(payload, pkg)
    assert result.is_valid is False
    assert any("must cite at least one valid evidence ID" in err for err in result.errors)


def test_validator_accepts_empty_citations_for_refusal():
    pkg = make_package([], sufficient=False)
    validator = AnswerValidator()

    payload = {
        "answer": "Insufficient evidence in the retrieved corpus.",
        "confidence": 0.0,
        "evidence_ids": [],
        "insufficient_evidence": True,
    }
    result = validator.validate(payload, pkg)
    assert result.is_valid is True
    assert len(result.errors) == 0


def test_validator_rejects_future_evidence_citation():
    # Package has past evidence, but also someone passed a future item citation
    ev_future = make_evidence("ev_future", time="2025-01-01T00:00:00+00:00")
    pkg = make_package([ev_future], ref_time="2024-06-01T00:00:00+00:00")
    validator = AnswerValidator()

    payload = {
        "answer": "Future fact claim.",
        "confidence": 0.9,
        "evidence_ids": ["ev_future"],
    }
    result = validator.validate(payload, pkg)
    assert result.is_valid is False
    assert result.temporal_valid is False
    assert any("violates reference cutoff" in err for err in result.errors)


def test_validator_rejects_invalid_confidence():
    ev = make_evidence("ev_1")
    pkg = make_package([ev])
    validator = AnswerValidator()

    payload = {
        "answer": "Test answer.",
        "confidence": 1.5,  # Out of bounds
        "evidence_ids": ["ev_1"],
    }
    result = validator.validate(payload, pkg)
    assert result.is_valid is False
    assert any("out of bounds" in err for err in result.errors)


def test_validator_rejects_missing_required_keys():
    ev = make_evidence("ev_1")
    pkg = make_package([ev])
    validator = AnswerValidator()

    payload = {
        "answer": "Missing confidence and evidence_ids.",
    }
    result = validator.validate(payload, pkg)
    assert result.is_valid is False
    assert any("Missing required fields" in err for err in result.errors)


def test_validator_parses_json_string_properly():
    ev = make_evidence("ev_1")
    pkg = make_package([ev])
    validator = AnswerValidator()

    json_str = '{"answer": "Parsed correctly", "confidence": 0.9, "evidence_ids": ["ev_1"]}'
    result = validator.validate(json_str, pkg)
    assert result.is_valid is True
    assert result.cited_evidence_ids == ["ev_1"]


def test_validator_rejects_malformed_json_string():
    ev = make_evidence("ev_1")
    pkg = make_package([ev])
    validator = AnswerValidator()

    result = validator.validate("NOT A JSON STRING", pkg)
    assert result.is_valid is False
    assert any("not valid JSON" in err for err in result.errors)


def test_q6_does_not_answer_supplier_only_question():
    """Verify that a supplier-only answer is rejected when the query asks for supplier risks (Q6)."""
    # Evidence only contains SUPPLIES relationship, without any risk relations
    ev = RetrievalEvidence(
        evidence_id="graph:edge:company:NVDA->SUPPLIES->company:Samsung",
        source_type="graph",
        score=0.9,
        rank=1,
        text="NVIDIA Corporation --[SUPPLIES]--> Samsung Electronics Co., Ltd.",
        document_id="doc_1",
        chunk_id="chunk_1",
        available_time="2026-02-25T00:00:00+00:00",
        metadata={"relationship": "SUPPLIES", "hop_count": 2},
    )
    from src.qa.query_analysis import QueryAnalysis
    qa = QueryAnalysis(
        original_query="What risks affect companies that supply NVIDIA?",
        entities=["company:NVDA"],
        unresolved_entities=[],
        relationship_intent=["AFFECTED_BY", "HAS_RISK", "SUPPLIES"],
        question_type="multi-hop",
        secondary_question_types=[],
        reference_time=None,
        temporal_window=None,
        expected_hop_depth=2,
        is_multi_hop=True,
        cross_document_required=True,
    )
    pkg = QAEvidencePackage(
        query="What risks affect companies that supply NVIDIA?",
        reference_time=None,
        question_type="multi-hop",
        expected_hops=2,
        evidence=[ev],
        graph_paths=[],
        sources=["doc_1"],
        retrieval_rounds=1,
        temporal_valid=True,
        sufficient=False,
        analysis=qa,
    )

    validator = AnswerValidator()
    # Attempting to return a supplier-only answer without risk coverage must fail validation
    payload = {
        "answer": "Companies that supply NVIDIA include Samsung Electronics Co., Ltd.",
        "confidence": 0.85,
        "evidence_ids": ["graph:edge:company:NVDA->SUPPLIES->company:Samsung"],
    }
    result = validator.validate(payload, pkg)
    assert result.is_valid is False
    assert any("Semantic misalignment" in err for err in result.errors)


def test_dependency_direction_is_preserved():
    """Verify that reverse supply relationships (A supplies B) cannot support A depending on B."""
    ev_reverse = RetrievalEvidence(
        evidence_id="graph:edge:company:NVDA->SUPPLIES->company:CustomerB",
        source_type="graph",
        score=0.9,
        rank=1,
        text="NVIDIA Corporation --[SUPPLIES]--> Customer B",
        document_id="doc_1",
        chunk_id="chunk_1",
        available_time="2025-08-27T00:00:00+00:00",
        metadata={"relationship": "SUPPLIES", "hop_count": 1},
    )
    from src.qa.query_analysis import QueryAnalysis
    qa = QueryAnalysis(
        original_query="What companies does NVIDIA depend on?",
        entities=["company:NVDA"],
        unresolved_entities=[],
        relationship_intent=["DEPENDS_ON", "SUPPLIES"],
        question_type="relationship",
        secondary_question_types=[],
        reference_time=None,
        temporal_window=None,
        expected_hop_depth=1,
        is_multi_hop=False,
        cross_document_required=False,
    )
    pkg = QAEvidencePackage(
        query="What companies does NVIDIA depend on?",
        reference_time=None,
        question_type="relationship",
        expected_hops=1,
        evidence=[ev_reverse],
        graph_paths=[],
        sources=["doc_1"],
        retrieval_rounds=1,
        temporal_valid=True,
        sufficient=True,
        analysis=qa,
    )

    validator = AnswerValidator()
    payload = {
        "answer": "NVIDIA depends on Customer B.",
        "confidence": 0.9,
        "evidence_ids": ["graph:edge:company:NVDA->SUPPLIES->company:CustomerB"],
    }
    result = validator.validate(payload, pkg)
    assert result.is_valid is False
    assert any("directionality violation" in err for err in result.errors)


def test_missing_risk_chain_causes_refusal():
    """Verify that if the risk chain is missing for supplier-risk questions, sufficiency checker flags it as insufficient."""
    from src.qa.evidence_sufficiency import EvidenceSufficiencyChecker
    from src.qa.query_analysis import QueryAnalysis

    checker = EvidenceSufficiencyChecker()
    qa = QueryAnalysis(
        original_query="What risks affect companies that supply NVIDIA?",
        entities=["company:NVDA"],
        unresolved_entities=[],
        relationship_intent=["AFFECTED_BY", "HAS_RISK", "SUPPLIES"],
        question_type="multi-hop",
        secondary_question_types=[],
        reference_time=None,
        temporal_window=None,
        expected_hop_depth=2,
        is_multi_hop=True,
        cross_document_required=True,
    )
    # Evidence has only supplier relationship
    ev = RetrievalEvidence(
        evidence_id="graph:edge:company:NVDA->SUPPLIES->company:Samsung",
        source_type="graph",
        score=0.9,
        rank=1,
        text="NVIDIA Corporation --[SUPPLIES]--> Samsung Electronics Co., Ltd.",
        document_id="doc_1",
        chunk_id="chunk_1",
        available_time="2026-02-25T00:00:00+00:00",
        metadata={"relationship": "SUPPLIES", "hop_count": 2},
    )

    res = checker.check(qa, [ev])
    assert res.sufficient is False
    assert "Insufficient compound relationship coverage" in res.reason


def test_multihop_relationship_chain_required():
    """Verify that multi-hop queries require evidence covering both parts of the chain."""
    from src.qa.evidence_sufficiency import EvidenceSufficiencyChecker
    from src.qa.query_analysis import QueryAnalysis

    checker = EvidenceSufficiencyChecker()
    qa = QueryAnalysis(
        original_query="What risks affect companies that supply NVIDIA?",
        entities=["company:NVDA"],
        unresolved_entities=[],
        relationship_intent=["AFFECTED_BY", "HAS_RISK", "SUPPLIES"],
        question_type="multi-hop",
        secondary_question_types=[],
        reference_time=None,
        temporal_window=None,
        expected_hop_depth=2,
        is_multi_hop=True,
        cross_document_required=True,
    )

    ev_supplier = RetrievalEvidence(
        evidence_id="ev_supp",
        source_type="graph",
        score=0.9,
        rank=1,
        text="Micron supplies NVIDIA",
        document_id="doc_1",
        chunk_id="chunk_1",
        available_time="2025-08-27T00:00:00+00:00",
        metadata={"relationship": "SUPPLIES", "hop_count": 2},
    )
    ev_risk = RetrievalEvidence(
        evidence_id="ev_risk",
        source_type="graph",
        score=0.9,
        rank=2,
        text="Micron affected by supply chain disruptions",
        document_id="doc_1",
        chunk_id="chunk_2",
        available_time="2025-08-27T00:00:00+00:00",
        metadata={"relationship": "HAS_RISK", "hop_count": 2},
    )

    # When both relations exist, sufficiency succeeds
    res_both = checker.check(qa, [ev_supplier, ev_risk])
    assert res_both.sufficient is True
