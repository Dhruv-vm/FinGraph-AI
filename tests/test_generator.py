from __future__ import annotations

import json
from unittest.mock import MagicMock
import pytest

from src.qa.agent import QAEvidencePackage
from src.qa.evidence_sufficiency import SufficiencyResult
from src.qa.generator import AnswerGenerator, QAAnswer
from src.retrieval.evidence import RetrievalEvidence


def make_evidence(eid: str, time: str = "2024-01-01T00:00:00+00:00") -> RetrievalEvidence:
    return RetrievalEvidence(
        evidence_id=eid,
        source_type="graph",
        score=0.92,
        rank=1,
        text=f"Company A supplies components to Company B as of {time[:10]}.",
        document_id="doc_10k_2024",
        chunk_id="chunk_42",
        available_time=time,
        metadata={"relationship": "SUPPLIES", "hop_count": 1},
    )


def make_package(
    evidence: list[RetrievalEvidence],
    ref_time: str | None = "2024-06-01T00:00:00+00:00",
    sufficient: bool = True,
) -> QAEvidencePackage:
    return QAEvidencePackage(
        query="What companies supply Company B?",
        reference_time=ref_time,
        question_type="relationship",
        expected_hops=1,
        evidence=evidence,
        graph_paths=[["Company A", "Company B"]],
        sources=["doc_10k_2024"],
        retrieval_rounds=1,
        temporal_valid=True,
        sufficient=sufficient,
        sufficiency_result=SufficiencyResult(sufficient=sufficient, reason="All covered"),
    )


def test_generator_produces_structured_qa_answer():
    ev = make_evidence("ev_sup_1")
    pkg = make_package([ev])

    mock_llm = MagicMock(return_value=json.dumps({
        "answer": "Company A supplies components to Company B.",
        "confidence": 0.95,
        "evidence_ids": ["ev_sup_1"],
        "reasoning_summary": "Documented in 10-K filing chunk 42.",
    }))

    generator = AnswerGenerator(llm_client=mock_llm)
    answer = generator.generate(pkg)

    assert isinstance(answer, QAAnswer)
    assert answer.answer == "Company A supplies components to Company B."
    assert answer.confidence == 0.95
    assert answer.evidence_ids == ["ev_sup_1"]
    assert answer.insufficient_evidence is False
    assert len(answer.citations) == 1
    assert answer.citations[0]["evidence_id"] == "ev_sup_1"
    assert answer.citations[0]["document_id"] == "doc_10k_2024"
    assert answer.citations[0]["available_time"] == "2024-01-01T00:00:00+00:00"
    assert answer.sources == ["doc_10k_2024"]
    # Check dict-like subscripting compatibility
    assert answer["answer"] == "Company A supplies components to Company B."
    assert answer["evidence_package"] is not None


def test_generator_deterministic_temperature():
    generator = AnswerGenerator()
    assert generator.temperature == 0.0


def test_generator_insufficient_evidence_refuses_without_llm_invocation():
    pkg = make_package([], sufficient=False)
    mock_llm = MagicMock()

    generator = AnswerGenerator(llm_client=mock_llm)
    answer = generator.generate(pkg)

    # LLM must NEVER be invoked when evidence is insufficient
    mock_llm.assert_not_called()
    assert answer.insufficient_evidence is True
    assert "Insufficient evidence" in answer.answer
    assert answer.confidence == 0.0
    assert answer.evidence_ids == []


def test_generator_refuses_package_with_future_evidence_before_calling_llm():
    ev_future = make_evidence("ev_future", time="2025-01-01T00:00:00+00:00")
    # Reference cutoff is 2024-06-01, but ev_future is 2025-01-01
    pkg = make_package([ev_future], ref_time="2024-06-01T00:00:00+00:00", sufficient=True)
    mock_llm = MagicMock()

    generator = AnswerGenerator(llm_client=mock_llm)

    with pytest.raises(ValueError, match="Temporal safety violation"):
        generator.generate(pkg)

    mock_llm.assert_not_called()


def test_generator_handles_llm_failure_safely():
    ev = make_evidence("ev_1")
    pkg = make_package([ev])

    def failing_llm(prompt, sys_prompt):
        raise ConnectionError("Ollama daemon unreachable at port 11434")

    generator = AnswerGenerator(llm_client=failing_llm)
    answer = generator.generate(pkg)

    assert answer.insufficient_evidence is True
    assert "Generation failed" in answer.answer
    assert answer.validation_status["is_valid"] is False
    assert any("Ollama daemon unreachable" in e for e in answer.validation_status["errors"])


def test_generator_handles_malformed_json_safely():
    ev = make_evidence("ev_1")
    pkg = make_package([ev])

    mock_llm = MagicMock(return_value="NOT VALID JSON OUTPUT AT ALL")

    generator = AnswerGenerator(llm_client=mock_llm)
    answer = generator.generate(pkg)

    assert answer.validation_status["is_valid"] is False
    assert answer.insufficient_evidence is True
    assert any("not valid JSON" in e for e in answer.validation_status["errors"])


def test_generator_preserves_evidence_ids_in_prompt():
    ev = make_evidence("ev_stable_identifier_123")
    pkg = make_package([ev])

    captured_prompt = []

    def capturing_llm(prompt, sys_prompt):
        captured_prompt.append(prompt)
        return json.dumps({
            "answer": "Test answer",
            "confidence": 0.9,
            "evidence_ids": ["ev_stable_identifier_123"],
            "reasoning_summary": "Test",
        })

    generator = AnswerGenerator(llm_client=capturing_llm)
    generator.generate(pkg)

    assert len(captured_prompt) == 1
    assert "ev_stable_identifier_123" in captured_prompt[0]
