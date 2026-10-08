from __future__ import annotations

from unittest.mock import MagicMock
import pytest

from src.evaluation.parametric_leakage import (
    ParametricLeakageEvaluator,
    ParametricLeakageTestItem,
)
from src.qa.agent import AnswerGenerator, QAEvidencePackage, RetrievalAgent
from src.qa.evidence_sufficiency import SufficiencyResult
from src.qa.query_analysis import QueryAnalysis
from src.retrieval.evidence import RetrievalEvidence
from src.retrieval.orchestrator import RetrievalOrchestrator


def make_valid_evidence(eid: str, time: str = "2024-01-01T00:00:00+00:00") -> RetrievalEvidence:
    return RetrievalEvidence(
        evidence_id=eid,
        source_type="graph",
        score=0.9,
        rank=1,
        text=f"Text for {eid}",
        document_id="doc_1",
        chunk_id="chunk_1",
        available_time=time,
        metadata={"relationship": "SUPPLIES", "hop_count": 1},
    )


def test_agent_sufficient_evidence_stops_immediately():
    mock_orchestrator = MagicMock()
    mock_orchestrator.orchestrate.return_value = (
        [make_valid_evidence("ev_1")],
        QueryAnalysis(original_query="q", entities=["company:NVDA"], relationship_intent=["SUPPLIES"]),
        SufficiencyResult(sufficient=True, reason="All covered", required_hops=1, observed_hops=1),
    )

    agent = RetrievalAgent(orchestrator=mock_orchestrator, max_additional_iterations=2)
    package = agent.run("What company supplies NVIDIA?")

    assert package.sufficient is True
    assert package.retrieval_rounds == 1
    assert len(package.trace) == 1
    assert package.trace[0]["action"] == "initial_orchestration"


def test_agent_insufficient_evidence_triggers_additional_retrieval():
    mock_orchestrator = MagicMock()
    # Round 1 returns insufficient
    mock_orchestrator.orchestrate.return_value = (
        [make_valid_evidence("ev_1")],
        QueryAnalysis(original_query="q", entities=["company:NVDA"], relationship_intent=["DEPENDS_ON"]),
        SufficiencyResult(sufficient=False, reason="Missing relationship", missing_relationships=["DEPENDS_ON"]),
    )

    # Secondary targeted retrieval in hybrid_retriever returns resolved evidence
    mock_orchestrator.hybrid_retriever.retrieve.return_value = [make_valid_evidence("ev_2_depends")]
    mock_orchestrator.reranker.rerank.return_value = [make_valid_evidence("ev_2_depends"), make_valid_evidence("ev_1")]
    # Sufficiency checker in round 2 returns sufficient
    mock_orchestrator.sufficiency_checker.check.return_value = SufficiencyResult(
        sufficient=True, reason="Resolved", required_hops=1, observed_hops=1
    )

    agent = RetrievalAgent(orchestrator=mock_orchestrator, max_additional_iterations=2)
    package = agent.run("What companies does NVIDIA depend on?")

    assert package.retrieval_rounds == 2
    assert len(package.trace) == 2
    assert package.trace[1]["action"] == "retrieve_missing_relationship"
    assert package.sufficient is True


def test_agent_maximum_two_additional_iterations_enforced():
    mock_orchestrator = MagicMock()
    # Always insufficient
    mock_orchestrator.orchestrate.return_value = (
        [make_valid_evidence("ev_1")],
        QueryAnalysis(original_query="q", entities=["company:NVDA"], relationship_intent=["DEPENDS_ON"]),
        SufficiencyResult(sufficient=False, reason="Missing", missing_relationships=["DEPENDS_ON"]),
    )
    mock_orchestrator.hybrid_retriever.retrieve.return_value = [make_valid_evidence("ev_sub")]
    mock_orchestrator.reranker.rerank.return_value = [make_valid_evidence("ev_sub")]
    mock_orchestrator.sufficiency_checker.check.return_value = SufficiencyResult(
        sufficient=False, reason="Still missing", missing_relationships=["DEPENDS_ON"]
    )

    agent = RetrievalAgent(orchestrator=mock_orchestrator, max_additional_iterations=2)
    package = agent.run("Query")

    # Round 1 initial + max 2 additional = 3 rounds total
    assert package.retrieval_rounds == 3
    assert len(package.trace) == 3


def test_agent_no_arbitrary_actions_possible():
    # Verify allowed actions set
    agent = RetrievalAgent(orchestrator=MagicMock())
    assert agent.ALLOWED_ACTIONS == {
        "initial_orchestration",
        "retrieve_missing_relationship",
        "increase_hop_depth",
        "retrieve_missing_entity",
        "broaden_retrieval",
    }


def test_agent_retrieval_trace_recorded():
    mock_orchestrator = MagicMock()
    mock_orchestrator.orchestrate.return_value = (
        [make_valid_evidence("ev_1")],
        QueryAnalysis(original_query="q", entities=[], reference_time="2024-03-31T00:00:00+00:00"),
        SufficiencyResult(sufficient=True, reason="Sufficient"),
    )

    agent = RetrievalAgent(orchestrator=mock_orchestrator)
    package = agent.run("Query")

    assert len(package.trace) == 1
    t = package.trace[0]
    assert t["round"] == 1
    assert t["action"] == "initial_orchestration"
    assert t["reference_time"] == "2024-03-31T00:00:00+00:00"


def test_future_evidence_cannot_enter_final_package():
    mock_orchestrator = MagicMock()
    # Simulate an orchestrator returning a future evidence item
    future_ev = make_valid_evidence("ev_future", time="2026-01-01T00:00:00+00:00")
    mock_orchestrator.orchestrate.return_value = (
        [future_ev],
        QueryAnalysis(original_query="q", reference_time="2024-01-01T00:00:00+00:00"),
        SufficiencyResult(sufficient=True, reason="Mock"),
    )

    agent = RetrievalAgent(orchestrator=mock_orchestrator)
    package = agent.run("Query as of 2024-01-01")

    # Package must mark temporal_valid as False because evidence is in the future
    assert package.temporal_valid is False
    assert len(package.evidence) == 0  # Future candidate purged from final package


def test_agent_additional_retrieval_enforces_temporal_cutoff():
    mock_orchestrator = MagicMock()
    cutoff = "2025-06-01T00:00:00+00:00"

    # Initial retrieval (Round 1): valid historical evidence, but insufficient
    initial_ev = make_valid_evidence("ev_init", time="2025-01-01T00:00:00+00:00")
    analysis = QueryAnalysis(
        original_query="What risks affected NVIDIA as of June 1, 2025?",
        entities=["company:NVDA"],
        relationship_intent=["HAS_RISK"],
        reference_time=cutoff,
    )
    suff_r1 = SufficiencyResult(
        sufficient=False,
        reason="Missing relationship",
        missing_relationships=["HAS_RISK"],
    )
    mock_orchestrator.orchestrate.return_value = ([initial_ev], analysis, suff_r1)

    # Additional retrieval (Round 2): returns one valid item and one future item
    valid_additional = make_valid_evidence("ev_add_valid", time="2025-03-01T00:00:00+00:00")
    future_additional = make_valid_evidence("ev_add_future", time="2025-08-01T00:00:00+00:00")
    mock_orchestrator.hybrid_retriever.retrieve.return_value = [valid_additional, future_additional]

    mock_orchestrator.reranker.rerank.side_effect = lambda query, candidates, top_k, reference_time: [
        c for c in candidates if c.available_time <= cutoff
    ]
    mock_orchestrator.sufficiency_checker.check.return_value = SufficiencyResult(
        sufficient=True,
        reason="Sufficient now",
        temporal_valid=True,
    )

    agent = RetrievalAgent(orchestrator=mock_orchestrator, max_additional_iterations=2)
    package = agent.run("What risks affected NVIDIA as of June 1, 2025?", reference_time=cutoff)

    assert package.retrieval_rounds == 2
    assert package.temporal_valid is True
    # Verify no future candidate leaked into package evidence
    for ev in package.evidence:
        assert ev.available_time <= cutoff
    ev_ids = [ev.evidence_id for ev in package.evidence]
    assert "ev_init" in ev_ids
    assert "ev_add_valid" in ev_ids
    assert "ev_add_future" not in ev_ids


def test_answer_generator_interface():
    gen = AnswerGenerator()
    pkg = QAEvidencePackage(
        query="What is X?",
        reference_time=None,
        question_type="factual",
        expected_hops=1,
        evidence=[],
        graph_paths=[],
        sources=[],
        retrieval_rounds=1,
        temporal_valid=True,
        sufficient=True,
    )
    result = gen.generate(pkg)
    assert "answer" in result
    assert result["answer"] is None
    assert "evidence_package" in result


def test_parametric_leakage_interface_does_not_invoke_retrieval():
    evaluator = ParametricLeakageEvaluator()
    item = ParametricLeakageTestItem(
        question_id="test_01",
        query="What chip did NVIDIA release in 2025?",
        reference_time="2024-01-01T00:00:00+00:00",
        ground_truth_answer="Unknown as of 2024",
        future_fact="Blackwell B200",
        future_event_date="2024-03-18",
    )

    # Evaluate without retrieval
    res = evaluator.evaluate(item)
    assert res.question_id == "test_01"
    assert res.reference_time == "2024-01-01T00:00:00+00:00"
    assert res.future_event_date == "2024-03-18"
    assert isinstance(res.parametric_leakage_detected, bool)
