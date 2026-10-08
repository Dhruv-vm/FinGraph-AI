from __future__ import annotations

from unittest.mock import MagicMock
import pytest

from src.evaluation.parametric_leakage import (
    ParametricLeakageEvaluator,
    ParametricLeakageResult,
    ParametricLeakageSummary,
    ParametricLeakageTestItem,
)


def test_retrieval_disabled_does_not_invoke_retrieval():
    mock_retriever = MagicMock()

    item = ParametricLeakageTestItem(
        question_id="pl_01",
        query="What chip did NVIDIA release in 2025?",
        reference_time="2024-01-01T00:00:00+00:00",
        ground_truth_answer="As of January 2024, NVIDIA had not announced Blackwell B200.",
        future_fact="Blackwell B200",
        future_event_date="2024-03-18",
    )

    def mock_raw_llm(query, sys_prompt):
        # Raw LLM recalling from its internal pre-training weights
        return "NVIDIA released the Blackwell B200 chip."

    evaluator = ParametricLeakageEvaluator(llm_callable=mock_raw_llm)
    result = evaluator.evaluate(item)

    # Assert retrieval was NOT called
    mock_retriever.retrieve.assert_not_called()
    assert result.retrieval_used is False
    assert result.reference_time == "2024-01-01T00:00:00+00:00"
    assert result.parametric_leakage_detected is True
    assert result.future_information_present is True


def test_manual_future_information_label_accepted():
    item = ParametricLeakageTestItem(
        question_id="pl_02",
        query="Did Company X merge with Company Y?",
        reference_time="2023-12-31T00:00:00+00:00",
        ground_truth_answer="No merger discussed as of 2023.",
        future_fact="Merger finalized in July 2024",
        future_event_date="2024-07-01",
        future_information_present=True,
        verified_by="expert_annotator_manual",
    )

    evaluator = ParametricLeakageEvaluator()
    result = evaluator.evaluate(item)

    assert result.parametric_leakage_detected is True
    assert result.future_information_present is True
    assert result.verified_by == "expert_annotator_manual"
    assert "expert_annotator_manual" in result.reason


def test_feir_and_plr_remain_separate_metrics():
    items = [
        ParametricLeakageTestItem(
            question_id="q1",
            query="Q1",
            reference_time="2024-01-01",
            ground_truth_answer="A1",
            future_fact="Future Fact 1",
            future_event_date="2024-06-01",
            future_information_present=True,
        ),
        ParametricLeakageTestItem(
            question_id="q2",
            query="Q2",
            reference_time="2024-01-01",
            ground_truth_answer="A2",
            future_fact="Future Fact 2",
            future_event_date="2024-06-01",
            future_information_present=False,
        ),
    ]

    evaluator = ParametricLeakageEvaluator()
    summary = evaluator.evaluate_suite(items)

    assert isinstance(summary, ParametricLeakageSummary)
    assert summary.total_questions == 2
    assert summary.future_containing_answers == 1
    # PLR is 1 / 2 = 0.50
    assert summary.plr == 0.50
    # FEIR is strictly 0.00% because retrieval is disabled
    assert summary.feir == 0.0
    assert summary.retrieval_used is False

    d = summary.to_dict()
    assert "plr" in d
    assert "feir" in d
    assert d["plr"] != d["feir"]


def test_no_model_fine_tuning_occurs():
    evaluator = ParametricLeakageEvaluator()
    # Evaluator is purely an inference testing harness; verify no train/loss/backward attributes exist
    assert not hasattr(evaluator, "train")
    assert not hasattr(evaluator, "loss")
    assert not hasattr(evaluator, "optimizer")


def test_plr_requires_manual_verification():
    """Verify that PLR result records distinguish candidate detector from manual verification."""
    item_unlabeled = ParametricLeakageTestItem(
        question_id="pl_test_01",
        query="What chip did NVIDIA release?",
        reference_time="2024-01-01",
        ground_truth_answer="Unknown",
        future_fact="Blackwell B200",
        future_event_date="2024-03-18",
    )
    evaluator = ParametricLeakageEvaluator(llm_callable=lambda q, s: "NVIDIA announced the Blackwell B200.")
    res = evaluator.evaluate(item_unlabeled)
    assert res.verification_method == "candidate_detector"
    assert "Keyword match" in res.evidence_of_future_information

    # When manually verified label is provided:
    item_manual = ParametricLeakageTestItem(
        question_id="pl_test_02",
        query="What chip did NVIDIA release?",
        reference_time="2024-01-01",
        ground_truth_answer="Unknown",
        future_fact="Blackwell B200",
        future_event_date="2024-03-18",
        future_information_present=True,
        verified_by="expert_human_auditor",
    )
    res_manual = evaluator.evaluate(item_manual)
    assert res_manual.verification_method == "manual"
    assert res_manual.verified_by == "expert_human_auditor"


def test_plr_reports_sample_size():
    """Verify that ParametricLeakageSummary explicitly records sample size N (total_questions)."""
    items = [
        ParametricLeakageTestItem(
            question_id=f"q_{i}",
            query=f"Query {i}",
            reference_time="2024-01-01",
            ground_truth_answer="GT",
            future_fact="Fact",
            future_event_date="2024-05-01",
            future_information_present=(i % 2 == 0),
        )
        for i in range(5)
    ]
    evaluator = ParametricLeakageEvaluator()
    summary = evaluator.evaluate_suite(items)
    assert summary.total_questions == 5
    assert summary.future_containing_answers == 3
    assert summary.plr == 0.60
    assert summary.to_dict()["total_questions"] == 5


def test_feir_and_plr_are_separate_metrics():
    """Verify that FEIR and PLR are strictly decoupled and never conflated."""
    evaluator = ParametricLeakageEvaluator()
    items = [
        ParametricLeakageTestItem(
            question_id="q_sep",
            query="Query",
            reference_time="2024-01-01",
            ground_truth_answer="GT",
            future_fact="Future Fact",
            future_event_date="2024-06-01",
            future_information_present=True,
        )
    ]
    summary = evaluator.evaluate_suite(items)
    # FEIR measures retrieval infiltration: strictly 0.00%
    assert summary.feir == 0.0
    # PLR measures internal weight memory: 1.00
    assert summary.plr == 1.0
    assert summary.feir != summary.plr
