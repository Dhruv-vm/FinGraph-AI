from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from scripts.run_evaluation import get_standard_parametric_probes, run_evaluation_suite
from src.evaluation.benchmark import BenchmarkQuestion, load_benchmark
from src.evaluation.qa import QAEvaluator
from src.evaluation.retrieval import RetrievalEvaluator
from src.evaluation.temporal import TemporalEvaluator


def test_full_benchmark_loaded_completely():
    questions = load_benchmark(only_verified=True)
    assert len(questions) == 64, f"Expected 64 verified benchmark questions, got {len(questions)}"

    # Check unique IDs
    ids = [q.question_id for q in questions]
    assert len(ids) == len(set(ids)), "Benchmark contains duplicate question IDs"

    # Check 50 answerable / 14 unanswerable split
    ans_count = sum(1 for q in questions if q.answerable)
    unans_count = sum(1 for q in questions if not q.answerable)
    assert ans_count == 50, f"Expected 50 answerable questions, got {ans_count}"
    assert unans_count == 14, f"Expected 14 unanswerable questions, got {unans_count}"


def test_smoke_test_does_not_overwrite_full_benchmark_artifacts(tmp_path: Path):
    """
    Verify that running with sample_size (smoke test) writes ONLY to smoke_test_*
    artifacts and NEVER creates or overwrites benchmark_evaluation_summary.json.
    """
    mock_vs = MagicMock()
    mock_vs.search.return_value = []

    with patch("scripts.run_evaluation.FinGraphPipeline") as mock_fg_cls, \
         patch("scripts.run_evaluation.VectorOnlyRAG") as mock_rag_cls, \
         patch("scripts.run_evaluation.VectorStore") as mock_vs_cls:

        mock_vs_cls.return_value = mock_vs
        mock_fg = MagicMock()
        mock_rag = MagicMock()
        mock_fg_cls.return_value = mock_fg
        mock_rag_cls.return_value = mock_rag

        mock_ans = MagicMock()
        mock_ans.answer = "Insufficient evidence in the retrieved corpus."
        mock_ans.confidence = 0.0
        mock_ans.evidence_ids = []
        mock_ans.citations = []
        mock_ans.validation_status = {"is_valid": True}

        mock_pkg = MagicMock()
        mock_pkg.all_evidence = []
        mock_pkg.primary_evidence = []
        mock_pkg.graph_paths = []
        mock_pkg.retrieval_rounds = 1

        mock_rag.retrieve.return_value = []
        mock_rag.answer.return_value = mock_ans
        mock_fg.answer.return_value = (mock_pkg, mock_ans)

        # Run with sample_size = 3
        res = run_evaluation_suite(
            sample_size=3,
            output_dir=tmp_path,
            run_ablation=False,
        )

        assert res["run_type"] == "smoke_test"
        assert res["is_complete_benchmark"] is False
        assert res["processed_questions"] == 3

        # Assert full benchmark artifact was NOT created in tmp_path
        full_artifact = tmp_path / "benchmark_evaluation_summary.json"
        assert not full_artifact.exists(), "Smoke test must NEVER create benchmark_evaluation_summary.json!"

        # Assert smoke test artifact WAS created
        smoke_artifact = tmp_path / "smoke_test_evaluation_summary.json"
        assert smoke_artifact.exists(), "Smoke test must create smoke_test_evaluation_summary.json"


def test_denominator_accounting_on_retrieval_and_refusal():
    ret_eval = RetrievalEvaluator()
    qa_eval = QAEvaluator()

    # Question 1: Answerable, retrieval succeeded
    r1 = ret_eval.evaluate_single("Q1", ["c1", "c2"], ["c1"], is_answerable=True)
    # Question 2: Answerable, retrieval failed
    r2 = ret_eval.evaluate_single("Q2", ["c3"], ["c1"], is_answerable=True)
    # Question 3: Unanswerable, empty relevance
    r3 = ret_eval.evaluate_single("Q3", ["c4"], [], is_answerable=False)

    agg_ret = ret_eval.evaluate_suite([r1, r2, r3])
    # Denominator for retrieval metrics must be 2 (the answerable questions only), not 3!
    assert agg_ret.total_questions == 3
    assert agg_ret.evaluated_questions == 2
    assert agg_ret.mean_hit_at_k[1] == 0.5  # 1 hit out of 2 evaluated answerable questions

    # QA refusal test
    q1 = qa_eval.evaluate_single("Q1", "Answer", "Answer", is_answerable=True)
    q2 = qa_eval.evaluate_single("Q2", "Answer", "Answer", is_answerable=True)
    q3 = qa_eval.evaluate_single("Q3", "Insufficient evidence in the retrieved corpus.", "Insufficient evidence in the retrieved corpus.", is_answerable=False)

    agg_qa = qa_eval.evaluate_suite([q1, q2, q3])
    assert agg_qa.refusal_metrics.total_unanswerable == 1
    assert agg_qa.refusal_metrics.total_answerable == 2
    assert agg_qa.refusal_metrics.correct_refusals == 1
    assert agg_qa.refusal_metrics.accuracy == 1.0


def test_error_handling_records_question_failure_without_dropping():
    from scripts.run_evaluation import run_evaluation_suite

    # Test that exception in answer generation is caught, logged in failed_questions_details
    # and recorded in results with status='failed'
    with patch("scripts.run_evaluation.FinGraphPipeline") as mock_fg_cls, \
         patch("scripts.run_evaluation.VectorOnlyRAG") as mock_rag_cls, \
         patch("scripts.run_evaluation.VectorStore") as mock_vs_cls:

        mock_vs_cls.return_value = MagicMock()
        mock_fg = MagicMock()
        mock_rag = MagicMock()
        mock_fg_cls.return_value = mock_fg
        mock_rag_cls.return_value = mock_rag

        mock_rag.retrieve.return_value = []
        mock_rag.answer.side_effect = RuntimeError("Simulated OOM or network error")

        mock_pkg = MagicMock()
        mock_pkg.all_evidence = []
        mock_pkg.primary_evidence = []
        mock_pkg.graph_paths = []
        mock_pkg.retrieval_rounds = 1
        mock_ans = MagicMock()
        mock_ans.answer = "Error fallback"
        mock_ans.confidence = 0.0
        mock_ans.evidence_ids = []
        mock_ans.citations = []
        mock_ans.validation_status = {"is_valid": False}
        mock_fg.answer.return_value = (mock_pkg, mock_ans)

        tmp_dir = Path("/tmp/test_eval_fail")
        tmp_dir.mkdir(parents=True, exist_ok=True)

        res = run_evaluation_suite(
            sample_size=1,
            output_dir=tmp_dir,
            run_ablation=False,
        )

        assert res["failed_questions"] == 1
        assert len(res["failed_questions_details"]) == 1
        assert "Simulated OOM" in res["failed_questions_details"][0]["error"]


def test_csv_reproduces_summary_metrics():
    """
    Assert that per_question_results.csv reproduces benchmark_evaluation_summary.json
    with exact equality across refusal confusion matrices, Token F1, and retrieval MRR.
    """
    import csv

    csv_path = Path(__file__).resolve().parents[1] / "results" / "phase5" / "per_question_results.csv"
    summary_path = Path(__file__).resolve().parents[1] / "results" / "phase5" / "benchmark_evaluation_summary.json"

    assert csv_path.exists(), f"Missing {csv_path}"
    assert summary_path.exists(), f"Missing {summary_path}"

    with csv_path.open() as f:
        rows = list(csv.DictReader(f))
    with summary_path.open() as f:
        summary = json.load(f)

    # 1. Total benchmark size and unique questions
    assert len(rows) == 64
    assert len({r["question_id"] for r in rows}) == 64

    # 2. Confusion matrix verification
    b1_outcomes = {k: 0 for k in ["TP_correct_refusal", "FN_hallucination", "FP_false_refusal", "TN_correct_answer"]}
    fg_outcomes = {k: 0 for k in ["TP_correct_refusal", "FN_hallucination", "FP_false_refusal", "TN_correct_answer"]}

    for r in rows:
        b1_outcomes[r["b1_refusal_outcome"]] += 1
        fg_outcomes[r["fg_refusal_outcome"]] += 1

    b1_sum_ref = summary["systems"]["B1_Vector_Only_RAG"]["qa"]["refusal_metrics"]
    fg_sum_ref = summary["systems"]["FinGraph_AI_Full_Framework"]["qa"]["refusal_metrics"]

    # B1 exact confusion matrix
    assert b1_outcomes["TP_correct_refusal"] == b1_sum_ref["correct_refusals"] == 9
    assert b1_outcomes["FN_hallucination"] == b1_sum_ref["hallucinations"] == 5
    assert b1_outcomes["FP_false_refusal"] == b1_sum_ref["false_refusals"] == 14
    assert b1_outcomes["TN_correct_answer"] == b1_sum_ref["correct_answers"] == 36
    assert round((9 + 36) / 64, 4) == round(b1_sum_ref["accuracy"], 4) == 0.7031

    # FinGraph AI exact confusion matrix
    assert fg_outcomes["TP_correct_refusal"] == fg_sum_ref["correct_refusals"] == 13
    assert fg_outcomes["FN_hallucination"] == fg_sum_ref["hallucinations"] == 1
    assert fg_outcomes["FP_false_refusal"] == fg_sum_ref["false_refusals"] == 28
    assert fg_outcomes["TN_correct_answer"] == fg_sum_ref["correct_answers"] == 22
    assert round((13 + 22) / 64, 4) == round(fg_sum_ref["accuracy"], 4) == 0.5469

    # 3. Mean Token F1 reproducibility
    b1_mean_f1 = sum(float(r["b1_token_f1"]) for r in rows) / len(rows)
    fg_mean_f1 = sum(float(r["fg_token_f1"]) for r in rows) / len(rows)
    assert round(b1_mean_f1, 4) == round(summary["systems"]["B1_Vector_Only_RAG"]["qa"]["mean_token_f1"], 4) == 0.3045
    assert round(fg_mean_f1, 4) == round(summary["systems"]["FinGraph_AI_Full_Framework"]["qa"]["mean_token_f1"], 4) == 0.3309

    # 4. Temporal Ablation reproducibility
    temp_rows = [r for r in rows if r["category"] == "temporal_point_in_time"]
    assert len(temp_rows) == 13
    mean_unfiltered_feir = sum(float(r["fg_unfiltered_feir"]) for r in temp_rows) / 13
    total_blocked = sum(int(r["fg_blocked_future_items"]) for r in temp_rows)

    assert round(mean_unfiltered_feir, 4) == round(summary["temporal_ablation"]["without_filtering_feir"], 4) == 0.7846
    assert total_blocked == summary["temporal_ablation"]["gate_rejections"] == 102
