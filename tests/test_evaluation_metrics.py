from __future__ import annotations

from datetime import datetime, timezone
import pytest

from src.evaluation.leakage import LeakageEvaluator
from src.evaluation.parametric_leakage import ParametricLeakageEvaluator, ParametricLeakageTestItem
from src.evaluation.qa import (
    QAEvaluator,
    SingleQAResult,
    exact_match_score,
    token_f1_score,
)
from src.evaluation.retrieval import RetrievalEvaluator, SingleRetrievalResult
from src.evaluation.temporal import (
    TemporalEvaluator,
    TemporalStageMetrics,
    calculate_feir,
)


def test_retrieval_evaluator_metrics():
    evaluator = RetrievalEvaluator(k_values=(1, 3, 5))

    retrieved_chunks = ["c1", "c2", "c3", "c4", "c5"]
    relevant_chunks = ["c2", "c4"]
    retrieved_docs = ["d1", "d2"]
    relevant_docs = ["d2"]

    res: SingleRetrievalResult = evaluator.evaluate_single(
        question_id="Q1",
        retrieved_chunk_ids=retrieved_chunks,
        relevant_chunk_ids=relevant_chunks,
        retrieved_doc_ids=retrieved_docs,
        relevant_doc_ids=relevant_docs,
        expected_graph_paths=[["company:A", "COMPETES_WITH", "company:B"]],
        retrieved_graph_paths=[["company:A", "COMPETES_WITH", "company:B"]],
        is_answerable=True,
    )

    # c1 is not relevant, c2 is relevant
    assert res.hit_at_k[1] == 0.0
    assert res.hit_at_k[3] == 1.0
    assert res.precision_at_k[1] == 0.0
    assert res.precision_at_k[3] == 1.0 / 3.0
    assert res.recall_at_k[3] == 0.5  # c2 out of {c2, c4}
    assert res.mrr == 0.5  # c2 is at rank 2
    assert res.multi_hop_path_accuracy == 1.0
    assert res.doc_hit_at_k[3] == 1.0


def test_retrieval_evaluator_empty_relevance():
    evaluator = RetrievalEvaluator()
    res = evaluator.evaluate_single(
        question_id="Q_UNANSWERABLE",
        retrieved_chunk_ids=["c1", "c2"],
        relevant_chunk_ids=[],
        is_answerable=False,
    )
    assert res.hit_at_k[1] == 0.0
    assert res.recall_at_k[5] == 0.0
    assert res.mrr == 0.0
    assert res.is_answerable is False


def test_qa_evaluator_em_and_f1():
    pred = "Apple manufactures iPhones and Macs."
    gt = "Apple manufactures iphones and macs."
    assert exact_match_score(pred, gt) == 1.0
    assert token_f1_score(pred, gt) == 1.0

    pred2 = "Apple manufactures computers."
    assert exact_match_score(pred2, gt) == 0.0
    assert 0.0 < token_f1_score(pred2, gt) < 1.0


def test_qa_evaluator_refusal_metrics():
    qa_eval = QAEvaluator()

    results = [
        # True Positive (correct refusal)
        qa_eval.evaluate_single("Q1", "Insufficient evidence in the retrieved corpus.", "Insufficient evidence in the retrieved corpus.", is_answerable=False),
        # False Positive (false refusal)
        qa_eval.evaluate_single("Q2", "Insufficient evidence in the retrieved corpus.", "Apple manufactures iPhones.", is_answerable=True),
        # False Negative (hallucination on unanswerable)
        qa_eval.evaluate_single("Q3", "Apple manufactures lunar spacecraft.", "Insufficient evidence in the retrieved corpus.", is_answerable=False),
        # True Negative (correct answer)
        qa_eval.evaluate_single("Q4", "Microsoft provides Azure services.", "Microsoft provides Azure cloud services.", is_answerable=True),
    ]

    agg = qa_eval.evaluate_suite(results)
    ref = agg.refusal_metrics

    assert ref.total_unanswerable == 2
    assert ref.total_answerable == 2
    assert ref.correct_refusals == 1
    assert ref.false_refusals == 1
    assert ref.hallucinations == 1
    assert ref.correct_answers == 1
    assert ref.precision == 0.5
    assert ref.recall == 0.5
    assert ref.accuracy == 0.5


def test_temporal_evaluator_feir():
    cutoff = "2025-06-01T00:00:00+00:00"

    evidence = [
        {"available_time": "2025-04-30T00:00:00+00:00"},  # Past (valid)
        {"available_time": "2025-05-15T00:00:00+00:00"},  # Past (valid)
        {"available_time": "2025-08-27T00:00:00+00:00"},  # Future (invalid!)
        {"available_time": "2026-02-25T00:00:00+00:00"},  # Future (invalid!)
    ]

    feir = calculate_feir(evidence, cutoff)
    assert feir == 0.5  # 2 future out of 4 total

    # Test clean evidence
    clean_ev = evidence[:2]
    assert calculate_feir(clean_ev, cutoff) == 0.0


def test_temporal_ablation():
    temp_eval = TemporalEvaluator()
    cutoff = "2025-06-01T00:00:00+00:00"

    with_filtering = [{"available_time": "2025-04-30T00:00:00+00:00"}]
    without_filtering = [
        {"available_time": "2025-04-30T00:00:00+00:00"},
        {"available_time": "2025-08-27T00:00:00+00:00"},
    ]

    ablation = temp_eval.evaluate_ablation(with_filtering, without_filtering, cutoff)
    assert ablation.with_filtering_feir == 0.0
    assert ablation.without_filtering_feir == 0.5
    assert ablation.absolute_reduction == 0.5
    assert ablation.gate_rejections == 1


def test_temporal_query_with_future_evidence_before_filter_and_clean_after():
    """
    Test scenario 1: Temporal query with future evidence present in pre-filter stage,
    properly filtered out in post-filter, context, and cited stages.
    """
    evaluator = TemporalEvaluator()
    cutoff = "2025-06-01T00:00:00+00:00"

    pre_filter = [
        {"available_time": "2025-04-01T00:00:00+00:00"},
        {"available_time": "2025-05-01T00:00:00+00:00"},
        {"available_time": "2025-07-01T00:00:00+00:00"},  # Future item 1
        {"available_time": "2025-08-01T00:00:00+00:00"},  # Future item 2
    ]
    post_filter = pre_filter[:2]  # Only past items survived
    context = post_filter
    cited = [{"available_time": "2025-04-01T00:00:00+00:00"}]

    res = evaluator.evaluate_single(
        question_id="TQ-01",
        reference_time=cutoff,
        pre_filter_evidence=pre_filter,
        post_filter_evidence=post_filter,
        context_evidence=context,
        cited_evidence=cited,
    )

    assert res.pre_filter_feir == 0.5
    assert res.post_filter_feir == 0.0
    assert res.context_feir == 0.0
    assert res.cited_feir == 0.0
    assert res.gate_rejections == 2
    assert res.is_temporally_valid is True
    assert res.pre_filter_count == 4
    assert res.post_filter_count == 2
    assert res.context_count == 2
    assert res.cited_count == 1


def test_temporal_counts_and_gate_rejections_propagation():
    """
    Test scenario 2: Actual counts and gate rejections propagate correctly
    from per-question SingleTemporalResult into aggregate suite metrics.
    """
    evaluator = TemporalEvaluator()
    cutoff = "2025-06-01T00:00:00+00:00"

    q1 = evaluator.evaluate_single(
        question_id="TQ-01",
        reference_time=cutoff,
        pre_filter_evidence=[
            {"available_time": "2025-05-01T00:00:00+00:00"},
            {"available_time": "2025-07-01T00:00:00+00:00"},
        ],
        post_filter_evidence=[{"available_time": "2025-05-01T00:00:00+00:00"}],
        context_evidence=[{"available_time": "2025-05-01T00:00:00+00:00"}],
        cited_evidence=[{"available_time": "2025-05-01T00:00:00+00:00"}],
    )
    # q1 has 1 rejection, 2 pre, 1 post, 1 ctx, 1 cited

    q2 = evaluator.evaluate_single(
        question_id="TQ-02",
        reference_time=cutoff,
        pre_filter_evidence=[
            {"available_time": "2025-04-01T00:00:00+00:00"},
            {"available_time": "2025-08-01T00:00:00+00:00"},
            {"available_time": "2025-09-01T00:00:00+00:00"},
        ],
        post_filter_evidence=[{"available_time": "2025-04-01T00:00:00+00:00"}],
        context_evidence=[{"available_time": "2025-04-01T00:00:00+00:00"}],
        cited_evidence=[],
    )
    # q2 has 2 rejections, 3 pre, 1 post, 1 ctx, 0 cited

    agg = evaluator.evaluate_suite([q1, q2])

    assert agg.total_queries == 2
    assert agg.temporal_queries == 2
    assert agg.stage_metrics.gate_rejections == 3  # 1 + 2
    assert agg.stage_metrics.total_pre_filter_items == 5  # 2 + 3
    assert agg.stage_metrics.total_post_filter_items == 2  # 1 + 1
    assert agg.stage_metrics.total_context_items == 2  # 1 + 1
    assert agg.stage_metrics.total_cited_items == 1  # 1 + 0
    assert agg.stage_metrics.feir_post_filter == 0.0
    assert agg.stage_metrics.feir_context == 0.0
    assert agg.stage_metrics.feir_cited == 0.0
    assert agg.temporal_compliance_rate == 1.0


def test_missing_and_empty_stage_evidence_semantics():
    """
    Test scenario 3: Undefined metrics semantics for missing (None) and empty ([]) evidence stages.
    """
    evaluator = TemporalEvaluator()
    cutoff = "2025-06-01T00:00:00+00:00"

    # Case 3a: Pre-filter is uninstrumented (None), post-filter clean, cited is empty (refusal)
    res_uninstrumented = evaluator.evaluate_single(
        question_id="TQ-UNINST",
        reference_time=cutoff,
        pre_filter_evidence=None,
        post_filter_evidence=[{"available_time": "2025-05-01T00:00:00+00:00"}],
        context_evidence=[{"available_time": "2025-05-01T00:00:00+00:00"}],
        cited_evidence=[],
    )
    assert res_uninstrumented.pre_filter_feir is None
    assert res_uninstrumented.pre_filter_count == 0
    assert res_uninstrumented.gate_rejections == 0
    assert res_uninstrumented.post_filter_feir == 0.0
    assert res_uninstrumented.post_filter_count == 1
    assert res_uninstrumented.cited_feir is None
    assert res_uninstrumented.cited_count == 0
    assert res_uninstrumented.is_temporally_valid is True  # Post-filter was evaluated and compliant

    agg_uninst = evaluator.evaluate_suite([res_uninstrumented])
    assert agg_uninst.stage_metrics.feir_pre_filter is None
    assert agg_uninst.stage_metrics.total_pre_filter_items == 0
    assert agg_uninst.stage_metrics.feir_post_filter == 0.0
    assert agg_uninst.stage_metrics.total_post_filter_items == 1
    assert agg_uninst.stage_metrics.feir_cited is None
    assert agg_uninst.stage_metrics.total_cited_items == 0

    # Case 3b: Completely empty evidence across all stages (zero evaluated items)
    res_empty = evaluator.evaluate_single(
        question_id="TQ-EMPTY",
        reference_time=cutoff,
        pre_filter_evidence=[],
        post_filter_evidence=[],
        context_evidence=[],
        cited_evidence=[],
    )
    assert res_empty.post_filter_feir is None
    assert res_empty.context_feir is None
    assert res_empty.cited_feir is None
    # Must NOT report compliance when zero items were evaluated!
    assert res_empty.is_temporally_valid is False

    agg_empty = evaluator.evaluate_suite([res_empty])
    assert agg_empty.temporal_compliance_rate == 0.0


def test_zero_temporal_queries_does_not_report_perfect_compliance():
    """
    Test scenario 4: Suite with zero temporal queries must report 0.0 compliance,
    not misleading 1.0 (100%) compliance.
    """
    evaluator = TemporalEvaluator()

    # Query without reference_time
    res_non_temp = evaluator.evaluate_single(
        question_id="NTQ-01",
        reference_time=None,
        post_filter_evidence=[{"text": "generic document"}],
    )
    assert res_non_temp.reference_time is None

    agg = evaluator.evaluate_suite([res_non_temp])
    assert agg.total_queries == 1
    assert agg.temporal_queries == 0
    assert agg.temporal_compliance_rate == 0.0
    assert agg.stage_metrics.feir_pre_filter is None
    assert agg.stage_metrics.feir_post_filter is None
    assert agg.stage_metrics.feir_context is None
    assert agg.stage_metrics.feir_cited is None
    assert agg.stage_metrics.total_pre_filter_items == 0
    assert agg.stage_metrics.total_post_filter_items == 0

    # Empty result list
    agg_empty_list = evaluator.evaluate_suite([])
    assert agg_empty_list.total_queries == 0
    assert agg_empty_list.temporal_queries == 0
    assert agg_empty_list.temporal_compliance_rate == 0.0


def test_temporal_serialization_consistency():
    """
    Test scenario 5: Stage-level dataclass objects and AggregatedTemporalMetrics
    produce consistent, JSON-serializable dictionaries matching field definitions.
    """
    import json

    evaluator = TemporalEvaluator()
    cutoff = "2025-06-01T00:00:00+00:00"

    res = evaluator.evaluate_single(
        question_id="TQ-SER",
        reference_time=cutoff,
        pre_filter_evidence=[
            {"available_time": "2025-04-01T00:00:00+00:00"},
            {"available_time": "2025-07-01T00:00:00+00:00"},
        ],
        post_filter_evidence=[{"available_time": "2025-04-01T00:00:00+00:00"}],
        context_evidence=[{"available_time": "2025-04-01T00:00:00+00:00"}],
        cited_evidence=[{"available_time": "2025-04-01T00:00:00+00:00"}],
    )

    agg = evaluator.evaluate_suite([res])
    agg_dict = agg.to_dict()

    # JSON round-trip validation
    json_str = json.dumps(agg_dict)
    reloaded = json.loads(json_str)

    assert reloaded["total_queries"] == 1
    assert reloaded["temporal_queries"] == 1
    assert reloaded["temporal_compliance_rate"] == 1.0

    stage = reloaded["stage_metrics"]
    assert stage["feir_pre_filter"] == 0.5
    assert stage["feir_post_filter"] == 0.0
    assert stage["feir_context"] == 0.0
    assert stage["feir_cited"] == 0.0
    assert stage["gate_rejections"] == 1
    assert stage["total_pre_filter_items"] == 2
    assert stage["total_post_filter_items"] == 1
    assert stage["total_context_items"] == 1
    assert stage["total_cited_items"] == 1

    # Single result serialization
    res_dict = res.to_dict()
    res_json = json.dumps(res_dict)
    res_reloaded = json.loads(res_json)
    assert res_reloaded["question_id"] == "TQ-SER"
    assert res_reloaded["pre_filter_count"] == 2
    assert res_reloaded["gate_rejections"] == 1
    assert res_reloaded["is_temporally_valid"] is True
