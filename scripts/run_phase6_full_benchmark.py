"""
FinGraph AI — Phase 6 Full Benchmark Verification Runner.

Evaluates all 64 benchmark questions under identical conditions across:
1. B1: Dense Vector-Only MiniLM/Qdrant Baseline (top_k=5)
2. K5: FinGraph AI Full Framework (default K5 evidence context)
3. K8: FinGraph AI Full Framework (expanded K8 evidence context)

Outputs strictly written to results/phase6/full_benchmark/:
- benchmark_evaluation_summary.json
- per_question_results.csv
- error_analysis.json
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Ensure project root is on sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_evaluation import get_standard_parametric_probes
from scripts.run_fingraph import FinGraphPipeline
from scripts.run_rag_baseline import VectorOnlyRAG
from src.evaluation.benchmark import BenchmarkQuestion, load_benchmark
from src.evaluation.leakage import LeakageEvaluator, UnifiedLeakageReport
from src.evaluation.qa import QAEvaluator, SingleQAResult
from src.evaluation.retrieval import RetrievalEvaluator, SingleRetrievalResult
from src.evaluation.temporal import (
    SingleTemporalResult,
    TemporalAblationResult,
    TemporalEvaluator,
)
from src.retrieval.vector import VectorStore

DEFAULT_OUTPUT_DIR = ROOT / "results" / "phase6" / "full_benchmark"
TOTAL_BENCHMARK_EXPECTED = 64


def classify_answerable_failure(
    q: BenchmarkQuestion,
    pkg: Any,
    ans: Any,
    qa_res: SingleQAResult,
    ret_res: SingleRetrievalResult,
    max_evidence_items: int,
) -> str:
    """
    Classify the principal failure mode for an answerable question:
    - EVIDENCE_NOT_RETRIEVED
    - EVIDENCE_RANKED_TOO_LOW
    - ENTITY_LINKING_FAILURE
    - GRAPH_PATH_FAILURE
    - SUFFICIENCY_GATE_FALSE_REFUSAL
    - GENERATOR_FAILURE
    - NONE_SUCCESS
    """
    if not q.answerable:
        return "N/A_UNANSWERABLE"

    # If the model answered with reasonable token overlap (F1 >= 0.20 or exact match)
    if not qa_res.is_refusal and (qa_res.token_f1 >= 0.20 or qa_res.exact_match > 0.0):
        return "NONE_SUCCESS"

    retrieved_cids = [ev.chunk_id for ev in pkg.all_evidence if ev.chunk_id] if pkg else []
    rel_cids = set(q.relevant_chunk_ids)

    # 1. Evidence not retrieved at all
    if rel_cids and not any(cid in rel_cids for cid in retrieved_cids):
        # Check if entity linking failed
        linked_entities = pkg.analysis.entities if (pkg and pkg.analysis) else []
        exp_entities = [e.lower() for e in q.entities]
        found_entity = any(any(ee in str(le).lower() for ee in exp_entities) for le in linked_entities)
        if not found_entity and exp_entities:
            return "ENTITY_LINKING_FAILURE"
        return "EVIDENCE_NOT_RETRIEVED"

    # 2. Evidence retrieved but ranked too low (outside top context items)
    context_cids = [ev.chunk_id for ev in pkg.evidence[:max_evidence_items] if ev.chunk_id] if pkg else []
    if rel_cids and not any(cid in rel_cids for cid in context_cids):
        return "EVIDENCE_RANKED_TOO_LOW"

    # 3. Graph path or hop depth failure
    if q.expected_reasoning_hops > 1 and q.expected_graph_path:
        if ret_res.multi_hop_path_accuracy == 0.0:
            if pkg and pkg.sufficiency_result and "reasoning depth" in pkg.sufficiency_result.reason.lower():
                return "GRAPH_PATH_FAILURE"

    # 4. Sufficiency gate false refusal
    if pkg and not pkg.sufficient:
        return "SUFFICIENCY_GATE_FALSE_REFUSAL"

    # 5. Generator answer or citation failure
    return "GENERATOR_FAILURE"


def classify_unanswerable_outcome(
    q: BenchmarkQuestion,
    ans: Any,
    qa_res: SingleQAResult,
) -> str:
    """
    Classify refusal/groundedness outcome for an unanswerable question:
    - CORRECT_REFUSAL: Refused correctly
    - UNGROUNDED_HALLUCINATION: Model attempted substantive answer to unanswerable query
    """
    if q.answerable:
        return "N/A_ANSWERABLE"

    if qa_res.is_refusal:
        return "CORRECT_REFUSAL"
    else:
        return "UNGROUNDED_HALLUCINATION"


def run_full_benchmark_verification(
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    sample_size: int | None = None,
    run_ablation: bool = True,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)

    all_questions = load_benchmark(only_verified=True)
    if sample_size is not None and sample_size < len(all_questions):
        benchmark_questions = all_questions[:sample_size]
        is_full_benchmark = False
    else:
        benchmark_questions = all_questions
        is_full_benchmark = True

    print("=" * 80)
    print("FinGraph AI — Phase 6 Full Benchmark Verification")
    print(f"Questions: {len(benchmark_questions)} (Total benchmark: {len(all_questions)})")
    print(f"Target Output Directory: {output_dir}")
    print("Systems: B1_Vector_Only_RAG, FinGraph_AI_K5, FinGraph_AI_K8")
    print("=" * 80)

    # 1. Initialize Evaluators and Systems
    retrieval_evaluator = RetrievalEvaluator(k_values=(1, 3, 5, 10))
    qa_evaluator = QAEvaluator()
    temporal_evaluator = TemporalEvaluator()
    leakage_evaluator = LeakageEvaluator(temporal_evaluator=temporal_evaluator)

    shared_vector_store = VectorStore(ROOT / "data" / "vector_store")
    baseline_rag = VectorOnlyRAG(vector_store=shared_vector_store, top_k=5)
    fingraph = FinGraphPipeline(vector_store=shared_vector_store)

    # Containers for results
    b1_ret_list: list[SingleRetrievalResult] = []
    fg_ret_list: list[SingleRetrievalResult] = []

    b1_qa_list: list[SingleQAResult] = []
    k5_qa_list: list[SingleQAResult] = []
    k8_qa_list: list[SingleQAResult] = []

    b1_temp_list: list[SingleTemporalResult] = []
    fg_temp_list: list[SingleTemporalResult] = []

    b1_latencies: list[float] = []
    k5_latencies: list[float] = []
    k8_latencies: list[float] = []

    csv_rows: list[dict[str, Any]] = []
    error_analysis_entries: list[dict[str, Any]] = []

    total_q = len(benchmark_questions)
    answerable_count = sum(1 for q in benchmark_questions if q.answerable)
    unanswerable_count = total_q - answerable_count

    t_suite_start = time.perf_counter()

    for idx, q in enumerate(benchmark_questions, 1):
        print(f"\n--- [{idx:02d}/{total_q}] {q.question_id} ({q.category}) ---")
        print(f"Query: {q.question}")

        # =====================================================================
        # A. Baseline B1 (Vector-Only RAG, top_k=5)
        # =====================================================================
        t0_b1 = time.perf_counter()
        try:
            b1_evidence = baseline_rag.retrieve(query=q.question, reference_time=q.reference_time)
            b1_ans = baseline_rag.answer(query=q.question, reference_time=q.reference_time)
        except Exception as exc:
            print(f"  [B1 Error] {exc}")
            b1_evidence = []
            b1_ans = None
        t1_b1 = time.perf_counter()
        b1_lat = t1_b1 - t0_b1
        b1_latencies.append(b1_lat)

        b1_cids = [ev.chunk_id for ev in b1_evidence if ev.chunk_id]
        b1_dids = sorted(list({ev.document_id for ev in b1_evidence if ev.document_id}))

        b1_ret = retrieval_evaluator.evaluate_single(
            question_id=q.question_id,
            retrieved_chunk_ids=b1_cids,
            relevant_chunk_ids=q.relevant_chunk_ids,
            retrieved_doc_ids=b1_dids,
            relevant_doc_ids=q.relevant_document_ids,
            retrieved_graph_paths=[],
            expected_graph_paths=q.expected_graph_path,
            is_answerable=q.answerable,
        )
        b1_ret_list.append(b1_ret)

        b1_ans_text = b1_ans.answer if b1_ans else "Error in B1 baseline"
        b1_qa = qa_evaluator.evaluate_single(
            question_id=q.question_id,
            predicted_answer=b1_ans_text,
            ground_truth_answer=q.ground_truth_answer,
            is_answerable=q.answerable,
            confidence=b1_ans.confidence if b1_ans else 0.0,
            evidence_ids_cited=b1_ans.evidence_ids if b1_ans else [],
            citations=b1_ans.citations if b1_ans else [],
            validation_passed=b1_ans.validation_status.get("is_valid", False) if (b1_ans and b1_ans.validation_status) else False,
        )
        b1_qa_list.append(b1_qa)

        b1_temp = temporal_evaluator.evaluate_single(
            question_id=q.question_id,
            reference_time=q.reference_time,
            pre_filter_evidence=None,
            post_filter_evidence=b1_evidence,
            context_evidence=b1_evidence[:5],
            cited_evidence=b1_ans.citations if b1_ans else [],
        )
        b1_temp_list.append(b1_temp)

        # =====================================================================
        # B. FinGraph AI (Shared Retrieval top_k=10, K5 vs K8 generation)
        # =====================================================================
        t0_fg_ret = time.perf_counter()
        try:
            ref_str = q.reference_time.isoformat() if isinstance(q.reference_time, datetime) else q.reference_time
            fg_pkg = fingraph.agent.run(query=q.question, reference_time=ref_str, top_k=10)
        except Exception as exc:
            print(f"  [FinGraph Retrieval Error] {exc}")
            fg_pkg = None
        t1_fg_ret = time.perf_counter()
        fg_ret_time = t1_fg_ret - t0_fg_ret

        fg_evidence = fg_pkg.all_evidence if fg_pkg else []
        fg_cids = [ev.chunk_id for ev in fg_evidence if ev.chunk_id]
        fg_dids = sorted(list({ev.document_id for ev in fg_evidence if ev.document_id}))
        fg_paths = fg_pkg.graph_paths if fg_pkg else []

        fg_ret = retrieval_evaluator.evaluate_single(
            question_id=q.question_id,
            retrieved_chunk_ids=fg_cids,
            relevant_chunk_ids=q.relevant_chunk_ids,
            retrieved_doc_ids=fg_dids,
            relevant_doc_ids=q.relevant_document_ids,
            retrieved_graph_paths=fg_paths,
            expected_graph_paths=q.expected_graph_path,
            is_answerable=q.answerable,
        )
        fg_ret_list.append(fg_ret)

        # Generate K5 answer (5 evidence items)
        t0_k5 = time.perf_counter()
        try:
            k5_ans = fingraph.generator.generate(fg_pkg, max_evidence_items=5) if fg_pkg else None
        except Exception as exc:
            print(f"  [FinGraph K5 Error] {exc}")
            k5_ans = None
        t1_k5 = time.perf_counter()
        k5_lat = fg_ret_time + (t1_k5 - t0_k5)
        k5_latencies.append(k5_lat)

        k5_ans_text = k5_ans.answer if k5_ans else "Error in K5 execution"
        k5_qa = qa_evaluator.evaluate_single(
            question_id=q.question_id,
            predicted_answer=k5_ans_text,
            ground_truth_answer=q.ground_truth_answer,
            is_answerable=q.answerable,
            confidence=k5_ans.confidence if k5_ans else 0.0,
            evidence_ids_cited=k5_ans.evidence_ids if k5_ans else [],
            citations=k5_ans.citations if k5_ans else [],
            validation_passed=k5_ans.validation_status.get("is_valid", False) if (k5_ans and k5_ans.validation_status) else False,
        )
        k5_qa_list.append(k5_qa)

        # Generate K8 answer (8 evidence items)
        t0_k8 = time.perf_counter()
        try:
            k8_ans = fingraph.generator.generate(fg_pkg, max_evidence_items=8) if fg_pkg else None
        except Exception as exc:
            print(f"  [FinGraph K8 Error] {exc}")
            k8_ans = None
        t1_k8 = time.perf_counter()
        k8_lat = fg_ret_time + (t1_k8 - t0_k8)
        k8_latencies.append(k8_lat)

        k8_ans_text = k8_ans.answer if k8_ans else "Error in K8 execution"
        k8_qa = qa_evaluator.evaluate_single(
            question_id=q.question_id,
            predicted_answer=k8_ans_text,
            ground_truth_answer=q.ground_truth_answer,
            is_answerable=q.answerable,
            confidence=k8_ans.confidence if k8_ans else 0.0,
            evidence_ids_cited=k8_ans.evidence_ids if k8_ans else [],
            citations=k8_ans.citations if k8_ans else [],
            validation_passed=k8_ans.validation_status.get("is_valid", False) if (k8_ans and k8_ans.validation_status) else False,
        )
        k8_qa_list.append(k8_qa)

        # Temporal evaluation (evaluated at post-filter and context)
        fg_temp = temporal_evaluator.evaluate_single(
            question_id=q.question_id,
            reference_time=q.reference_time,
            pre_filter_evidence=None,
            post_filter_evidence=fg_evidence,
            context_evidence=fg_evidence[:5],
            cited_evidence=k5_ans.citations if k5_ans else [],
        )
        fg_temp_list.append(fg_temp)

        # Refusal outcomes classification
        # Refusal definition:
        # TP_correct_refusal: unanswerable and refused
        # FN_hallucination: unanswerable and answered (failed to refuse)
        # FP_false_refusal: answerable and refused (false refusal)
        # TN_correct_decision_to_answer: answerable and answered
        def _get_outcome(is_ans: bool, is_ref: bool) -> str:
            if not is_ans:
                return "TP_correct_refusal" if is_ref else "FN_hallucination"
            else:
                return "FP_false_refusal" if is_ref else "TN_correct_decision_to_answer"

        b1_outcome = _get_outcome(q.answerable, b1_qa.is_refusal)
        k5_outcome = _get_outcome(q.answerable, k5_qa.is_refusal)
        k8_outcome = _get_outcome(q.answerable, k8_qa.is_refusal)

        # Classify failures for answerable questions
        k5_failure_mode = classify_answerable_failure(q, fg_pkg, k5_ans, k5_qa, fg_ret, 5)
        k8_failure_mode = classify_answerable_failure(q, fg_pkg, k8_ans, k8_qa, fg_ret, 8)
        k5_unans_mode = classify_unanswerable_outcome(q, k5_ans, k5_qa)
        k8_unans_mode = classify_unanswerable_outcome(q, k8_ans, k8_qa)

        print(f"  [B1 Vector RAG] Outcome: {b1_outcome:<28} | F1: {b1_qa.token_f1:.2f} | Time: {b1_lat:.2f}s")
        print(f"  [FinGraph K5]   Outcome: {k5_outcome:<28} | F1: {k5_qa.token_f1:.2f} | PathAcc: {fg_ret.multi_hop_path_accuracy:.2f} | Time: {k5_lat:.2f}s")
        print(f"  [FinGraph K8]   Outcome: {k8_outcome:<28} | F1: {k8_qa.token_f1:.2f} | Time: {k8_lat:.2f}s")

        # Record CSV row
        csv_rows.append({
            "question_id": q.question_id,
            "category": q.category,
            "hop_level": q.hop_level,
            "answerable": q.answerable,
            "reference_time": q.reference_time or "",
            "b1_outcome": b1_outcome,
            "b1_token_f1": round(b1_qa.token_f1, 4),
            "b1_exact_match": round(b1_qa.exact_match, 4),
            "b1_hit_at_1": round(b1_ret.hit_at_k.get(1, 0.0), 4),
            "b1_hit_at_5": round(b1_ret.hit_at_k.get(5, 0.0), 4),
            "b1_hit_at_10": round(b1_ret.hit_at_k.get(10, 0.0), 4),
            "b1_mrr": round(b1_ret.mrr, 4),
            "b1_ndcg_at_5": round(b1_ret.ndcg_at_k.get(5, 0.0), 4),
            "b1_doc_recall_at_5": round(b1_ret.doc_recall_at_k.get(5, 0.0), 4),
            "b1_feir": round(b1_temp.post_filter_feir, 4) if b1_temp.post_filter_feir is not None else 0.0,
            "b1_latency_s": round(b1_lat, 3),
            "fg_hit_at_1": round(fg_ret.hit_at_k.get(1, 0.0), 4),
            "fg_hit_at_5": round(fg_ret.hit_at_k.get(5, 0.0), 4),
            "fg_hit_at_10": round(fg_ret.hit_at_k.get(10, 0.0), 4),
            "fg_mrr": round(fg_ret.mrr, 4),
            "fg_ndcg_at_5": round(fg_ret.ndcg_at_k.get(5, 0.0), 4),
            "fg_doc_recall_at_5": round(fg_ret.doc_recall_at_k.get(5, 0.0), 4),
            "fg_cross_doc_coverage": round(fg_ret.cross_doc_coverage, 4),
            "fg_path_accuracy": round(fg_ret.multi_hop_path_accuracy, 4),
            "fg_feir": round(fg_temp.post_filter_feir, 4) if fg_temp.post_filter_feir is not None else 0.0,
            "k5_outcome": k5_outcome,
            "k5_token_f1": round(k5_qa.token_f1, 4),
            "k5_exact_match": round(k5_qa.exact_match, 4),
            "k5_latency_s": round(k5_lat, 3),
            "k5_failure_mode": k5_failure_mode,
            "k8_outcome": k8_outcome,
            "k8_token_f1": round(k8_qa.token_f1, 4),
            "k8_exact_match": round(k8_qa.exact_match, 4),
            "k8_latency_s": round(k8_lat, 3),
            "k8_failure_mode": k8_failure_mode,
            "b1_predicted_answer": b1_ans_text[:120].replace("\n", " "),
            "k5_predicted_answer": k5_ans_text[:120].replace("\n", " "),
            "k8_predicted_answer": k8_ans_text[:120].replace("\n", " "),
            "ground_truth_answer": q.ground_truth_answer[:120].replace("\n", " "),
        })

        error_analysis_entries.append({
            "question_id": q.question_id,
            "category": q.category,
            "answerable": q.answerable,
            "question": q.question,
            "b1": {
                "outcome": b1_outcome,
                "token_f1": round(b1_qa.token_f1, 4),
                "exact_match": round(b1_qa.exact_match, 4),
            },
            "k5": {
                "outcome": k5_outcome,
                "token_f1": round(k5_qa.token_f1, 4),
                "exact_match": round(k5_qa.exact_match, 4),
                "failure_mode": k5_failure_mode if q.answerable else k5_unans_mode,
                "sufficient_evidence": fg_pkg.sufficient if fg_pkg else False,
                "sufficiency_reason": fg_pkg.sufficiency_result.reason if (fg_pkg and fg_pkg.sufficiency_result) else "N/A",
            },
            "k8": {
                "outcome": k8_outcome,
                "token_f1": round(k8_qa.token_f1, 4),
                "exact_match": round(k8_qa.exact_match, 4),
                "failure_mode": k8_failure_mode if q.answerable else k8_unans_mode,
            },
        })

    t_suite_end = time.perf_counter()
    total_duration = t_suite_end - t_suite_start
    print(f"\nExecution of {total_q} questions completed in {total_duration:.2f}s.")

    # =========================================================================
    # C. Aggregate Metrics Computation with Explicit Denominators
    # =========================================================================
    def _compute_system_metrics(
        name: str,
        qa_results: list[SingleQAResult],
        ret_results: list[SingleRetrievalResult],
        temp_results: list[SingleTemporalResult],
        latencies: list[float],
    ) -> dict[str, Any]:
        ans_qa = [r for r in qa_results if r.is_answerable]
        unans_qa = [r for r in qa_results if not r.is_answerable]

        # Confusion Matrix
        # TP: unanswerable and refused
        # FN: unanswerable and answered (hallucination)
        # FP: answerable and refused (false refusal)
        # TN: answerable and answered (correct decision to answer)
        tp = sum(1 for r in unans_qa if r.is_refusal)
        fn = sum(1 for r in unans_qa if not r.is_refusal)
        fp = sum(1 for r in ans_qa if r.is_refusal)
        tn = sum(1 for r in ans_qa if not r.is_refusal)

        # Decision accuracy = (TP + TN) / Total
        decision_acc = (tp + tn) / total_q if total_q > 0 else 0.0

        # Refusal metrics
        refusal_recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        refusal_precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        false_refusal_rate = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        hallucination_rate = fn / (tp + fn) if (tp + fn) > 0 else 0.0

        # Answer Correctness on Answerable Questions (N = 50)
        # Mean Token F1 across all answerable questions (refusals get 0.0)
        ans_mean_f1 = sum(r.token_f1 for r in ans_qa) / len(ans_qa) if ans_qa else 0.0
        ans_mean_em = sum(r.exact_match for r in ans_qa) / len(ans_qa) if ans_qa else 0.0

        # Attempted-only metrics (only where an answer was attempted, N = TN)
        attempted_qa = [r for r in ans_qa if not r.is_refusal]
        att_mean_f1 = sum(r.token_f1 for r in attempted_qa) / len(attempted_qa) if attempted_qa else 0.0
        att_mean_em = sum(r.exact_match for r in attempted_qa) / len(attempted_qa) if attempted_qa else 0.0

        # Overall benchmark metrics (all 64 questions, where correct refusal gets 1.0)
        overall_mean_f1 = sum(r.token_f1 for r in qa_results) / total_q if total_q > 0 else 0.0
        overall_mean_em = sum(r.exact_match for r in qa_results) / total_q if total_q > 0 else 0.0

        # Retrieval metrics (evaluated across answerable questions, N = 50)
        ans_ret = [r for r in ret_results if r.is_answerable]
        n_ret = len(ans_ret)
        hit_1 = sum(r.hit_at_k.get(1, 0.0) for r in ans_ret) / n_ret if n_ret > 0 else 0.0
        hit_5 = sum(r.hit_at_k.get(5, 0.0) for r in ans_ret) / n_ret if n_ret > 0 else 0.0
        hit_10 = sum(r.hit_at_k.get(10, 0.0) for r in ans_ret) / n_ret if n_ret > 0 else 0.0
        mrr = sum(r.mrr for r in ans_ret) / n_ret if n_ret > 0 else 0.0
        ndcg_5 = sum(r.ndcg_at_k.get(5, 0.0) for r in ans_ret) / n_ret if n_ret > 0 else 0.0

        doc_rec_5 = sum(r.doc_recall_at_k.get(5, 0.0) for r in ans_ret) / n_ret if n_ret > 0 else 0.0
        cross_doc_cov = sum(r.cross_doc_coverage for r in ans_ret) / n_ret if n_ret > 0 else 0.0
        path_acc = sum(r.multi_hop_path_accuracy for r in ans_ret) / n_ret if n_ret > 0 else 0.0

        # Temporal metrics
        agg_temp = temporal_evaluator.evaluate_suite(temp_results)
        temporal_compliance = agg_temp.temporal_compliance_rate
        mean_feir = agg_temp.stage_metrics.feir_post_filter if agg_temp.stage_metrics.feir_post_filter is not None else 0.0

        # Latencies
        mean_lat = statistics.mean(latencies) if latencies else 0.0
        median_lat = statistics.median(latencies) if latencies else 0.0
        min_lat = min(latencies) if latencies else 0.0
        max_lat = max(latencies) if latencies else 0.0

        return {
            "system_name": name,
            "denominators": {
                "total_questions": total_q,
                "answerable_questions": len(ans_qa),
                "unanswerable_questions": len(unans_qa),
                "attempted_answers": tn,
                "temporal_questions": 13,
            },
            "confusion_matrix": {
                "TP_correct_refusal": tp,
                "FN_hallucination": fn,
                "FP_false_refusal": fp,
                "TN_correct_decision_to_answer": tn,
            },
            "decision_accuracy": round(decision_acc, 4),
            "refusal_metrics": {
                "refusal_recall": round(refusal_recall, 4),
                "refusal_precision": round(refusal_precision, 4),
                "false_refusal_rate": round(false_refusal_rate, 4),
                "hallucination_rate": round(hallucination_rate, 4),
                "hallucinations_count": fn,
            },
            "answer_correctness": {
                "answerable_mean_token_f1": round(ans_mean_f1, 4),
                "answerable_exact_match": round(ans_mean_em, 4),
                "attempted_only_token_f1": round(att_mean_f1, 4),
                "attempted_only_exact_match": round(att_mean_em, 4),
                "overall_benchmark_token_f1": round(overall_mean_f1, 4),
                "overall_benchmark_exact_match": round(overall_mean_em, 4),
            },
            "retrieval_metrics": {
                "hit_at_1": round(hit_1, 4),
                "hit_at_5": round(hit_5, 4),
                "hit_at_10": round(hit_10, 4),
                "mrr": round(mrr, 4),
                "ndcg_at_5": round(ndcg_5, 4),
                "doc_recall_at_5": round(doc_rec_5, 4),
                "cross_doc_coverage": round(cross_doc_cov, 4),
                "multi_hop_path_accuracy": round(path_acc, 4),
            },
            "temporal_metrics": {
                "temporal_compliance_rate": round(temporal_compliance, 4),
                "mean_post_filter_feir": round(mean_feir, 4),
                "evaluated_temporal_queries": agg_temp.temporal_queries,
            },
            "latency_metrics": {
                "mean_seconds": round(mean_lat, 3),
                "median_seconds": round(median_lat, 3),
                "min_seconds": round(min_lat, 3),
                "max_seconds": round(max_lat, 3),
            },
        }

    b1_metrics = _compute_system_metrics("B1_Vector_Only_RAG", b1_qa_list, b1_ret_list, b1_temp_list, b1_latencies)
    k5_metrics = _compute_system_metrics("FinGraph_AI_K5", k5_qa_list, fg_ret_list, fg_temp_list, k5_latencies)
    k8_metrics = _compute_system_metrics("FinGraph_AI_K8", k8_qa_list, fg_ret_list, fg_temp_list, k8_latencies)

    # =========================================================================
    # D. Temporal Ablation (13 temporal queries with and without filtering)
    # =========================================================================
    temporal_ablation_res = None
    if run_ablation and is_full_benchmark:
        print("\n[Temporal Ablation] Evaluating point-in-time filtering causal impact...")
        temporal_qs = [q for q in benchmark_questions if q.category == "temporal_point_in_time"]
        total_with_feir = 0.0
        total_without_feir = 0.0
        total_rejections = 0

        for tq in temporal_qs:
            ev_with = fingraph.hybrid_retriever.retrieve(query=tq.question, reference_time=tq.reference_time, top_k=10)
            ev_without = fingraph.hybrid_retriever.retrieve(query=tq.question, reference_time=None, top_k=10)
            single_abl = temporal_evaluator.evaluate_ablation(
                with_filtering_evidence=ev_with,
                without_filtering_evidence=ev_without,
                reference_time=tq.reference_time,
            )
            total_with_feir += single_abl.with_filtering_feir
            total_without_feir += single_abl.without_filtering_feir
            total_rejections += single_abl.gate_rejections

        n_tqs = len(temporal_qs)
        mean_with = total_with_feir / n_tqs if n_tqs > 0 else 0.0
        mean_without = total_without_feir / n_tqs if n_tqs > 0 else 0.0
        diff = mean_without - mean_with
        rel_pct = (diff / mean_without * 100.0) if mean_without > 0 else 0.0

        temporal_ablation_res = {
            "temporal_queries_evaluated": n_tqs,
            "with_filtering_feir": round(mean_with, 4),
            "without_filtering_feir": round(mean_without, 4),
            "absolute_reduction": round(diff, 4),
            "relative_reduction_pct": round(rel_pct, 2),
            "future_items_blocked": total_rejections,
        }

    # =========================================================================
    # E. Parametric Leakage Probes (PLR)
    # =========================================================================
    print("\n[Parametric Probes] Evaluating intrinsic parametric cutoff leakage...")
    probes = get_standard_parametric_probes()
    leakage_rep: UnifiedLeakageReport = leakage_evaluator.evaluate_leakage(
        stage_metrics=temporal_evaluator.evaluate_suite(fg_temp_list).stage_metrics,
        probe_items=probes,
    )

    # =========================================================================
    # F. Save Outputs
    # =========================================================================
    summary_report = {
        "report_title": "FinGraph AI — Phase 6 Full Benchmark Verification Report",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "is_full_benchmark": is_full_benchmark,
        "total_questions": total_q,
        "answerable_questions": answerable_count,
        "unanswerable_questions": unanswerable_count,
        "systems": {
            "B1_Vector_Only_RAG": b1_metrics,
            "FinGraph_AI_K5": k5_metrics,
            "FinGraph_AI_K8": k8_metrics,
        },
        "temporal_ablation": temporal_ablation_res,
        "parametric_leakage_probes": leakage_rep.to_dict(),
    }

    # 1. Summary JSON
    summary_file = output_dir / "benchmark_evaluation_summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary_report, f, indent=2)

    # 2. Per-question CSV
    csv_file = output_dir / "per_question_results.csv"
    if csv_rows:
        with open(csv_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
            writer.writeheader()
            writer.writerows(csv_rows)

    # 3. Error analysis JSON
    error_file = output_dir / "error_analysis.json"
    with open(error_file, "w") as f:
        json.dump({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "total_questions": total_q,
            "entries": error_analysis_entries,
        }, f, indent=2)

    print("\n" + "=" * 80)
    print("PHASE 6 FULL BENCHMARK VERIFICATION COMPLETE")
    print("=" * 80)
    print(f"Artifacts saved to {output_dir}:")
    print(f"  Summary JSON:   {summary_file}")
    print(f"  Per-Q CSV:      {csv_file}")
    print(f"  Error Analysis: {error_file}")

    return summary_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run FinGraph AI Phase 6 Full Benchmark Verification")
    parser.add_argument("--output-dir", "-o", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--sample", "-s", type=int, default=None)
    parser.add_argument("--no-ablation", action="store_true")
    args = parser.parse_args()

    run_full_benchmark_verification(
        output_dir=args.output_dir,
        sample_size=args.sample,
        run_ablation=not args.no_ablation,
    )


if __name__ == "__main__":
    main()
