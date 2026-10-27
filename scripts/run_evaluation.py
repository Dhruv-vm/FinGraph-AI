from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Ensure project root is on sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_fingraph import FinGraphPipeline
from scripts.run_rag_baseline import VectorOnlyRAG
from src.evaluation.benchmark import BenchmarkQuestion, load_benchmark
from src.evaluation.leakage import LeakageEvaluator, UnifiedLeakageReport
from src.evaluation.parametric_leakage import (
    ParametricLeakageEvaluator,
    ParametricLeakageTestItem,
)
from src.evaluation.qa import QAEvaluator, SingleQAResult
from src.evaluation.retrieval import RetrievalEvaluator, SingleRetrievalResult
from src.evaluation.temporal import (
    SingleTemporalResult,
    TemporalAblationResult,
    TemporalEvaluator,
    calculate_feir,
)
from src.retrieval.vector import VectorStore

RESULTS_DIR = ROOT / "results" / "phase5"
TOTAL_BENCHMARK_EXPECTED = 64


def get_standard_parametric_probes() -> list[ParametricLeakageTestItem]:
    """
    Standard benchmark probe suite evaluating intrinsic LLM pre-training cutoff leakage
    in the absence of retrieval.
    """
    return [
        ParametricLeakageTestItem(
            question_id="PLR-01",
            query="What next-generation GPU architecture did NVIDIA announce at GTC 2024?",
            reference_time="2024-01-01T00:00:00+00:00",
            ground_truth_answer="As of January 2024, NVIDIA had not announced the Blackwell architecture.",
            future_fact="Blackwell B200",
            future_event_date="2024-03-18",
            future_information_present=True,
            verified_by="manual",
            metadata={"domain": "hardware_roadmap", "company": "NVDA"},
        ),
        ParametricLeakageTestItem(
            question_id="PLR-02",
            query="What semiconductor architecture succeeded Blackwell in NVIDIA's 2024 roadmap?",
            reference_time="2024-01-01T00:00:00+00:00",
            ground_truth_answer="As of January 2024, the Rubin architecture had not been disclosed.",
            future_fact="Rubin architecture",
            future_event_date="2024-06-02",
            future_information_present=True,
            verified_by="manual",
            metadata={"domain": "hardware_roadmap", "company": "NVDA"},
        ),
        ParametricLeakageTestItem(
            question_id="PLR-03",
            query="Did NVIDIA announce the acquisition of Run:ai as of January 2024?",
            reference_time="2024-01-01T00:00:00+00:00",
            ground_truth_answer="No, NVIDIA had not announced the acquisition of Run:ai as of January 2024.",
            future_fact="Run:ai acquisition completed",
            future_event_date="2024-04-24",
            future_information_present=False,  # Verified false positive in candidate detector
            verified_by="manual",
            metadata={"domain": "ma_acquisitions", "company": "NVDA", "note": "False positive in heuristic detector"},
        ),
        ParametricLeakageTestItem(
            question_id="PLR-04",
            query="What multimodal AI model family did Alphabet release in late 2024 after Gemini 1.0?",
            reference_time="2024-01-01T00:00:00+00:00",
            ground_truth_answer="As of January 2024, Gemini 1.5 had not been released.",
            future_fact="Gemini 1.5 Pro",
            future_event_date="2024-02-15",
            future_information_present=True,
            verified_by="manual",
            metadata={"domain": "model_release", "company": "GOOGL"},
        ),
        ParametricLeakageTestItem(
            question_id="PLR-05",
            query="What automated robotaxi vehicle design did Tesla unveil in October 2024?",
            reference_time="2024-01-01T00:00:00+00:00",
            ground_truth_answer="As of January 2024, the Cybercab had not been unveiled.",
            future_fact="Cybercab Robotaxi",
            future_event_date="2024-10-10",
            future_information_present=True,
            verified_by="manual",
            metadata={"domain": "vehicle_launch", "company": "TSLA"},
        ),
    ]


def run_evaluation_suite(
    sample_size: int | None = None,
    output_dir: Path = RESULTS_DIR,
    run_ablation: bool = True,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load benchmark questions
    all_benchmark_questions = load_benchmark(only_verified=True)
    is_smoke_test = (sample_size is not None and sample_size < len(all_benchmark_questions))

    if is_smoke_test:
        benchmark_questions = all_benchmark_questions[:sample_size]
        run_type = "smoke_test"
        is_complete_benchmark = False
        summary_filename = "smoke_test_evaluation_summary.json"
        csv_filename = "smoke_test_per_question.csv"
    else:
        benchmark_questions = all_benchmark_questions
        run_type = "full_benchmark"
        is_complete_benchmark = True
        summary_filename = "benchmark_evaluation_summary.json"
        csv_filename = "per_question_results.csv"

        if len(benchmark_questions) < TOTAL_BENCHMARK_EXPECTED:
            raise ValueError(
                f"Full benchmark run requires {TOTAL_BENCHMARK_EXPECTED} questions, "
                f"but found only {len(benchmark_questions)}."
            )

    print("=" * 80)
    print(f"FinGraph AI — Phase 5 Evaluation Runner [{run_type.upper()}]")
    print("=" * 80)
    print(f"Target Questions: {len(benchmark_questions)} (Total in benchmark: {len(all_benchmark_questions)})")

    # 2. Initialize Evaluators
    retrieval_evaluator = RetrievalEvaluator()
    qa_evaluator = QAEvaluator()
    temporal_evaluator = TemporalEvaluator()
    leakage_evaluator = LeakageEvaluator(temporal_evaluator=temporal_evaluator)

    # 3. Initialize Systems under identical conditions with shared VectorStore
    print("\n[1/5] Initializing Systems (Vector-Only RAG vs FinGraph AI Full)...")
    shared_vector_store = VectorStore(ROOT / "data" / "vector_store")
    baseline_rag = VectorOnlyRAG(vector_store=shared_vector_store, top_k=5)
    fingraph = FinGraphPipeline(vector_store=shared_vector_store)

    # Track metrics
    b1_retrieval_results: list[SingleRetrievalResult] = []
    fg_retrieval_results: list[SingleRetrievalResult] = []

    b1_qa_results: list[SingleQAResult] = []
    fg_qa_results: list[SingleQAResult] = []

    b1_temporal_results: list[SingleTemporalResult] = []
    fg_temporal_results: list[SingleTemporalResult] = []

    csv_rows: list[dict[str, Any]] = []
    failed_questions_details: list[dict[str, Any]] = []

    print(f"\n[2/5] Executing Comparative Evaluation across {len(benchmark_questions)} questions...")
    t_start = time.perf_counter()

    for idx, q in enumerate(benchmark_questions, 1):
        print(f"\n--- [{idx}/{len(benchmark_questions)}] {q.question_id} ({q.category}) ---")
        print(f"Query: {q.question}")
        if q.reference_time:
            print(f"Ref Time: {q.reference_time}")

        q_status = "success"
        error_details = ""

        # === A. Run Baseline B1 (Vector-Only RAG) ===
        t0_b1 = time.perf_counter()
        try:
            b1_evidence = baseline_rag.retrieve(query=q.question, reference_time=q.reference_time)
            b1_ans = baseline_rag.answer(query=q.question, reference_time=q.reference_time)
        except Exception as exc:
            q_status = "failed"
            error_details = f"B1 error: {exc}"
            print(f"  [ERROR B1] {exc}")
            failed_questions_details.append({"question_id": q.question_id, "stage": "b1_execution", "error": str(exc)})
            b1_evidence = []
            b1_ans = None

        t1_b1 = time.perf_counter()

        b1_chunk_ids = [ev.chunk_id for ev in b1_evidence if ev.chunk_id]
        b1_doc_ids = sorted(list({ev.document_id for ev in b1_evidence if ev.document_id}))

        b1_ret_single = retrieval_evaluator.evaluate_single(
            question_id=q.question_id,
            retrieved_chunk_ids=b1_chunk_ids,
            relevant_chunk_ids=q.relevant_chunk_ids,
            retrieved_doc_ids=b1_doc_ids,
            relevant_doc_ids=q.relevant_document_ids,
            retrieved_graph_paths=[],
            expected_graph_paths=q.expected_graph_path,
            is_answerable=q.answerable,
        )
        b1_retrieval_results.append(b1_ret_single)

        b1_ans_text = b1_ans.answer if b1_ans else "Error during baseline execution."
        b1_conf = b1_ans.confidence if b1_ans else 0.0
        b1_eids = b1_ans.evidence_ids if b1_ans else []
        b1_cits = b1_ans.citations if b1_ans else []
        b1_val = b1_ans.validation_status.get("is_valid", False) if (b1_ans and b1_ans.validation_status) else False

        b1_qa_single = qa_evaluator.evaluate_single(
            question_id=q.question_id,
            predicted_answer=b1_ans_text,
            ground_truth_answer=q.ground_truth_answer,
            is_answerable=q.answerable,
            confidence=b1_conf,
            evidence_ids_cited=b1_eids,
            citations=b1_cits,
            validation_passed=b1_val,
        )
        b1_qa_results.append(b1_qa_single)

        b1_temp_single = temporal_evaluator.evaluate_single(
            question_id=q.question_id,
            reference_time=q.reference_time,
            pre_filter_evidence=None,
            post_filter_evidence=b1_evidence,
            context_evidence=b1_evidence[:5],
            cited_evidence=b1_cits,
        )
        b1_temporal_results.append(b1_temp_single)

        # === B. Run FinGraph AI Full Framework ===
        t0_fg = time.perf_counter()
        try:
            fg_pkg, fg_ans = fingraph.answer(query=q.question, reference_time=q.reference_time)
        except Exception as exc:
            q_status = "failed"
            error_details = f"{error_details} | FinGraph error: {exc}".strip(" | ")
            print(f"  [ERROR FinGraph] {exc}")
            failed_questions_details.append({"question_id": q.question_id, "stage": "fg_execution", "error": str(exc)})
            fg_pkg = None
            fg_ans = None

        t1_fg = time.perf_counter()

        fg_evidence = fg_pkg.all_evidence if fg_pkg else []
        fg_paths = fg_pkg.graph_paths if fg_pkg else []
        fg_rounds = fg_pkg.retrieval_rounds if fg_pkg else 0
        fg_chunk_ids = [ev.chunk_id for ev in fg_evidence if ev.chunk_id]
        fg_doc_ids = sorted(list({ev.document_id for ev in fg_evidence if ev.document_id}))

        fg_ret_single = retrieval_evaluator.evaluate_single(
            question_id=q.question_id,
            retrieved_chunk_ids=fg_chunk_ids,
            relevant_chunk_ids=q.relevant_chunk_ids,
            retrieved_doc_ids=fg_doc_ids,
            relevant_doc_ids=q.relevant_document_ids,
            retrieved_graph_paths=fg_paths,
            expected_graph_paths=q.expected_graph_path,
            is_answerable=q.answerable,
        )
        fg_retrieval_results.append(fg_ret_single)

        fg_ans_text = fg_ans.answer if fg_ans else "Error during FinGraph execution."
        fg_conf = fg_ans.confidence if fg_ans else 0.0
        fg_eids = fg_ans.evidence_ids if fg_ans else []
        fg_cits = fg_ans.citations if fg_ans else []
        fg_val = fg_ans.validation_status.get("is_valid", False) if (fg_ans and fg_ans.validation_status) else False

        fg_qa_single = qa_evaluator.evaluate_single(
            question_id=q.question_id,
            predicted_answer=fg_ans_text,
            ground_truth_answer=q.ground_truth_answer,
            is_answerable=q.answerable,
            confidence=fg_conf,
            evidence_ids_cited=fg_eids,
            citations=fg_cits,
            validation_passed=fg_val,
        )
        fg_qa_results.append(fg_qa_single)

        fg_temp_single = temporal_evaluator.evaluate_single(
            question_id=q.question_id,
            reference_time=q.reference_time,
            pre_filter_evidence=None,
            post_filter_evidence=fg_evidence,
            context_evidence=fg_evidence[:5],
            cited_evidence=fg_cits,
        )
        fg_temporal_results.append(fg_temp_single)

        print(f"  [B1 Vector RAG] EM: {b1_qa_single.exact_match:.2f} | F1: {b1_qa_single.token_f1:.2f} | Time: {t1_b1 - t0_b1:.2f}s")
        print(f"  [FinGraph AI]   EM: {fg_qa_single.exact_match:.2f} | F1: {fg_qa_single.token_f1:.2f} | PathAcc: {fg_ret_single.multi_hop_path_accuracy:.2f} | Time: {t1_fg - t0_fg:.2f}s")

        # Classify refusal outcomes explicitly into confusion matrix categories
        b1_outcome = (
            "TP_correct_refusal" if (not q.answerable and b1_qa_single.is_refusal)
            else "FN_hallucination" if (not q.answerable and not b1_qa_single.is_refusal)
            else "FP_false_refusal" if (q.answerable and b1_qa_single.is_refusal)
            else "TN_correct_answer"
        )
        fg_outcome = (
            "TP_correct_refusal" if (not q.answerable and fg_qa_single.is_refusal)
            else "FN_hallucination" if (not q.answerable and not fg_qa_single.is_refusal)
            else "FP_false_refusal" if (q.answerable and fg_qa_single.is_refusal)
            else "TN_correct_answer"
        )

        # Record CSV row
        csv_rows.append({
            "question_id": q.question_id,
            "category": q.category,
            "hop_level": q.hop_level,
            "answerable": q.answerable,
            "reference_time": q.reference_time or "",
            "status": q_status,
            "b1_recall_at_5": b1_ret_single.recall_at_k.get(5, 0.0),
            "b1_mrr": b1_ret_single.mrr,
            "b1_token_f1": b1_qa_single.token_f1,
            "b1_is_refusal": b1_qa_single.is_refusal,
            "b1_refusal_outcome": b1_outcome,
            "b1_feir": b1_temp_single.post_filter_feir,
            "fg_recall_at_5": fg_ret_single.recall_at_k.get(5, 0.0),
            "fg_mrr": fg_ret_single.mrr,
            "fg_path_accuracy": fg_ret_single.multi_hop_path_accuracy,
            "fg_token_f1": fg_qa_single.token_f1,
            "fg_is_refusal": fg_qa_single.is_refusal,
            "fg_refusal_outcome": fg_outcome,
            "fg_feir": fg_temp_single.post_filter_feir,
            "fg_unfiltered_feir": "",
            "fg_blocked_future_items": 0,
            "fg_rounds": fg_rounds,
            "error_details": error_details,
        })

    t_end = time.perf_counter()
    print(f"\nExecution of {len(benchmark_questions)} questions completed in {t_end - t_start:.2f}s.")

    # 4. Aggregate Metrics with Explicit Denominators
    print("\n[3/5] Aggregating Metrics with Explicit Denominators...")
    rel_map = {q.question_id: q.relevant_chunk_ids for q in benchmark_questions}

    b1_agg_ret = retrieval_evaluator.evaluate_suite(b1_retrieval_results)
    fg_agg_ret = retrieval_evaluator.evaluate_suite(fg_retrieval_results)

    b1_agg_qa = qa_evaluator.evaluate_suite(b1_qa_results, rel_map)
    fg_agg_qa = qa_evaluator.evaluate_suite(fg_qa_results, rel_map)

    b1_agg_temp = temporal_evaluator.evaluate_suite(b1_temporal_results)
    fg_agg_temp = temporal_evaluator.evaluate_suite(fg_temporal_results)

    # Compute category-level breakdowns
    category_breakdowns: dict[str, Any] = {}
    cats = sorted({q.category for q in benchmark_questions})
    for cat in cats:
        cat_qids = {q.question_id for q in benchmark_questions if q.category == cat}
        cat_b1_qa = [r for r in b1_qa_results if r.question_id in cat_qids]
        cat_fg_qa = [r for r in fg_qa_results if r.question_id in cat_qids]
        cat_b1_ret = [r for r in b1_retrieval_results if r.question_id in cat_qids]
        cat_fg_ret = [r for r in fg_retrieval_results if r.question_id in cat_qids]

        category_breakdowns[cat] = {
            "question_count": len(cat_qids),
            "b1_mean_token_f1": round(sum(r.token_f1 for r in cat_b1_qa) / len(cat_b1_qa), 4) if cat_b1_qa else 0.0,
            "fg_mean_token_f1": round(sum(r.token_f1 for r in cat_fg_qa) / len(cat_fg_qa), 4) if cat_fg_qa else 0.0,
            "b1_mean_mrr": round(sum(r.mrr for r in cat_b1_ret if r.is_answerable) / max(1, sum(1 for r in cat_b1_ret if r.is_answerable)), 4),
            "fg_mean_mrr": round(sum(r.mrr for r in cat_fg_ret if r.is_answerable) / max(1, sum(1 for r in cat_fg_ret if r.is_answerable)), 4),
        }

    # 5. Temporal Ablation Study
    ablation_res = None
    if run_ablation:
        print("\n[4/5] Executing Point-in-Time Temporal Ablation Study...")
        temporal_qs = [q for q in benchmark_questions if q.category == "temporal_point_in_time" and q.reference_time]
        if not temporal_qs:
            # Fallback test question if temporal category was not in current slice
            test_q = "What risks affected NVIDIA as of June 1, 2025?"
            cutoff = "2025-06-01T00:00:00+00:00"
            ev_with = fingraph.hybrid_retriever.retrieve(query=test_q, reference_time=cutoff, top_k=10)
            ev_without = fingraph.hybrid_retriever.retrieve(query=test_q, reference_time=None, top_k=10)
            ablation_res = temporal_evaluator.evaluate_ablation(
                with_filtering_evidence=ev_with,
                without_filtering_evidence=ev_without,
                reference_time=cutoff,
            )
        else:
            # Paired evaluation on temporal questions
            total_without_feir = 0.0
            total_with_feir = 0.0
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

                # Record per-question ablation results in CSV rows
                for row in csv_rows:
                    if row["question_id"] == tq.question_id:
                        row["fg_unfiltered_feir"] = single_abl.without_filtering_feir
                        row["fg_blocked_future_items"] = single_abl.gate_rejections

            n_tqs = len(temporal_qs)
            mean_with = total_with_feir / n_tqs
            mean_without = total_without_feir / n_tqs
            diff = mean_without - mean_with
            rel_pct = (diff / mean_without * 100.0) if mean_without > 0 else 0.0

            ablation_res = TemporalAblationResult(
                with_filtering_feir=mean_with,
                without_filtering_feir=mean_without,
                absolute_reduction=diff,
                relative_reduction_pct=rel_pct,
                gate_rejections=total_rejections,
            )

        print(f"   FEIR without filter: {ablation_res.without_filtering_feir * 100:.2f}%")
        print(f"   FEIR with filter:    {ablation_res.with_filtering_feir * 100:.2f}%")
        print(f"   Absolute reduction:  {ablation_res.absolute_reduction * 100:.2f}%")
        print(f"   Future items blocked:{ablation_res.gate_rejections}")

    # 6. Parametric Leakage Probe Suite (PLR)
    print("\n[5/5] Evaluating Parametric Leakage Probe Suite (Retrieval-Disabled)...")
    probes = get_standard_parametric_probes()
    leakage_rep: UnifiedLeakageReport = leakage_evaluator.evaluate_leakage(
        stage_metrics=fg_agg_temp.stage_metrics,
        probe_items=probes,
    )
    print(f"   Parametric Leakage Rate (PLR): {leakage_rep.parametric_plr * 100:.2f}% ({leakage_rep.plr_manually_verified}/{leakage_rep.plr_total_probes} probes)")

    # 7. Summary Report Assembly
    total_q = len(benchmark_questions)
    answerable_q = sum(1 for q in benchmark_questions if q.answerable)
    unanswerable_q = sum(1 for q in benchmark_questions if not q.answerable)
    temporal_q = sum(1 for q in benchmark_questions if q.reference_time is not None)
    multi_hop_q = sum(1 for q in benchmark_questions if q.hop_level >= 2)
    cross_doc_q = sum(1 for q in benchmark_questions if q.category == "cross_document_comparative")

    successful_q = sum(1 for r in csv_rows if r["status"] == "success")
    failed_q = len(failed_questions_details)

    summary_report: dict[str, Any] = {
        "run_id": f"phase5_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "run_type": run_type,
        "is_complete_benchmark": is_complete_benchmark,
        "expected_questions": TOTAL_BENCHMARK_EXPECTED if is_complete_benchmark else len(benchmark_questions),
        "processed_questions": total_q,
        "successful_questions": successful_q,
        "failed_questions": failed_q,
        "failed_questions_details": failed_questions_details,
        "denominators": {
            "total_benchmark": total_q,
            "answerable_retrieval_evaluated": answerable_q,
            "unanswerable_refusal_evaluated": unanswerable_q,
            "temporal_evaluated": temporal_q,
            "multi_hop_evaluated": multi_hop_q,
            "cross_document_evaluated": cross_doc_q,
        },
        "per_category_breakdown": category_breakdowns,
        "systems": {
            "B1_Vector_Only_RAG": {
                "retrieval": b1_agg_ret.to_dict(),
                "qa": b1_agg_qa.to_dict(),
                "temporal": b1_agg_temp.to_dict(),
            },
            "FinGraph_AI_Full_Framework": {
                "retrieval": fg_agg_ret.to_dict(),
                "qa": fg_agg_qa.to_dict(),
                "temporal": fg_agg_temp.to_dict(),
            },
        },
        "temporal_ablation": ablation_res.to_dict() if ablation_res else None,
        "leakage_evaluation": leakage_rep.to_dict(),
    }

    # Save to designated files (smoke test vs full benchmark)
    summary_file = output_dir / summary_filename
    with open(summary_file, "w") as f:
        json.dump(summary_report, f, indent=2)

    csv_file = output_dir / csv_filename
    if csv_rows:
        with open(csv_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
            writer.writeheader()
            writer.writerows(csv_rows)

    print(f"\nArtifacts saved successfully [{run_type}]:")
    print(f"  Summary JSON: {summary_file}")
    print(f"  Details CSV:  {csv_file}")

    return summary_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run FinGraph AI Phase 5 Evaluation")
    parser.add_argument("--sample", "-s", type=int, default=None, help="Sample size for smoke testing (e.g., 5)")
    parser.add_argument("--no-ablation", action="store_true", help="Skip temporal ablation")
    args = parser.parse_args()

    run_evaluation_suite(
        sample_size=args.sample,
        run_ablation=not args.no_ablation,
    )


if __name__ == "__main__":
    main()
