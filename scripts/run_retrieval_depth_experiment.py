"""
Controlled Retrieval-Depth Experiment on Multi-Hop and Cross-Document Questions.

Compares retrieval depth configurations:
- K5_DEFAULT: top_k=10, max_evidence_items=5
- K8_EXPANDED: top_k=10, max_evidence_items=8
- K10_DEEP: top_k=10, max_evidence_items=10

Target categories:
- relational_multi_hop (13 questions)
- cross_document_comparative (12 questions)
Total: 25 questions (18 answerable, 7 unanswerable)

Outputs saved strictly to results/phase6/retrieval_depth/:
- experiment_summary.json
- per_question_depth_results.csv
"""

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
from src.evaluation.benchmark import BenchmarkQuestion, load_benchmark
from src.evaluation.qa import QAEvaluator, SingleQAResult
from src.evaluation.retrieval import RetrievalEvaluator, SingleRetrievalResult
from src.evaluation.temporal import SingleTemporalResult, TemporalEvaluator
from src.retrieval.vector import VectorStore

OUTPUT_DIR = ROOT / "results" / "phase6" / "retrieval_depth"
TARGET_CATEGORIES = ("relational_multi_hop", "cross_document_comparative")

CONFIGURATIONS = [
    {
        "name": "K5_DEFAULT",
        "description": "Standard retrieval depth (5 evidence items for LLM context)",
        "top_k": 10,
        "max_evidence_items": 5,
    },
    {
        "name": "K8_EXPANDED",
        "description": "Expanded retrieval depth (8 evidence items for LLM context)",
        "top_k": 10,
        "max_evidence_items": 8,
    },
    {
        "name": "K10_DEEP",
        "description": "Deep retrieval depth (10 evidence items for LLM context)",
        "top_k": 10,
        "max_evidence_items": 10,
    },
]


def run_experiment(output_dir: Path = OUTPUT_DIR) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load target questions
    all_questions = load_benchmark(only_verified=True)
    target_questions = [q for q in all_questions if q.category in TARGET_CATEGORIES]

    num_total = len(target_questions)
    num_answerable = sum(1 for q in target_questions if q.answerable)
    num_unanswerable = sum(1 for q in target_questions if not q.answerable)

    print("=" * 80)
    print("FinGraph AI — Phase 6 Retrieval-Depth Experiment")
    print(f"Target Questions: {num_total} (Answerable: {num_answerable}, Unanswerable: {num_unanswerable})")
    print(f"Categories: {', '.join(TARGET_CATEGORIES)}")
    print(f"Configurations to test: {len(CONFIGURATIONS)}")
    print("=" * 80)

    # 2. Initialize Evaluators and Pipeline
    retrieval_evaluator = RetrievalEvaluator()
    qa_evaluator = QAEvaluator()
    temporal_evaluator = TemporalEvaluator()

    shared_vector_store = VectorStore(ROOT / "data" / "vector_store")
    pipeline = FinGraphPipeline(vector_store=shared_vector_store)

    csv_rows: list[dict[str, Any]] = []
    config_summaries: dict[str, Any] = {}

    rel_map = {q.question_id: q.relevant_chunk_ids for q in target_questions}

    for config in CONFIGURATIONS:
        cfg_name = config["name"]
        top_k = config["top_k"]
        max_items = config["max_evidence_items"]

        print(f"\n>>> Running Configuration: {cfg_name} (top_k={top_k}, max_evidence_items={max_items})")

        qa_results: list[SingleQAResult] = []
        ret_results: list[SingleRetrievalResult] = []
        temp_results: list[SingleTemporalResult] = []
        latencies: list[float] = []

        cfg_t0 = time.perf_counter()

        for idx, q in enumerate(target_questions, 1):
            t0 = time.perf_counter()
            try:
                pkg, ans = pipeline.answer(
                    query=q.question,
                    reference_time=q.reference_time,
                    top_k=top_k,
                    max_evidence_items=max_items,
                )
            except Exception as exc:
                print(f"  [{idx}/{num_total}] ERROR on {q.question_id}: {exc}")
                pkg = None
                ans = None

            t1 = time.perf_counter()
            latency = t1 - t0
            latencies.append(latency)

            evidence = pkg.all_evidence if pkg else []
            chunk_ids = [ev.chunk_id for ev in evidence if ev.chunk_id]
            doc_ids = sorted(list({ev.document_id for ev in evidence if ev.document_id}))
            paths = pkg.graph_paths if pkg else []

            ret_res = retrieval_evaluator.evaluate_single(
                question_id=q.question_id,
                retrieved_chunk_ids=chunk_ids,
                relevant_chunk_ids=q.relevant_chunk_ids,
                retrieved_doc_ids=doc_ids,
                relevant_doc_ids=q.relevant_document_ids,
                retrieved_graph_paths=paths,
                expected_graph_paths=q.expected_graph_path,
                is_answerable=q.answerable,
            )
            ret_results.append(ret_res)

            ans_text = ans.answer if ans else "Error during pipeline execution."
            confidence = ans.confidence if ans else 0.0
            evidence_ids = ans.evidence_ids if ans else []
            citations = ans.citations if ans else []
            validation_passed = ans.validation_status.get("is_valid", False) if (ans and ans.validation_status) else False

            qa_res = qa_evaluator.evaluate_single(
                question_id=q.question_id,
                predicted_answer=ans_text,
                ground_truth_answer=q.ground_truth_answer,
                is_answerable=q.answerable,
                confidence=confidence,
                evidence_ids_cited=evidence_ids,
                citations=citations,
                validation_passed=validation_passed,
            )
            qa_results.append(qa_res)

            temp_res = temporal_evaluator.evaluate_single(
                question_id=q.question_id,
                reference_time=q.reference_time,
                pre_filter_evidence=None,
                post_filter_evidence=evidence,
                context_evidence=evidence[:max_items],
                cited_evidence=citations,
            )
            temp_results.append(temp_res)

            # Refusal categorization
            if not q.answerable:
                refusal_outcome = "TP_correct_refusal" if qa_res.is_refusal else "FN_hallucination"
            else:
                refusal_outcome = "FP_false_refusal" if qa_res.is_refusal else "TN_correct_answer"

            decision_match = (refusal_outcome in ("TP_correct_refusal", "TN_correct_answer"))

            csv_rows.append({
                "config_name": cfg_name,
                "top_k": top_k,
                "max_evidence_items": max_items,
                "question_id": q.question_id,
                "category": q.category,
                "hop_level": q.hop_level,
                "answerable": q.answerable,
                "decision_match": decision_match,
                "refusal_outcome": refusal_outcome,
                "exact_match": qa_res.exact_match,
                "token_f1": qa_res.token_f1,
                "hit_at_5": ret_res.hit_at_k.get(5, 0.0),
                "mrr": ret_res.mrr,
                "path_accuracy": ret_res.multi_hop_path_accuracy,
                "feir": temp_res.post_filter_feir if temp_res.post_filter_feir is not None else 0.0,
                "evidence_count": len(evidence),
                "latency_s": round(latency, 3),
                "predicted_answer": ans_text[:120].replace("\n", " "),
                "ground_truth_answer": q.ground_truth_answer[:120].replace("\n", " "),
            })

            print(
                f"  [{idx:02d}/{num_total}] {q.question_id} ({q.category[:10]}): "
                f"outcome={refusal_outcome} | F1={qa_res.token_f1:.2f} | "
                f"PathAcc={ret_res.multi_hop_path_accuracy:.2f} | Time={latency:.2f}s"
            )

        cfg_duration = time.perf_counter() - cfg_t0

        # Aggregated metrics for this configuration
        agg_ret = retrieval_evaluator.evaluate_suite(ret_results)
        agg_qa = qa_evaluator.evaluate_suite(qa_results, rel_map)
        agg_temp = temporal_evaluator.evaluate_suite(temp_results)

        # Breakdowns
        tp = sum(1 for r in qa_results if r.is_refusal and not r.is_answerable)
        fn = sum(1 for r in qa_results if not r.is_refusal and not r.is_answerable)
        fp = sum(1 for r in qa_results if r.is_refusal and r.is_answerable)
        tn = sum(1 for r in qa_results if not r.is_refusal and r.is_answerable)

        decision_acc = (tp + tn) / num_total if num_total > 0 else 0.0
        refusal_rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        false_ref_rate = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        hallucination_rate = fn / (tp + fn) if (tp + fn) > 0 else 0.0

        ans_results = [r for r in qa_results if r.is_answerable]
        ans_em = sum(r.exact_match for r in ans_results) / len(ans_results) if ans_results else 0.0
        ans_f1 = sum(r.token_f1 for r in ans_results) / len(ans_results) if ans_results else 0.0

        avg_lat = sum(latencies) / len(latencies) if latencies else 0.0

        config_summaries[cfg_name] = {
            "config": config,
            "duration_seconds": round(cfg_duration, 2),
            "avg_latency_seconds": round(avg_lat, 3),
            "denominators": {
                "total_questions": num_total,
                "answerable": num_answerable,
                "unanswerable": num_unanswerable,
            },
            "decision_accuracy": round(decision_acc, 4),
            "confusion_matrix": {
                "TP_correct_refusal": tp,
                "FN_hallucination": fn,
                "FP_false_refusal": fp,
                "TN_correct_answer": tn,
            },
            "refusal_metrics": {
                "refusal_recall": round(refusal_rec, 4),
                "false_refusal_rate": round(false_ref_rate, 4),
                "hallucination_rate": round(hallucination_rate, 4),
                "hallucinations_count": fn,
            },
            "answerable_qa_metrics": {
                "answerable_count": num_answerable,
                "mean_exact_match": round(ans_em, 4),
                "mean_token_f1": round(ans_f1, 4),
            },
            "overall_qa_metrics": {
                "mean_exact_match": round(agg_qa.mean_exact_match, 4),
                "mean_token_f1": round(agg_qa.mean_token_f1, 4),
            },
            "retrieval_metrics": {
                "hit_at_5": round(agg_ret.mean_hit_at_k.get(5, 0.0), 4),
                "mrr": round(agg_ret.mean_mrr, 4),
                "path_accuracy": round(agg_ret.multi_hop_path_accuracy, 4),
            },
            "temporal_metrics": {
                "compliance_rate": round(agg_temp.temporal_compliance_rate, 4),
                "post_filter_feir": round(agg_temp.stage_metrics.feir_post_filter, 4) if agg_temp.stage_metrics.feir_post_filter is not None else 0.0,
            },
        }

    # Summary dictionary
    experiment_summary = {
        "experiment_title": "FinGraph AI Phase 6 Retrieval-Depth Experiment",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "target_categories": list(TARGET_CATEGORIES),
        "question_count": num_total,
        "answerable_count": num_answerable,
        "unanswerable_count": num_unanswerable,
        "configurations": config_summaries,
    }

    # Save summary JSON
    summary_path = output_dir / "experiment_summary.json"
    with open(summary_path, "w") as f:
        json.dump(experiment_summary, f, indent=2)

    # Save CSV
    csv_path = output_dir / "per_question_depth_results.csv"
    if csv_rows:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
            writer.writeheader()
            writer.writerows(csv_rows)

    print("\n" + "=" * 80)
    print("RETRIEVAL DEPTH EXPERIMENT RESULTS SUMMARY")
    print("=" * 80)
    print(f"{'Metric':<28} | {'K5_DEFAULT':<12} | {'K8_EXPANDED':<12} | {'K10_DEEP':<12}")
    print("-" * 72)

    def _fmt(val: Any) -> str:
        return f"{val:.4f}" if isinstance(val, float) else str(val)

    metrics_to_print = [
        ("Decision Accuracy", lambda c: c["decision_accuracy"]),
        ("Correct Answers (TN)", lambda c: c["confusion_matrix"]["TN_correct_answer"]),
        ("False Refusals (FP)", lambda c: c["confusion_matrix"]["FP_false_refusal"]),
        ("Correct Refusals (TP)", lambda c: c["confusion_matrix"]["TP_correct_refusal"]),
        ("Hallucinations (FN)", lambda c: c["confusion_matrix"]["FN_hallucination"]),
        ("Refusal Recall", lambda c: c["refusal_metrics"]["refusal_recall"]),
        ("False Refusal Rate", lambda c: c["refusal_metrics"]["false_refusal_rate"]),
        ("Ans-Only Token F1", lambda c: c["answerable_qa_metrics"]["mean_token_f1"]),
        ("Overall Token F1", lambda c: c["overall_qa_metrics"]["mean_token_f1"]),
        ("Retrieval Hit@5", lambda c: c["retrieval_metrics"]["hit_at_5"]),
        ("Retrieval MRR", lambda c: c["retrieval_metrics"]["mrr"]),
        ("Path Accuracy", lambda c: c["retrieval_metrics"]["path_accuracy"]),
        ("Temporal Compliance", lambda c: c["temporal_metrics"]["compliance_rate"]),
        ("Temporal FEIR", lambda c: c["temporal_metrics"]["post_filter_feir"]),
        ("Avg Latency (s)", lambda c: c["avg_latency_seconds"]),
    ]

    for label, getter in metrics_to_print:
        v5 = _fmt(getter(config_summaries["K5_DEFAULT"]))
        v8 = _fmt(getter(config_summaries["K8_EXPANDED"]))
        v10 = _fmt(getter(config_summaries["K10_DEEP"]))
        print(f"{label:<28} | {v5:<12} | {v8:<12} | {v10:<12}")

    print("-" * 72)
    print(f"\nArtifacts saved to:")
    print(f"  JSON: {summary_path}")
    print(f"  CSV:  {csv_path}")

    return experiment_summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Retrieval-Depth Experiment")
    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        default=OUTPUT_DIR,
        help="Directory to save experimental results",
    )
    args = parser.parse_args()
    run_experiment(output_dir=args.output_dir)


if __name__ == "__main__":
    main()
