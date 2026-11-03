"""
Controlled Retrieval-Recall Experiment across Candidate Pool Depths.

Compares candidate pool sizes entering reranking:
- POOL_20_DEFAULT: candidate_pool_size=20 entering reranker (40 initial candidates fetched)
- POOL_25_EXPANDED: candidate_pool_size=25 entering reranker (50 initial candidates fetched)
- POOL_50_DEEP: candidate_pool_size=50 entering reranker (100 initial candidates fetched)

Holding strictly constant:
- Generation context: top 8 evidence items (max_evidence_items=8)
- Model: qwen3.5:4b (temperature=0.0)
- Prompt template and instructions
- Evidence sufficiency gate and temporal point-in-time filtering

Outputs saved strictly to results/phase6/retrieval_recall/:
- experiment_summary.json
- per_question_recall_results.csv
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

OUTPUT_DIR = ROOT / "results" / "phase6" / "retrieval_recall"

CONFIGURATIONS = [
    {
        "name": "POOL_20_DEFAULT",
        "description": "Standard candidate pool (20 candidates entering reranker, 40 fetched)",
        "candidate_pool_size": 20,
        "max_evidence_items": 8,
    },
    {
        "name": "POOL_25_EXPANDED",
        "description": "Expanded candidate pool (25 candidates entering reranker, 50 fetched)",
        "candidate_pool_size": 25,
        "max_evidence_items": 8,
    },
    {
        "name": "POOL_50_DEEP",
        "description": "Deep candidate pool (50 candidates entering reranker, 100 fetched)",
        "candidate_pool_size": 50,
        "max_evidence_items": 8,
    },
]


def select_stratified_subset(all_questions: list[BenchmarkQuestion], total_target: int = 15) -> list[BenchmarkQuestion]:
    """Select a stratified representative subset across the 5 benchmark categories."""
    by_cat: dict[str, list[BenchmarkQuestion]] = defaultdict(list)
    for q in all_questions:
        by_cat[q.category].append(q)

    per_cat = max(1, total_target // len(by_cat))
    selected: list[BenchmarkQuestion] = []

    for cat, qs in sorted(by_cat.items()):
        # Select answerable and unanswerable representation
        ans = [q for q in qs if q.answerable]
        unans = [q for q in qs if not q.answerable]
        
        # Pick 2 answerable and 1 unanswerable if available
        cat_picks = ans[:2] + unans[:1]
        if len(cat_picks) < per_cat and len(ans) > 2:
            cat_picks.extend(ans[2 : 2 + (per_cat - len(cat_picks))])
        selected.extend(cat_picks[:per_cat])

    return selected


def run_experiment(
    output_dir: Path = OUTPUT_DIR,
    sample_size: int | None = None,
    full_benchmark: bool = False,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)

    all_questions = load_benchmark(only_verified=True)

    if full_benchmark:
        qa_questions = all_questions
        qa_scope = "full_64"
    elif sample_size is not None and sample_size < len(all_questions):
        qa_questions = all_questions[:sample_size]
        qa_scope = f"head_{sample_size}"
    else:
        qa_questions = select_stratified_subset(all_questions, total_target=15)
        qa_scope = f"stratified_{len(qa_questions)}"

    print("=" * 80)
    print("FinGraph AI — Phase 6.1 Retrieval-Recall Experiment")
    print(f"Retrieval Evaluation Scope: All {len(all_questions)} benchmark questions")
    print(f"QA Generation Evaluation Scope: {len(qa_questions)} questions ({qa_scope})")
    print(f"Configurations to test: {[c['name'] for c in CONFIGURATIONS]}")
    print(f"Output Directory: {output_dir}")
    print("=" * 80)

    # Evaluators
    ret_evaluator = RetrievalEvaluator(k_values=(1, 3, 5, 10))
    qa_evaluator = QAEvaluator()
    temp_evaluator = TemporalEvaluator()

    # Systems
    shared_vector_store = VectorStore(ROOT / "data" / "vector_store")
    pipeline = FinGraphPipeline(vector_store=shared_vector_store)

    experiment_summary: dict[str, Any] = {
        "metadata": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "llm_model": pipeline.generator.model,
            "temperature": pipeline.generator.temperature,
            "generation_max_evidence_items": 8,
            "qa_scope": qa_scope,
            "total_questions_in_benchmark": len(all_questions),
            "qa_questions_evaluated": len(qa_questions),
        },
        "retrieval_benchmark_64": {},
        "configurations": {},
    }

    # =========================================================================
    # Part 1: Full 64-Question Retrieval-Recall Evaluation for each Pool
    # =========================================================================
    print("\n--- Part 1: Full 64-Question Retrieval Evaluation ---")
    retrieval_records_by_config: dict[str, list[SingleRetrievalResult]] = {}
    temporal_records_by_config: dict[str, list[SingleTemporalResult]] = {}
    actual_candidate_counts: dict[str, list[int]] = defaultdict(list)

    for cfg in CONFIGURATIONS:
        c_name = cfg["name"]
        pool_size = cfg["candidate_pool_size"]
        print(f"Evaluating 64-question retrieval for {c_name} (candidate_pool_size={pool_size})...")

        ret_results: list[SingleRetrievalResult] = []
        temp_results: list[SingleTemporalResult] = []

        for q in all_questions:
            ref_str = q.reference_time.isoformat() if hasattr(q.reference_time, "isoformat") else q.reference_time
            pkg = pipeline.agent.run(
                query=q.question,
                reference_time=ref_str,
                top_k=10,
                candidate_pool_size=pool_size,
            )

            # Record actual candidates count in Round 1 trace
            r1_count = pkg.trace[0]["result_count"] if pkg.trace else len(pkg.evidence)
            actual_candidate_counts[c_name].append(r1_count)

            cids = [ev.chunk_id for ev in pkg.evidence if ev.chunk_id]
            dids = sorted(list({ev.document_id for ev in pkg.evidence if ev.document_id}))

            ret_res = ret_evaluator.evaluate_single(
                question_id=q.question_id,
                retrieved_chunk_ids=cids,
                relevant_chunk_ids=q.relevant_chunk_ids,
                retrieved_doc_ids=dids,
                relevant_doc_ids=q.relevant_document_ids,
                retrieved_graph_paths=[],
                expected_graph_paths=q.expected_graph_path,
                is_answerable=q.answerable,
            )
            ret_results.append(ret_res)

            temp_res = temp_evaluator.evaluate_single(
                question_id=q.question_id,
                reference_time=q.reference_time,
                pre_filter_evidence=None,
                post_filter_evidence=pkg.evidence,
                context_evidence=pkg.evidence[:8],
                cited_evidence=[],
            )
            temp_results.append(temp_res)

        retrieval_records_by_config[c_name] = ret_results
        temporal_records_by_config[c_name] = temp_results

        agg_ret = ret_evaluator.evaluate_suite(ret_results)
        agg_temp = temp_evaluator.evaluate_suite(temp_results)

        experiment_summary["retrieval_benchmark_64"][c_name] = {
            "candidate_pool_size": pool_size,
            "mean_candidates_entering_reranker": sum(actual_candidate_counts[c_name]) / len(actual_candidate_counts[c_name]),
            "chunk_hit_at_1": agg_ret.mean_hit_at_k.get(1, 0.0),
            "chunk_hit_at_5": agg_ret.mean_hit_at_k.get(5, 0.0),
            "chunk_hit_at_10": agg_ret.mean_hit_at_k.get(10, 0.0),
            "mrr": agg_ret.mean_mrr,
            "ndcg_at_5": agg_ret.mean_ndcg_at_k.get(5, 0.0),
            "doc_recall_at_5": agg_ret.mean_doc_recall_at_k.get(5, 0.0),
            "cross_doc_coverage": agg_ret.cross_doc_coverage,
            "feir_post_filter": agg_temp.stage_metrics.feir_post_filter,
            "feir_context": agg_temp.stage_metrics.feir_context,
            "temporal_compliance_rate": agg_temp.temporal_compliance_rate,
        }

    # =========================================================================
    # Part 2: Controlled QA Generation Evaluation
    # =========================================================================
    print(f"\n--- Part 2: QA Generation Evaluation on {len(qa_questions)} Questions ---")
    per_question_rows: list[dict[str, Any]] = []

    # Map question_id -> BenchmarkQuestion
    q_by_id = {q.question_id: q for q in qa_questions}

    cache_path = output_dir / ".qa_cache.json"
    qa_cache: dict[str, Any] = {}
    if cache_path.exists():
        try:
            with open(cache_path) as cf:
                qa_cache = json.load(cf)
        except Exception:
            qa_cache = {}

    qa_results_by_config: dict[str, list[SingleQAResult]] = defaultdict(list)
    latencies_by_config: dict[str, list[float]] = defaultdict(list)
    sufficiency_by_config: dict[str, dict[str, int]] = defaultdict(lambda: {"ans_pass": 0, "ans_fail": 0, "unans_pass": 0, "unans_fail": 0})
    per_q_outcomes: dict[str, dict[str, str]] = defaultdict(dict)

    for q_idx, q in enumerate(qa_questions, 1):
        print(f"\n[{q_idx}/{len(qa_questions)}] {q.question_id} ({q.category}): {q.question[:60]}...")
        ref_str = q.reference_time.isoformat() if hasattr(q.reference_time, "isoformat") else q.reference_time

        row: dict[str, Any] = {
            "question_id": q.question_id,
            "category": q.category,
            "hop_level": q.hop_level,
            "answerable": q.answerable,
            "reference_time": ref_str,
            "ground_truth_answer": q.ground_truth_answer,
        }

        for cfg in CONFIGURATIONS:
            c_name = cfg["name"]
            pool_size = cfg["candidate_pool_size"]

            cache_key = f"{q.question_id}:{c_name}"
            if cache_key in qa_cache:
                entry = qa_cache[cache_key]
                ans_text = entry["answer"]
                confidence = entry.get("confidence", 0.0)
                citations = entry.get("citations", [])
                evidence_ids = entry.get("evidence_ids", [])
                val_status = entry.get("validation_status", {})
                lat = entry.get("latency_s", 0.0)
                pkg_sufficient = entry.get("sufficient", True)
            else:
                t0 = time.perf_counter()
                pkg = pipeline.agent.run(
                    query=q.question,
                    reference_time=ref_str,
                    top_k=10,
                    candidate_pool_size=pool_size,
                )
                # Generate answer holding max_evidence_items=8 constant
                ans = pipeline.generator.generate(package=pkg, max_evidence_items=8)
                lat = time.perf_counter() - t0
                ans_text = ans.answer
                confidence = ans.confidence
                citations = ans.citations
                evidence_ids = ans.evidence_ids
                val_status = ans.validation_status
                pkg_sufficient = pkg.sufficient

                qa_cache[cache_key] = {
                    "answer": ans_text,
                    "confidence": confidence,
                    "citations": citations,
                    "evidence_ids": evidence_ids,
                    "validation_status": val_status,
                    "latency_s": lat,
                    "sufficient": pkg_sufficient,
                }
                with open(cache_path, "w") as cf:
                    json.dump(qa_cache, cf, indent=2)

            latencies_by_config[c_name].append(lat)

            # Record sufficiency gate behavior
            if q.answerable:
                if pkg_sufficient:
                    sufficiency_by_config[c_name]["ans_pass"] += 1
                else:
                    sufficiency_by_config[c_name]["ans_fail"] += 1
            else:
                if pkg_sufficient:
                    sufficiency_by_config[c_name]["unans_pass"] += 1
                else:
                    sufficiency_by_config[c_name]["unans_fail"] += 1

            # Evaluate QA
            qa_res = qa_evaluator.evaluate_single(
                question_id=q.question_id,
                predicted_answer=ans_text,
                ground_truth_answer=q.ground_truth_answer,
                is_answerable=q.answerable,
                confidence=confidence,
                evidence_ids_cited=evidence_ids,
                citations=citations,
                validation_passed=val_status.get("is_valid", False) if val_status else False,
            )
            qa_results_by_config[c_name].append(qa_res)

            # Classify outcome
            if not q.answerable:
                outcome = "TN_refusal" if qa_res.is_refusal else "FN_hallucination"
            else:
                outcome = "FP_refusal" if qa_res.is_refusal else "TP_answered"
            per_q_outcomes[c_name][q.question_id] = outcome

            # Add to row
            prefix = c_name.lower()
            row[f"{prefix}_outcome"] = outcome
            row[f"{prefix}_token_f1"] = round(qa_res.token_f1, 4)
            row[f"{prefix}_exact_match"] = round(qa_res.exact_match, 4)
            row[f"{prefix}_is_refusal"] = qa_res.is_refusal
            row[f"{prefix}_sufficient"] = pkg_sufficient
            row[f"{prefix}_latency_s"] = round(lat, 3)
            row[f"{prefix}_answer"] = ans_text[:120].replace("\n", " ")

        per_question_rows.append(row)

    # =========================================================================
    # Part 3: Aggregate Metrics & Summaries
    # =========================================================================
    for cfg in CONFIGURATIONS:
        c_name = cfg["name"]
        pool_size = cfg["candidate_pool_size"]
        qa_list = qa_results_by_config[c_name]
        lats = sorted(latencies_by_config[c_name])
        suff = sufficiency_by_config[c_name]

        agg_qa = qa_evaluator.evaluate_suite(qa_list)

        # Confusion Matrix
        tp = sum(1 for q in qa_questions if q.answerable and not per_q_outcomes[c_name][q.question_id].startswith("FP"))
        fp = sum(1 for q in qa_questions if q.answerable and per_q_outcomes[c_name][q.question_id] == "FP_refusal")
        tn = sum(1 for q in qa_questions if not q.answerable and per_q_outcomes[c_name][q.question_id] == "TN_refusal")
        fn = sum(1 for q in qa_questions if not q.answerable and per_q_outcomes[c_name][q.question_id] == "FN_hallucination")

        total_ans = sum(1 for q in qa_questions if q.answerable)
        total_unans = sum(1 for q in qa_questions if not q.answerable)
        total_q = len(qa_questions)

        decision_accuracy = (tp + tn) / total_q if total_q > 0 else 0.0
        false_refusal_rate = fp / total_ans if total_ans > 0 else 0.0
        refusal_recall = tn / total_unans if total_unans > 0 else 0.0

        # Answerable-only metrics
        ans_qa = [r for q, r in zip(qa_questions, qa_list) if q.answerable]
        ans_token_f1 = sum(r.token_f1 for r in ans_qa) / len(ans_qa) if ans_qa else 0.0
        ans_em = sum(r.exact_match for r in ans_qa) / len(ans_qa) if ans_qa else 0.0

        # Attempted-only metrics
        att_qa = [r for q, r in zip(qa_questions, qa_list) if q.answerable and not r.is_refusal]
        att_token_f1 = sum(r.token_f1 for r in att_qa) / len(att_qa) if att_qa else 0.0
        att_em = sum(r.exact_match for r in att_qa) / len(att_qa) if att_qa else 0.0

        mean_lat = sum(lats) / len(lats) if lats else 0.0
        median_lat = lats[len(lats) // 2] if lats else 0.0

        # Citation & validation metrics
        mean_citation_prec = agg_qa.citation_metrics.precision if agg_qa.citation_metrics else 0.0
        mean_grounding = agg_qa.validation_pass_rate

        experiment_summary["configurations"][c_name] = {
            "candidate_pool_size": pool_size,
            "max_evidence_items": 8,
            "qa_questions_count": total_q,
            "answerable_count": total_ans,
            "unanswerable_count": total_unans,
            "decision_accuracy": round(decision_accuracy, 4),
            "answerable_token_f1": round(ans_token_f1, 4),
            "answerable_exact_match": round(ans_em, 4),
            "overall_token_f1": round(agg_qa.mean_token_f1, 4),
            "overall_exact_match": round(agg_qa.mean_exact_match, 4),
            "attempted_token_f1": round(att_token_f1, 4),
            "attempted_exact_match": round(att_em, 4),
            "confusion_matrix": {
                "TP_answered": tp,
                "FP_false_refusal": fp,
                "TN_correct_refusal": tn,
                "FN_hallucination": fn,
            },
            "false_refusal_count": fp,
            "false_refusal_rate": round(false_refusal_rate, 4),
            "refusal_recall": round(refusal_recall, 4),
            "hallucination_count": fn,
            "citation_precision": round(mean_citation_prec, 4),
            "grounding_faithfulness": round(mean_grounding, 4),
            "latency_mean_s": round(mean_lat, 3),
            "latency_median_s": round(median_lat, 3),
            "sufficiency_gate_breakdown": suff,
        }

    # =========================================================================
    # Part 4: Save Deliverable Artifacts
    # =========================================================================
    summary_path = output_dir / "experiment_summary.json"
    with open(summary_path, "w") as f:
        json.dump(experiment_summary, f, indent=2)
    print(f"\n[Saved] Experiment summary: {summary_path}")

    csv_path = output_dir / "per_question_recall_results.csv"
    if per_question_rows:
        fieldnames = list(per_question_rows[0].keys())
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(per_question_rows)
        print(f"[Saved] Per-question CSV: {csv_path}")

    print("\n" + "=" * 80)
    print("RECALL EXPERIMENT COMPLETED SUCCESSFULLY")
    print("=" * 80)
    return experiment_summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Retrieval Recall Experiment across Candidate Pools")
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR))
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--full", action="store_true", help="Run QA on full 64 benchmark questions")
    args = parser.parse_args()

    run_experiment(
        output_dir=Path(args.output_dir),
        sample_size=args.sample_size,
        full_benchmark=args.full,
    )


if __name__ == "__main__":
    main()
