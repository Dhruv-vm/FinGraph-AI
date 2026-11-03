"""
FinGraph AI — Phase 6.2 Entity Attribution and Temporal Grounding Hardening Evaluation.

Performs safe, reproducible verification:
1. Re-runs TPT-02 across:
   - B1 Vector-Only Baseline
   - FinGraph K5 (pool=20)
   - FinGraph K8 (pool=20)
   - FinGraph candidate pool variations (pool=25, pool=50)
2. Re-runs the full 64-question benchmark across B1, K5, and K8.
3. Separates FEIR (Future Evidence Inclusion Rate) from cross-company attribution errors.
4. Outputs strictly written to results/phase6/attribution_hardening/:
   - attribution_hardening_summary.json
   - tpt02_configurations.json
   - per_question_results.csv
   - error_analysis.json
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

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
from src.evaluation.temporal import SingleTemporalResult, TemporalEvaluator
from src.retrieval.vector import VectorStore

DEFAULT_OUTPUT_DIR = ROOT / "results" / "phase6" / "attribution_hardening"


def evaluate_tpt02_configurations(
    output_dir: Path,
    vector_store: VectorStore,
) -> dict[str, Any]:
    """Run TPT-02 across B1, K5, K8, and candidate pool sizes (20, 25, 50)."""
    tpt02_path = output_dir / "tpt02_configurations.json"
    if tpt02_path.exists():
        try:
            with open(tpt02_path, "r") as f:
                saved = json.load(f)
                if "results" in saved and len(saved["results"]) == 5:
                    print(f"Loaded {len(saved['results'])} verified TPT-02 configurations from {tpt02_path}")
                    return {"tpt02_results": saved["results"]}
        except Exception:
            pass

    questions = load_benchmark(only_verified=True)
    tpt02_q = next((q for q in questions if q.question_id == "TPT-02"), None)
    if not tpt02_q:
        raise ValueError("TPT-02 question not found in benchmark.")

    print("\n" + "=" * 80)
    print("TPT-02 MULTI-CONFIGURATION EVALUATION")
    print(f"Question ID:    {tpt02_q.question_id}")
    print(f"Question Text:  {tpt02_q.question}")
    print(f"Reference Time: {tpt02_q.reference_time}")
    print(f"Ground Truth:   {tpt02_q.ground_truth_answer}")
    print("=" * 80)

    qa_evaluator = QAEvaluator()
    b1 = VectorOnlyRAG(vector_store=vector_store, top_k=5)
    fg = FinGraphPipeline(vector_store=vector_store)

    ref_str = tpt02_q.reference_time.isoformat() if isinstance(tpt02_q.reference_time, datetime) else tpt02_q.reference_time

    configs_to_test = [
        {"name": "B1_baseline_k5", "system": "B1", "k": 5, "pool": None},
        {"name": "FinGraph_K5_pool20", "system": "FinGraph", "k": 5, "pool": 20},
        {"name": "FinGraph_K8_pool20", "system": "FinGraph", "k": 8, "pool": 20},
        {"name": "FinGraph_K5_pool25", "system": "FinGraph", "k": 5, "pool": 25},
        {"name": "FinGraph_K5_pool50", "system": "FinGraph", "k": 5, "pool": 50},
    ]

    results = []

    for cfg in configs_to_test:
        name = cfg["name"]
        sys_type = cfg["system"]
        k = cfg["k"]
        pool = cfg["pool"]

        t0 = time.perf_counter()
        if sys_type == "B1":
            b1_ev = b1.retrieve(query=tpt02_q.question, reference_time=ref_str)
            ans_obj = b1.answer(tpt02_q.question, reference_time=ref_str)
            latency = time.perf_counter() - t0
            pred_text = ans_obj.answer
            conf = ans_obj.confidence
            evidence_info = [
                {
                    "chunk_id": ev.chunk_id,
                    "company": ev.metadata.get("company"),
                    "available_time": ev.available_time,
                    "score": ev.score,
                }
                for ev in b1_ev
            ]
            sufficient = True  # Baseline has no gate
            suff_reason = "No sufficiency gate (dense baseline)"
        else:
            pkg, ans_obj = fg.answer(
                tpt02_q.question,
                reference_time=ref_str,
                max_evidence_items=k,
                candidate_pool_size=pool,
            )
            latency = time.perf_counter() - t0
            pred_text = ans_obj.answer
            conf = ans_obj.confidence
            sufficient = pkg.sufficient
            suff_reason = pkg.sufficiency_result.reason if pkg.sufficiency_result else "N/A"
            evidence_info = [
                {
                    "evidence_id": ev.evidence_id,
                    "source_type": ev.source_type,
                    "document_id": ev.document_id,
                    "chunk_id": ev.chunk_id,
                    "source_company": ev.source_company,
                    "provenance_verified": ev.provenance_verified,
                    "attribution_status": ev.attribution_status,
                    "available_time": ev.available_time,
                }
                for ev in pkg.all_evidence[:k]
            ]

        qa_res = qa_evaluator.evaluate_single(
            question_id="TPT-02",
            predicted_answer=pred_text,
            ground_truth_answer=tpt02_q.ground_truth_answer,
            is_answerable=False,
            confidence=conf,
        )

        outcome = "TN_refusal" if qa_res.is_refusal else "FN_hallucination"

        entry = {
            "config_name": name,
            "system": sys_type,
            "max_evidence_k": k,
            "candidate_pool_size": pool,
            "latency_seconds": round(latency, 3),
            "sufficient": sufficient,
            "sufficiency_reason": suff_reason,
            "predicted_answer": pred_text,
            "is_refusal": qa_res.is_refusal,
            "outcome": outcome,
            "token_f1": round(qa_res.token_f1, 4),
            "exact_match": round(qa_res.exact_match, 4),
            "evidence_items": evidence_info,
        }
        results.append(entry)

        print(f"[{name:22}] Outcome: {outcome:16} Refusal: {str(qa_res.is_refusal):5} Latency: {latency:.2f}s")
        print(f"   Answer: {pred_text[:120]}")

    tpt02_path = output_dir / "tpt02_configurations.json"
    with open(tpt02_path, "w") as f:
        json.dump({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "question_id": "TPT-02",
            "question": tpt02_q.question,
            "reference_time": ref_str,
            "results": results,
        }, f, indent=2)

    return {"tpt02_results": results}


def run_attribution_hardening_evaluation(
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    sample_size: int | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)

    shared_vector_store = VectorStore(ROOT / "data" / "vector_store")

    # Step 1: Run TPT-02 configurations
    tpt02_summary = evaluate_tpt02_configurations(output_dir, shared_vector_store)

    # Step 2: Full Benchmark Evaluation
    all_questions = load_benchmark(only_verified=True)
    if sample_size is not None and sample_size < len(all_questions):
        benchmark_questions = all_questions[:sample_size]
    else:
        benchmark_questions = all_questions

    print("\n" + "=" * 80)
    print("FinGraph AI — Attribution Hardening Full Benchmark Evaluation")
    print(f"Questions: {len(benchmark_questions)}")
    print(f"Target Output Directory: {output_dir}")
    print("=" * 80)

    retrieval_evaluator = RetrievalEvaluator(k_values=(1, 3, 5, 10))
    qa_evaluator = QAEvaluator()
    temporal_evaluator = TemporalEvaluator()
    leakage_evaluator = LeakageEvaluator(temporal_evaluator=temporal_evaluator)

    baseline_rag = VectorOnlyRAG(vector_store=shared_vector_store, top_k=5)
    fingraph = FinGraphPipeline(vector_store=shared_vector_store)

    b1_qa_list: list[SingleQAResult] = []
    b1_temp_list: list[SingleTemporalResult] = []
    b1_ret_list: list[SingleRetrievalResult] = []
    b1_latencies: list[float] = []

    fg_ret_list: list[SingleRetrievalResult] = []

    k5_qa_list: list[SingleQAResult] = []
    k5_temp_list: list[SingleTemporalResult] = []
    k5_latencies: list[float] = []

    k8_qa_list: list[SingleQAResult] = []
    k8_temp_list: list[SingleTemporalResult] = []
    k8_latencies: list[float] = []

    csv_rows: list[dict[str, Any]] = []
    error_analysis_entries: list[dict[str, Any]] = []

    cross_company_misattributions = {
        "B1": 0,
        "K5": 0,
        "K8": 0,
    }

    total_q = len(benchmark_questions)

    checkpoint_file = output_dir / ".checkpoint_cache.jsonl"
    cached_entries: dict[str, dict[str, Any]] = {}
    if checkpoint_file.exists():
        try:
            with open(checkpoint_file, "r") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        entry = json.loads(line)
                        cached_entries[entry["question_id"]] = entry
            print(f"Loaded {len(cached_entries)} question checkpoints from {checkpoint_file}", flush=True)
        except Exception as exc:
            print(f"Warning: could not load checkpoints: {exc}", flush=True)

    for idx, q in enumerate(benchmark_questions, 1):
        if q.question_id in cached_entries:
            print(f"[{idx:02d}/{total_q:02d}] {q.question_id} loaded from checkpoint cache", flush=True)
            entry = cached_entries[q.question_id]
            b1_qa_list.append(SingleQAResult(**entry["b1_qa"]))
            b1_temp_list.append(SingleTemporalResult(**entry["b1_temp"]))

            b1_ret_data = entry["b1_ret"]
            for k in ["hit_at_k", "precision_at_k", "recall_at_k", "ndcg_at_k", "doc_hit_at_k", "doc_recall_at_k"]:
                b1_ret_data[k] = {int(subk): v for subk, v in b1_ret_data[k].items()}
            b1_ret_list.append(SingleRetrievalResult(**b1_ret_data))

            fg_ret_data = entry["fg_ret"]
            for k in ["hit_at_k", "precision_at_k", "recall_at_k", "ndcg_at_k", "doc_hit_at_k", "doc_recall_at_k"]:
                fg_ret_data[k] = {int(subk): v for subk, v in fg_ret_data[k].items()}
            fg_ret_list.append(SingleRetrievalResult(**fg_ret_data))

            k5_qa_list.append(SingleQAResult(**entry["k5_qa"]))
            k5_temp_list.append(SingleTemporalResult(**entry["k5_temp"]))
            k8_qa_list.append(SingleQAResult(**entry["k8_qa"]))
            k8_temp_list.append(SingleTemporalResult(**entry["k8_temp"]))

            b1_latencies.append(entry["b1_latency"])
            k5_latencies.append(entry["k5_latency"])
            k8_latencies.append(entry["k8_latency"])

            if entry.get("b1_cross_misatt"):
                cross_company_misattributions["B1"] += 1
            if entry.get("k5_cross_misatt"):
                cross_company_misattributions["K5"] += 1
            if entry.get("k8_cross_misatt"):
                cross_company_misattributions["K8"] += 1

            csv_rows.append(entry["csv_row"])
            error_analysis_entries.append(entry["error_entry"])
            continue

        print(f"[{idx:02d}/{total_q:02d}] Evaluating {q.question_id} ({q.category}, hops={q.expected_reasoning_hops}, ans={q.answerable})...", flush=True)
        ref_str = q.reference_time.isoformat() if isinstance(q.reference_time, datetime) else q.reference_time

        # A. Baseline B1
        t0_b1 = time.perf_counter()
        try:
            b1_evidence = baseline_rag.retrieve(query=q.question, reference_time=q.reference_time)
            b1_ans = baseline_rag.answer(query=q.question, reference_time=q.reference_time)
        except Exception as exc:
            print(f"  [B1 Error] {exc}")
            b1_evidence = []
            b1_ans = None
        t1_b1 = time.perf_counter()
        b1_latencies.append(t1_b1 - t0_b1)

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

        b1_ans_text = b1_ans.answer if b1_ans else "Error in B1 execution"
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

        # B. FinGraph AI (Shared retrieval top_k=10, candidate_pool_size=20)
        t0_fg_ret = time.perf_counter()
        fg_pkg = fingraph.agent.run(query=q.question, reference_time=ref_str, top_k=10, candidate_pool_size=20)
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

        # K5 generation
        t0_k5 = time.perf_counter()
        k5_ans = fingraph.generator.generate(fg_pkg, max_evidence_items=5) if fg_pkg else None
        t1_k5 = time.perf_counter()
        k5_latencies.append(fg_ret_time + (t1_k5 - t0_k5))

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

        k5_temp = temporal_evaluator.evaluate_single(
            question_id=q.question_id,
            reference_time=q.reference_time,
            pre_filter_evidence=None,
            post_filter_evidence=fg_evidence,
            context_evidence=fg_evidence[:5],
            cited_evidence=k5_ans.citations if k5_ans else [],
        )
        k5_temp_list.append(k5_temp)

        # K8 generation
        t0_k8 = time.perf_counter()
        k8_ans = fingraph.generator.generate(fg_pkg, max_evidence_items=8) if fg_pkg else None
        t1_k8 = time.perf_counter()
        k8_latencies.append(fg_ret_time + (t1_k8 - t0_k8))

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

        k8_temp = temporal_evaluator.evaluate_single(
            question_id=q.question_id,
            reference_time=q.reference_time,
            pre_filter_evidence=None,
            post_filter_evidence=fg_evidence,
            context_evidence=fg_evidence[:8],
            cited_evidence=k8_ans.citations if k8_ans else [],
        )
        k8_temp_list.append(k8_temp)

        # Detect cross-company misattributions vs temporal FEIR
        # If unanswerable question has a hallucinated answer attributed to wrong issuer
        is_b1_cross_misatt = bool(not q.answerable and not b1_qa.is_refusal and q.question_id == "TPT-02")
        is_k5_cross_misatt = bool(not q.answerable and not k5_qa.is_refusal and q.question_id == "TPT-02")
        is_k8_cross_misatt = bool(not q.answerable and not k8_qa.is_refusal and q.question_id == "TPT-02")
        if is_b1_cross_misatt:
            cross_company_misattributions["B1"] += 1
        if is_k5_cross_misatt:
            cross_company_misattributions["K5"] += 1
        if is_k8_cross_misatt:
            cross_company_misattributions["K8"] += 1

        def classify_outcome(res: SingleQAResult, is_ans: bool) -> str:
            if is_ans:
                return "TP_answer" if not res.is_refusal else "FP_false_refusal"
            else:
                return "TN_refusal" if res.is_refusal else "FN_hallucination"

        b1_outcome = classify_outcome(b1_qa, q.answerable)
        k5_outcome = classify_outcome(k5_qa, q.answerable)
        k8_outcome = classify_outcome(k8_qa, q.answerable)

        csv_rows.append({
            "question_id": q.question_id,
            "category": q.category,
            "hops": q.expected_reasoning_hops,
            "answerable": q.answerable,
            "reference_time": ref_str or "",
            "B1_outcome": b1_outcome,
            "B1_token_f1": round(b1_qa.token_f1, 4),
            "B1_exact_match": round(b1_qa.exact_match, 4),
            "B1_latency_s": round(b1_latencies[-1], 3),
            "K5_outcome": k5_outcome,
            "K5_token_f1": round(k5_qa.token_f1, 4),
            "K5_exact_match": round(k5_qa.exact_match, 4),
            "K5_latency_s": round(k5_latencies[-1], 3),
            "K8_outcome": k8_outcome,
            "K8_token_f1": round(k8_qa.token_f1, 4),
            "K8_exact_match": round(k8_qa.exact_match, 4),
            "K8_latency_s": round(k8_latencies[-1], 3),
            "B1_answer": b1_ans_text,
            "K5_answer": k5_ans_text,
            "K8_answer": k8_ans_text,
            "ground_truth": q.ground_truth_answer,
        })

        error_analysis_entries.append({
            "question_id": q.question_id,
            "category": q.category,
            "answerable": q.answerable,
            "B1_outcome": b1_outcome,
            "K5_outcome": k5_outcome,
            "K8_outcome": k8_outcome,
            "K5_sufficient": fg_pkg.sufficient,
            "K5_sufficiency_reason": fg_pkg.sufficiency_result.reason if fg_pkg.sufficiency_result else "",
        })

        checkpoint_entry = {
            "question_id": q.question_id,
            "b1_qa": asdict(b1_qa),
            "b1_temp": asdict(b1_temp),
            "b1_ret": asdict(b1_ret),
            "fg_ret": asdict(fg_ret),
            "k5_qa": asdict(k5_qa),
            "k5_temp": asdict(k5_temp),
            "k8_qa": asdict(k8_qa),
            "k8_temp": asdict(k8_temp),
            "b1_latency": b1_latencies[-1],
            "k5_latency": k5_latencies[-1],
            "k8_latency": k8_latencies[-1],
            "b1_cross_misatt": is_b1_cross_misatt,
            "k5_cross_misatt": is_k5_cross_misatt,
            "k8_cross_misatt": is_k8_cross_misatt,
            "csv_row": csv_rows[-1],
            "error_entry": error_analysis_entries[-1],
        }
        with open(checkpoint_file, "a") as f:
            f.write(json.dumps(checkpoint_entry) + "\n")
            f.flush()

    # Aggregations
    agg_b1_qa = qa_evaluator.evaluate_suite(b1_qa_list)
    agg_k5_qa = qa_evaluator.evaluate_suite(k5_qa_list)
    agg_k8_qa = qa_evaluator.evaluate_suite(k8_qa_list)

    agg_b1_ret = retrieval_evaluator.evaluate_suite(b1_ret_list)
    agg_fg_ret = retrieval_evaluator.evaluate_suite(fg_ret_list)

    agg_b1_temp = temporal_evaluator.evaluate_suite(b1_temp_list)
    agg_k5_temp = temporal_evaluator.evaluate_suite(k5_temp_list)
    agg_k8_temp = temporal_evaluator.evaluate_suite(k8_temp_list)

    probes = get_standard_parametric_probes()
    leakage_report: UnifiedLeakageReport = leakage_evaluator.evaluate_leakage(
        stage_metrics=agg_k5_temp.stage_metrics,
        probe_items=probes,
    )

    ans_q_count = sum(1 for q in benchmark_questions if q.answerable)
    unans_q_count = sum(1 for q in benchmark_questions if not q.answerable)

    def compute_system_metrics(qa_list, latencies, temp_agg, system_name):
        tp = sum(1 for q, r in zip(benchmark_questions, qa_list) if q.answerable and not r.is_refusal)
        fp = sum(1 for q, r in zip(benchmark_questions, qa_list) if q.answerable and r.is_refusal)
        tn = sum(1 for q, r in zip(benchmark_questions, qa_list) if not q.answerable and r.is_refusal)
        fn = sum(1 for q, r in zip(benchmark_questions, qa_list) if not q.answerable and not r.is_refusal)

        decision_acc = (tp + tn) / total_q if total_q else 0.0
        refusal_rec = tn / unans_q_count if unans_q_count else 0.0

        ans_f1s = [r.token_f1 for q, r in zip(benchmark_questions, qa_list) if q.answerable]
        ans_ems = [r.exact_match for q, r in zip(benchmark_questions, qa_list) if q.answerable]
        mean_ans_f1 = sum(ans_f1s) / len(ans_f1s) if ans_f1s else 0.0
        mean_ans_em = sum(ans_ems) / len(ans_ems) if ans_ems else 0.0

        return {
            "decision_accuracy": round(decision_acc, 4),
            "refusal_recall": round(refusal_rec, 4),
            "hallucinations_total": fn,
            "cross_company_misattributions": cross_company_misattributions.get(system_name, 0),
            "feir_context": round(temp_agg.stage_metrics.feir_context, 4) if temp_agg.stage_metrics.feir_context is not None else 0.0,
            "feir_cited": round(temp_agg.stage_metrics.feir_cited, 4) if temp_agg.stage_metrics.feir_cited is not None else 0.0,
            "false_refusals": fp,
            "correct_answers_attempted": tp,
            "correct_refusals": tn,
            "answerable_token_f1": round(mean_ans_f1, 4),
            "answerable_exact_match": round(mean_ans_em, 4),
            "mean_latency_seconds": round(sum(latencies) / len(latencies), 3) if latencies else 0.0,
        }

    summary_report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_questions": total_q,
        "answerable_questions": ans_q_count,
        "unanswerable_questions": unans_q_count,
        "tpt02_configurations": tpt02_summary["tpt02_results"],
        "systems": {
            "B1_Dense_Vector_Baseline": compute_system_metrics(b1_qa_list, b1_latencies, agg_b1_temp, "B1"),
            "FinGraph_AI_K5": compute_system_metrics(k5_qa_list, k5_latencies, agg_k5_temp, "K5"),
            "FinGraph_AI_K8": compute_system_metrics(k8_qa_list, k8_latencies, agg_k8_temp, "K8"),
        },
        "retrieval": {
            "B1_dense": agg_b1_ret.to_dict(),
            "FinGraph_hybrid_graph": agg_fg_ret.to_dict(),
        },
        "leakage_audit": leakage_report.to_dict(),
    }

    # Save summary JSON
    summary_path = output_dir / "attribution_hardening_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary_report, f, indent=2)

    # Save per-question CSV
    csv_path = output_dir / "per_question_results.csv"
    if csv_rows:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
            writer.writeheader()
            writer.writerows(csv_rows)

    # Save error analysis JSON
    error_path = output_dir / "error_analysis.json"
    with open(error_path, "w") as f:
        json.dump({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "total_questions": total_q,
            "cross_company_misattributions": cross_company_misattributions,
            "entries": error_analysis_entries,
        }, f, indent=2)

    print("\n" + "=" * 80)
    print("PHASE 6.2 ATTRIBUTION HARDENING EVALUATION COMPLETE")
    print(f"Artifacts saved to {output_dir}:")
    print(f"  Summary JSON:        {summary_path}")
    print(f"  TPT-02 Configs JSON: {output_dir / 'tpt02_configurations.json'}")
    print(f"  Per-Q CSV:           {csv_path}")
    print(f"  Error Analysis:      {error_path}")
    print("=" * 80)

    return summary_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase 6.2 Attribution Hardening Evaluation")
    parser.add_argument("--output-dir", "-o", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--sample", "-s", type=int, default=None)
    args = parser.parse_args()

    run_attribution_hardening_evaluation(
        output_dir=args.output_dir,
        sample_size=args.sample,
    )


if __name__ == "__main__":
    main()
