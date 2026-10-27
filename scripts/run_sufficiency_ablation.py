from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
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
from src.qa.agent import QAEvidencePackage
from src.qa.evidence_sufficiency import EvidenceSufficiencyChecker, SufficiencyResult
from src.qa.generator import QAAnswer
from src.retrieval.vector import VectorStore

ABLATION_RESULTS_DIR = ROOT / "results" / "phase5" / "sufficiency_ablation"


@dataclass
class GateConfig:
    config_id: str
    name: str
    description: str
    entity_mode: str
    relationship_mode: str
    hop_mode: str
    gate_enabled: bool


CONFIGURATIONS = [
    GateConfig(
        config_id="CFG0",
        name="Phase 5 Strict Gate (Baseline)",
        description="Strict entity (100%), strict relationship & compound chains, strict hop depth.",
        entity_mode="strict",
        relationship_mode="strict",
        hop_mode="strict",
        gate_enabled=True,
    ),
    GateConfig(
        config_id="CFG1",
        name="Soft Hop Gate",
        description="Strict entity (100%), strict relationship, soft hop depth (allows single-hop evidence).",
        entity_mode="strict",
        relationship_mode="strict",
        hop_mode="soft",
        gate_enabled=True,
    ),
    GateConfig(
        config_id="CFG2",
        name="Partial Entity + Soft Hop",
        description="Partial entity coverage (>=50%), strict relationship, soft hop depth.",
        entity_mode="partial",
        relationship_mode="strict",
        hop_mode="soft",
        gate_enabled=True,
    ),
    GateConfig(
        config_id="CFG3",
        name="Textual Relationship + Soft Hop",
        description="Strict entity (100%), textual relationship keyword matching, soft hop depth.",
        entity_mode="strict",
        relationship_mode="textual",
        hop_mode="soft",
        gate_enabled=True,
    ),
    GateConfig(
        config_id="CFG4",
        name="Balanced Gate (Partial Entity + Textual Rel + Soft Hop)",
        description="Partial entity coverage, textual relationship matching, soft hop depth.",
        entity_mode="partial",
        relationship_mode="textual",
        hop_mode="soft",
        gate_enabled=True,
    ),
    GateConfig(
        config_id="CFG5",
        name="Disabled Sufficiency Gate (No-Gate Ablation)",
        description="Sufficiency gate disabled; generates answer whenever retrieved evidence exists (distinct from B1 vector-only).",
        entity_mode="disabled",
        relationship_mode="disabled",
        hop_mode="disabled",
        gate_enabled=False,
    ),
]


def run_sufficiency_ablation_study(
    output_dir: Path | str = ABLATION_RESULTS_DIR,
    sample_size: int | None = None,
    configs: list[str] | None = None,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    benchmark_questions = load_benchmark(only_verified=True)
    if sample_size is not None and sample_size < len(benchmark_questions):
        benchmark_questions = benchmark_questions[:sample_size]

    config_map = {cfg.config_id: cfg for cfg in CONFIGURATIONS}
    if configs is not None:
        for c in configs:
            if c not in config_map:
                raise ValueError(
                    f"Unknown configuration ID: '{c}'. Valid IDs are: {list(config_map.keys())}"
                )
        selected_configs = [config_map[c] for c in configs]
    else:
        selected_configs = list(CONFIGURATIONS)

    print("=" * 80)
    print("FinGraph AI — Evidence-Sufficiency Threshold Ablation Study")
    print("=" * 80)
    print(f"Target Questions: {len(benchmark_questions)}")
    print(f"Configurations:   {len(selected_configs)} ({[c.config_id for c in selected_configs]})")
    print(f"Output Directory: {output_dir}")

    # 1. Initialize Pipeline & Evaluator
    vector_store = VectorStore(ROOT / "data" / "vector_store")
    pipeline = FinGraphPipeline(vector_store=vector_store)
    qa_evaluator = QAEvaluator()

    # 2. Stage 1: Retrieve and Cache Evidence for all questions once
    # This guarantees that RETRIEVAL IS 100% IDENTICAL across all 64 questions
    print("\n[1/3] Caching Multi-Hop Hybrid Retrieval Evidence across Questions...")
    cached_packages: dict[str, QAEvidencePackage] = {}
    t_ret0 = time.perf_counter()

    for idx, q in enumerate(benchmark_questions, 1):
        print(f"  [{idx}/{len(benchmark_questions)}] Retrieving evidence for {q.question_id}...")
        # Run agentic retrieval once to obtain point-in-time filtered evidence pool
        pkg = pipeline.agent.run(query=q.question, reference_time=q.reference_time)
        cached_packages[q.question_id] = pkg

    t_ret1 = time.perf_counter()
    print(f"Retrieval caching completed in {t_ret1 - t_ret0:.2f}s.")

    # 3. Stage 2: Evaluate Configurations and Generate Answers
    print("\n[2/3] Evaluating Configurations & Generating Conditioned Answers...")
    config_summaries: dict[str, Any] = {}
    csv_rows: list[dict[str, Any]] = []

    # Cache generated answers per question when sufficient:
    # Because evidence is constant, an answer generated for a question can be reused
    # across configurations that both deem the evidence sufficient!
    generated_answer_cache: dict[str, QAAnswer] = {}

    for cfg in selected_configs:
        print(f"\n--- Running {cfg.config_id}: {cfg.name} ---")
        checker = EvidenceSufficiencyChecker(
            graph_retriever=pipeline.graph_retriever,
            entity_mode=cfg.entity_mode,
            relationship_mode=cfg.relationship_mode,
            hop_mode=cfg.hop_mode,
            gate_enabled=cfg.gate_enabled,
        )

        cfg_qa_results: list[SingleQAResult] = []
        cfg_refusals = 0
        cfg_answers = 0

        for q_idx, q in enumerate(benchmark_questions, 1):
            pkg = cached_packages[q.question_id]

            # Re-evaluate sufficiency under this configuration's gate
            sufficiency = checker.check(pkg.analysis, pkg.all_evidence)

            # Build configuration-conditioned evidence package
            cfg_pkg = QAEvidencePackage(
                query=pkg.query,
                reference_time=pkg.reference_time,
                question_type=pkg.question_type,
                expected_hops=pkg.expected_hops,
                evidence=pkg.all_evidence,
                graph_paths=pkg.graph_paths,
                sources=pkg.sources,
                retrieval_rounds=pkg.retrieval_rounds,
                temporal_valid=pkg.temporal_valid,
                sufficient=sufficiency.sufficient,
                trace=pkg.trace,
                primary_evidence=pkg.primary_evidence,
                supporting_evidence=pkg.supporting_evidence,
                sufficiency_result=sufficiency,
                retrieval_trace=pkg.retrieval_trace,
                analysis=pkg.analysis,
            )

            # Generate or refuse
            if not sufficiency.sufficient or len(cfg_pkg.all_evidence) == 0:
                answer = QAAnswer(
                    answer="Insufficient evidence in the retrieved corpus.",
                    confidence=0.0,
                    evidence_ids=[],
                    citations=[],
                    temporal_reference=cfg_pkg.reference_time,
                    insufficient_evidence=True,
                    reasoning_summary=sufficiency.reason,
                    evidence=[ev.to_dict() for ev in cfg_pkg.all_evidence],
                    sources=cfg_pkg.sources,
                    validation_status={"is_valid": True, "errors": []},
                    evidence_package=cfg_pkg.to_dict(),
                )
                cfg_refusals += 1
                if q_idx % 10 == 0 or q_idx == len(benchmark_questions):
                    print(f"  [{q_idx}/{len(benchmark_questions)}] {q.question_id}: REFUSAL ({sufficiency.reason[:40]}...)")
            else:
                # Evidence is sufficient: generate with LLM (or retrieve from deterministic cache)
                if q.question_id in generated_answer_cache:
                    answer = generated_answer_cache[q.question_id]
                else:
                    t_gen0 = time.perf_counter()
                    answer = pipeline.generator.generate(cfg_pkg)
                    t_gen1 = time.perf_counter()
                    generated_answer_cache[q.question_id] = answer
                cfg_answers += 1
                if q_idx % 10 == 0 or q_idx == len(benchmark_questions):
                    print(f"  [{q_idx}/{len(benchmark_questions)}] {q.question_id}: ANSWERED")

            # Evaluate QA
            is_valid = answer.validation_status.get("is_valid", False) if answer.validation_status else False
            qa_res = qa_evaluator.evaluate_single(
                question_id=q.question_id,
                predicted_answer=answer.answer,
                ground_truth_answer=q.ground_truth_answer,
                is_answerable=q.answerable,
                confidence=answer.confidence,
                evidence_ids_cited=answer.evidence_ids,
                citations=answer.citations,
                validation_passed=is_valid,
            )
            cfg_qa_results.append(qa_res)

            # Record outcome classification
            outcome = (
                "TP_correct_refusal" if (not q.answerable and qa_res.is_refusal)
                else "FN_hallucination" if (not q.answerable and not qa_res.is_refusal)
                else "FP_false_refusal" if (q.answerable and qa_res.is_refusal)
                else "TN_correct_answer"
            )

            csv_rows.append({
                "config_id": cfg.config_id,
                "config_name": cfg.name,
                "question_id": q.question_id,
                "category": q.category,
                "hop_level": q.hop_level,
                "answerable": q.answerable,
                "is_refusal": qa_res.is_refusal,
                "refusal_outcome": outcome,
                "exact_match": qa_res.exact_match,
                "token_f1": qa_res.token_f1,
                "confidence": answer.confidence,
                "validation_passed": is_valid,
                "sufficiency_reason": sufficiency.reason,
            })

        # Calculate confusion matrix
        total_q = len(benchmark_questions)
        ans_results = [r for r, q in zip(cfg_qa_results, benchmark_questions) if q.answerable]
        unans_results = [r for r, q in zip(cfg_qa_results, benchmark_questions) if not q.answerable]

        tp = sum(1 for r in unans_results if r.is_refusal)
        fn = sum(1 for r in unans_results if not r.is_refusal)
        fp = sum(1 for r in ans_results if r.is_refusal)
        tn = sum(1 for r in ans_results if not r.is_refusal)

        precision = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        recall = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        accuracy = (tp + tn) / total_q if total_q > 0 else 0.0

        mean_f1_all = sum(r.token_f1 for r in cfg_qa_results) / total_q if total_q > 0 else 0.0
        mean_f1_ans = sum(r.token_f1 for r in ans_results) / len(ans_results) if ans_results else 0.0
        mean_f1_unans = sum(r.token_f1 for r in unans_results) / len(unans_results) if unans_results else 0.0
        mean_em = sum(r.exact_match for r in cfg_qa_results) / total_q if total_q > 0 else 0.0

        ans_coverage = (tn / len(ans_results)) if ans_results else 0.0
        hallucination_rate = (fn / len(unans_results)) if unans_results else 0.0

        print(f"  TP: {tp} | FN: {fn} | FP: {fp} | TN: {tn}")
        print(f"  Refusal Recall: {recall*100:.2f}% | Precision: {precision*100:.2f}% | Decision Accuracy: {accuracy*100:.2f}%")
        print(f"  Answer Coverage (Answerable): {ans_coverage*100:.2f}% | Hallucination Rate: {hallucination_rate*100:.2f}%")
        print(f"  Mean Token F1 (All): {mean_f1_all:.4f} | (Ans): {mean_f1_ans:.4f} | Mean EM: {mean_em:.4f}")

        config_summaries[cfg.config_id] = {
            "config": asdict(cfg),
            "confusion_matrix": {
                "TP": tp,
                "FN": fn,
                "FP": fp,
                "TN": tn,
                "total_questions": total_q,
                "total_answerable": len(ans_results),
                "total_unanswerable": len(unans_results),
            },
            "classification_metrics": {
                "refusal_precision": round(precision, 4),
                "refusal_recall": round(recall, 4),
                "refusal_f1": round(f1, 4),
                "decision_accuracy": round(accuracy, 4),
                "hallucination_rate_on_unanswerable": round(hallucination_rate, 4),
                "false_refusal_rate_on_answerable": round(fp / len(ans_results), 4) if ans_results else 0.0,
                "answer_coverage_on_answerable": round(ans_coverage, 4),
            },
            "generation_metrics": {
                "mean_token_f1_all": round(mean_f1_all, 4),
                "mean_token_f1_answerable": round(mean_f1_ans, 4),
                "mean_token_f1_unanswerable": round(mean_f1_unans, 4),
                "mean_exact_match": round(mean_em, 4),
                "validation_pass_rate": round(sum(1 for r in cfg_qa_results if r.validation_passed) / total_q, 4) if total_q > 0 else 0.0,
            },
        }

    # 4. Stage 3: Save Ablation Artifacts
    print("\n[3/3] Saving Ablation Experiment Artifacts...")
    summary_path = output_dir / "ablation_summary.json"
    full_report = {
        "run_id": f"ablation_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_benchmark_questions": len(benchmark_questions),
        "selected_configurations": [cfg.config_id for cfg in selected_configs],
        "retrieval_mode": "fresh_retrieval_in_memory_cached",
        "reference_time_policy": "strict_point_in_time",
        "configurations": config_summaries,
    }
    with open(summary_path, "w") as f:
        json.dump(full_report, f, indent=2)

    csv_path = output_dir / "per_question_ablation_results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(csv_rows)

    print(f"Artifacts successfully written:")
    print(f"  Summary JSON: {summary_path}")
    print(f"  Details CSV:  {csv_path}")

    return full_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run FinGraph AI Evidence Sufficiency Ablation")
    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        default=ABLATION_RESULTS_DIR,
        help="Output directory for ablation artifacts",
    )
    parser.add_argument(
        "--configs",
        "-c",
        nargs="+",
        default=None,
        help="Specific configuration IDs to run (e.g. CFG0 CFG2)",
    )
    parser.add_argument("--sample", "-s", type=int, default=None, help="Sample size for testing")
    args = parser.parse_args()

    run_sufficiency_ablation_study(
        output_dir=args.output_dir,
        sample_size=args.sample,
        configs=args.configs,
    )


if __name__ == "__main__":
    main()
