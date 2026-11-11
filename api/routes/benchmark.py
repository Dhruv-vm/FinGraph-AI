"""
Benchmark demonstration endpoints for FinGraph AI API.
"""

from __future__ import annotations

from functools import lru_cache
from fastapi import APIRouter

from api.schemas.benchmark import (
    CachedBenchmarkResult,
    DemoQuestionItem,
    DemoQuestionsResponse,
)

router = APIRouter(prefix="/benchmark", tags=["Benchmark"])

_VERIFIED_DEMO_QUESTIONS: list[DemoQuestionItem] = [
    DemoQuestionItem(
        question_id="FSH-07",
        question="What major video gaming and entertainment company did Microsoft acquire?",
        category="factual_single_hop",
        answerable=True,
        expected_hops=1,
        reference_time=None,
        entities=["company:MSFT"],
        ground_truth_answer="Microsoft acquired Activision Blizzard.",
        unanswerability_rationale=None,
        cached_phase7_result=CachedBenchmarkResult(
            decision="answer",
            token_f1=1.0,
            generated_answer="Microsoft acquired Activision Blizzard.",
            citations=[
                "hybrid:doc:sec:6a50d47d6a8dc43d:chunk:160",
                "hybrid:doc:sec:076412151b515b96:chunk:62",
                "graph:edge:company:MSFT->ACQUIRED->company:33188a412bf30ded:2f753e8ce5c9099b",
                "graph:edge:company:MSFT->ACQUIRED->company:33188a412bf30ded:bbf4bfa2f9963a71",
                "graph:edge:company:MSFT->ACQUIRED->company:33188a412bf30ded:0b4300bafd741e22",
                "hybrid:doc:sec:e312824081333ffd:chunk:46",
                "hybrid:doc:sec:6a50d47d6a8dc43d:chunk:20",
                "hybrid:doc:sec:e312824081333ffd:chunk:53",
            ],
            latency_sec=20.17,
            outcome_status="TN_correct_decision_to_answer",
        ),
    ),
    DemoQuestionItem(
        question_id="CDC-04",
        question="How do NVIDIA and Microsoft disclose their respective relationships with AI cloud computing infrastructure?",
        category="cross_document_comparative",
        answerable=True,
        expected_hops=2,
        reference_time=None,
        entities=["company:NVDA", "company:MSFT"],
        ground_truth_answer="NVIDIA supplies GPU architectures and enters AI cloud agreements, while Microsoft deploys AI cloud infrastructure partnering with OpenAI and hosting enterprise models on Azure.",
        unanswerability_rationale=None,
        cached_phase7_result=CachedBenchmarkResult(
            decision="answer",
            token_f1=0.3279,
            generated_answer="NVIDIA discloses its relationship with AI cloud computing infrastructure by stating that AI clouds procure its data center infrastructure products and commit to cloud services, while Microsoft is described as delivering services including generative AI content creation.",
            citations=[
                "hybrid:doc:sec:6a50d47d6a8dc43d:chunk:10",
                "hybrid:doc:sec:98d8cee8a8ae0f7d:chunk:8",
                "hybrid:doc:sec:98d8cee8a8ae0f7d:chunk:9",
                "hybrid:doc:sec:6aadaca6341c317e:chunk:41",
                "vector:doc:sec:98d8cee8a8ae0f7d:chunk:12",
                "vector:doc:sec:98d8cee8a8ae0f7d:chunk:14",
                "vector:doc:sec:1ad2e96011122307:chunk:26",
                "vector:doc:sec:98d8cee8a8ae0f7d:chunk:119",
            ],
            latency_sec=20.09,
            outcome_status="TN_correct_decision_to_answer",
        ),
    ),
    DemoQuestionItem(
        question_id="RMH-07",
        question="What gaming company acquired by Microsoft had its transaction challenged by competition regulators?",
        category="relational_multi_hop",
        answerable=True,
        expected_hops=2,
        reference_time=None,
        entities=["company:MSFT"],
        ground_truth_answer="Microsoft acquired Activision Blizzard, an acquisition subject to scrutiny and agreements regarding cross-platform availability.",
        unanswerability_rationale=None,
        cached_phase7_result=CachedBenchmarkResult(
            decision="answer",
            token_f1=0.2759,
            generated_answer="Activision Blizzard is the gaming company acquired by Microsoft that had its transaction challenged by competition regulators.",
            citations=[
                "hybrid:doc:sec:6a50d47d6a8dc43d:chunk:160",
                "graph:edge:company:MSFT->ACQUIRED->company:33188a412bf30ded:2f753e8ce5c9099b",
            ],
            latency_sec=2.64,
            outcome_status="TN_correct_decision_to_answer",
        ),
    ),
    DemoQuestionItem(
        question_id="TPT-01",
        question="What financial or cloud service revenue did Microsoft disclose as of May 1, 2025?",
        category="temporal_point_in_time",
        answerable=True,
        expected_hops=1,
        reference_time="2025-05-01T00:00:00+00:00",
        entities=["company:MSFT"],
        ground_truth_answer="As of May 1, 2025, Microsoft disclosed financial metrics including Microsoft Cloud revenue in its April 2025 Form 10-Q filing.",
        unanswerability_rationale=None,
        cached_phase7_result=CachedBenchmarkResult(
            decision="answer",
            token_f1=0.2308,
            generated_answer="Microsoft disclosed that Microsoft 365 Consumer products and cloud services revenue increased $173 million or 10%, while Intelligent Cloud Revenue increased $12.7 billion or 20% for the three months ended March 31, 2025.",
            citations=[
                "hybrid:doc:sec:076412151b515b96:chunk:52",
                "hybrid:doc:sec:076412151b515b96:chunk:59",
                "hybrid:doc:sec:076412151b515b96:chunk:58",
                "hybrid:doc:sec:076412151b515b96:chunk:61",
                "hybrid:doc:sec:076412151b515b96:chunk:62",
                "hybrid:doc:sec:076412151b515b96:chunk:49",
                "hybrid:doc:sec:076412151b515b96:chunk:26",
                "hybrid:doc:sec:076412151b515b96:chunk:55",
            ],
            latency_sec=24.11,
            outcome_status="TN_correct_decision_to_answer",
        ),
    ),
    DemoQuestionItem(
        question_id="TPT-02",
        question="What risks affected NVIDIA as of June 1, 2025?",
        category="temporal_point_in_time",
        answerable=False,
        expected_hops=1,
        reference_time="2025-06-01T00:00:00+00:00",
        entities=["company:NVDA"],
        ground_truth_answer="Insufficient evidence in the retrieved corpus.",
        unanswerability_rationale="The earliest NVIDIA SEC filing in the 99-filing corpus is dated August 27, 2025. As of June 1, 2025, no NVIDIA filings exist in the corpus, requiring refusal.",
        cached_phase7_result=CachedBenchmarkResult(
            decision="refuse",
            token_f1=1.0,
            generated_answer="Insufficient evidence in the retrieved corpus.",
            citations=[],
            latency_sec=0.47,
            outcome_status="TP_correct_refusal",
        ),
    ),
]


@router.get(
    "/demo-questions",
    response_model=DemoQuestionsResponse,
    summary="Get 5 audited demonstration questions with cached Phase 7 results",
    description=(
        "Returns five representative questions verified during Phase 7 (factual, comparative, "
        "regulatory multi-hop, temporal point-in-time, and temporal refusal). "
        "Provides ground truth, rationale, and cached evaluation metrics, allowing the UI "
        "to display baseline results or run live queries."
    ),
)
async def get_demo_questions() -> DemoQuestionsResponse:
    return DemoQuestionsResponse(
        total_count=len(_VERIFIED_DEMO_QUESTIONS),
        questions=_VERIFIED_DEMO_QUESTIONS,
    )


@router.get(
    "/summary",
    summary="Get benchmark ablation evaluation summaries across all experimental configurations",
)
@lru_cache(maxsize=1)
def get_benchmark_summary():
    import json
    from pathlib import Path
    root = Path(__file__).resolve().parents[2]
    summary_path = root / "results" / "phase7" / "phase7_summary.json"
    if not summary_path.exists():
        return {"error": "Phase 7 summary not found", "ablation_comparison": []}
    with open(summary_path, mode="r", encoding="utf-8") as f:
        return json.load(f)



@router.get(
    "/per-question",
    summary="Get historical Phase 7 per-question outcomes (read-only)",
    description=(
        "Parses results/phase7/per_question_results.csv without modification. "
        "These are historical benchmark records, not live pipeline runs."
    ),
)
@lru_cache(maxsize=1)
def get_per_question_results():
    import csv
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    csv_path = root / "results" / "phase7" / "per_question_results.csv"
    if not csv_path.exists():
        return {"error": "Phase 7 per-question results not found", "questions": []}

    def _float(value):
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    rows = []
    with open(csv_path, mode="r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            answerable = str(row.get("answerable", "")).strip().lower() == "true"
            combined_decision = (row.get("combined_decision") or "").strip()
            baseline_decision = (row.get("baseline_decision") or "").strip()
            expected = "answer" if answerable else "refuse"
            rows.append(
                {
                    "question_id": row.get("question_id"),
                    "category": row.get("category"),
                    "answerable": answerable,
                    "expected_hops": int(row["expected_hops"]) if (row.get("expected_hops") or "").isdigit() else None,
                    "baseline_decision": baseline_decision,
                    "baseline_token_f1": _float(row.get("baseline_token_f1")),
                    "combined_decision": combined_decision,
                    "combined_token_f1": _float(row.get("combined_token_f1")),
                    "combined_decision_correct": combined_decision == expected,
                    "baseline_decision_correct": baseline_decision == expected,
                    "outcome_change": row.get("outcome_change"),
                    "combined_latency_sec": _float(row.get("combined_latency_sec")),
                }
            )
    return {
        "source": "results/phase7/per_question_results.csv",
        "baseline_label": "Frozen Phase 6.4 K8 Baseline",
        "combined_label": "Combined Phase 7 Configuration",
        "total": len(rows),
        "questions": rows,
    }


@router.get(
    "/parametric-leakage",
    summary="Get parametric leakage results: Phase 6 labelled probes and Phase 8.3 measured benchmark (read-only)",
    description=(
        "Returns the parametric_leakage_probes block from "
        "results/phase6/full_benchmark/benchmark_evaluation_summary.json without modification. "
        "These closed-book probes test whether the LLM states facts that post-date the reference time "
        "from its pre-trained knowledge; retrieval filtering does not address this."
    ),
)
@lru_cache(maxsize=1)
def get_parametric_leakage():
    import json
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    path = root / "results" / "phase6" / "full_benchmark" / "benchmark_evaluation_summary.json"
    if not path.exists():
        return {"error": "Phase 6 full benchmark summary not found", "probes": None}
    with open(path, mode="r", encoding="utf-8") as f:
        data = json.load(f)
    phase8_3 = None
    p83 = root / "results" / "phase8_3" / "metrics" / "final_metrics.json"
    if p83.exists():
        with open(p83, mode="r", encoding="utf-8") as f:
            phase8_3 = json.load(f)
    return {
        "source": "results/phase6/full_benchmark/benchmark_evaluation_summary.json",
        "timestamp": data.get("timestamp"),
        "report_title": data.get("report_title"),
        "probes": data.get("parametric_leakage_probes"),
        "phase6_provenance": (
            "Five manually labelled probes about 2024 events. The artifact stores no model responses "
            "(answer=null) and no model name; labels were set by hand. Reported as measured on Llama 3.1 8B, "
            "not the production Qwen3.5:4B model."
        ),
        "phase8_3_source": "results/phase8_3/metrics/final_metrics.json" if phase8_3 is not None else None,
        "phase8_3": phase8_3,
    }
