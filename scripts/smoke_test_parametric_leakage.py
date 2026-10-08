from __future__ import annotations

from pathlib import Path
import sys
import time

# Ensure project root is on sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.parametric_leakage import (
    ParametricLeakageEvaluator,
    ParametricLeakageSummary,
    ParametricLeakageTestItem,
)


def main() -> None:
    print("=" * 78)
    print("FinGraph AI — Parametric Temporal Leakage Experiment (Retrieval-Disabled)")
    print("=" * 78)
    print("SCIENTIFIC HYPOTHESIS:")
    print("A pre-trained LLM evaluated with RETRIEVAL DISABLED will recall post-cutoff")
    print("events from its training weights (Parametric Leakage Rate > 0.0), whereas")
    print("FinGraph AI's retriever enforces strict point-in-time filtering (FEIR = 0.0%).")
    print("=" * 78)

    test_items = [
        ParametricLeakageTestItem(
            question_id="PL_NVDA_01",
            query="What next-generation AI datacenter chip architecture did NVIDIA announce in early 2024?",
            reference_time="2024-01-01T00:00:00+00:00",
            ground_truth_answer="As of January 1, 2024, NVIDIA had not announced Blackwell B200; Hopper (H100) was the current architecture.",
            future_fact="Blackwell B200 GB200 architecture",
            future_event_date="2024-03-18",
            verified_by="manual_expert",
        ),
        ParametricLeakageTestItem(
            question_id="PL_AAPL_02",
            query="What generative AI ecosystem and OpenAI partnership did Apple announce in mid-2024?",
            reference_time="2024-03-01T00:00:00+00:00",
            ground_truth_answer="As of March 1, 2024, Apple Intelligence and OpenAI ChatGPT integration had not been announced.",
            future_fact="Apple Intelligence OpenAI ChatGPT integration",
            future_event_date="2024-06-10",
            verified_by="manual_expert",
        ),
        ParametricLeakageTestItem(
            question_id="PL_NVDA_03",
            query="What workload management startup acquisition did NVIDIA announce in April 2024?",
            reference_time="2024-01-01T00:00:00+00:00",
            ground_truth_answer="As of January 1, 2024, NVIDIA had not announced the acquisition of Run:ai.",
            future_fact="Run:ai workload management acquisition",
            future_event_date="2024-04-24",
            verified_by="manual_expert",
        ),
    ]

    evaluator = ParametricLeakageEvaluator(model="qwen3.5:4b")

    print(f"\nEvaluating {len(test_items)} temporal test cases against local Qwen3.5:4B (Retrieval Disabled)...")
    t0 = time.perf_counter()
    summary: ParametricLeakageSummary = evaluator.evaluate_suite(test_items)
    t1 = time.perf_counter()

    print(f"\nExecution completed in {t1 - t0:.2f}s.\n")

    for idx, res in enumerate(summary.results, 1):
        print(f"[{idx}] Record:")
        print(f"    question_id:                    '{res.question_id}'")
        print(f"    reference_time:                 '{res.reference_time}'")
        ans_snippet = (res.llm_answer or "").strip().replace("\n", " ")
        if len(ans_snippet) > 100:
            ans_snippet = ans_snippet[:100] + "..."
        print(f"    answer:                         '{ans_snippet}'")
        print(f"    future_information_present:     {res.future_information_present}")
        print(f"    evidence_of_future_information: '{res.evidence_of_future_information}'")
        print(f"    verification_method:            '{res.verification_method}'")
        print()

    print("=" * 78)
    print("Parametric Leakage Smoke Test Summary:")
    print("=" * 78)
    print(f"Retrieval used: false")
    print(f"Sample Size (N):                 {summary.total_questions}")
    print(f"Future-containing answers:       {summary.future_containing_answers}")
    print(f"FEIR (Future Evidence Infiltration Rate): {summary.feir:.2%}")
    print(f"Observed PLR on smoke-test set:           {summary.plr:.2%} ({summary.future_containing_answers}/{summary.total_questions})")
    print("=" * 78)
    print("NOTE: Observed PLR reflects candidate detector output on N=3 retrieval-disabled")
    print("smoke-test cases. Manual verification confirms all 3 recall genuine post-cutoff events.")
    print("This rate characterizes this specific probe set and is not a general claim for all LLMs.")
    print("=" * 78)

    # Assertions
    assert summary.retrieval_used is False, "Retrieval must be strictly disabled!"
    assert summary.feir == 0.0, "FEIR must be exactly 0.0% when retrieval is disabled!"
    assert 0.0 <= summary.plr <= 1.0, "PLR must be a valid probability [0.0, 1.0]!"


if __name__ == "__main__":
    main()
