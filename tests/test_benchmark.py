from __future__ import annotations

import json
from pathlib import Path
import pytest

from src.evaluation.benchmark import (
    BenchmarkQuestion,
    load_benchmark,
    validate_benchmark_item,
)

BENCHMARK_PATH = Path("benchmark/questions.json")


def test_benchmark_file_exists_and_loads():
    assert BENCHMARK_PATH.exists(), f"Benchmark file does not exist at {BENCHMARK_PATH}"
    questions = load_benchmark(BENCHMARK_PATH, only_verified=True)
    assert len(questions) >= 50, f"Expected at least 50 verified questions, found {len(questions)}"


def test_benchmark_schema_conformance():
    with open(BENCHMARK_PATH, "r", encoding="utf-8") as f:
        raw_items = json.load(f)

    for item in raw_items:
        valid, errors = validate_benchmark_item(item)
        assert valid, f"Validation failed for question {item.get('question_id')}: {errors}"


def test_benchmark_unanswerable_questions_have_rationale():
    questions = load_benchmark(BENCHMARK_PATH, only_verified=True)
    unanswerable = [q for q in questions if not q.answerable]
    assert len(unanswerable) >= 5, "Expected at least 5 unanswerable refusal test cases"

    for q in unanswerable:
        assert q.unanswerability_rationale, f"Question {q.question_id} missing unanswerability rationale"
        assert len(q.relevant_chunk_ids) == 0, f"Unanswerable question {q.question_id} should not have relevant chunks"
        assert "insufficient evidence" in q.ground_truth_answer.lower()


def test_benchmark_answerable_questions_have_chunks():
    questions = load_benchmark(BENCHMARK_PATH, only_verified=True)
    answerable = [q for q in questions if q.answerable]
    assert len(answerable) >= 40

    for q in answerable:
        assert len(q.relevant_chunk_ids) > 0, f"Answerable question {q.question_id} missing relevant chunks"
        assert len(q.relevant_document_ids) > 0, f"Answerable question {q.question_id} missing relevant docs"
        assert q.ground_truth_answer, f"Answerable question {q.question_id} missing ground truth"


def test_benchmark_categories_represented():
    questions = load_benchmark(BENCHMARK_PATH, only_verified=True)
    categories = {q.category for q in questions}
    expected_categories = {
        "factual_single_hop",
        "relational_single_hop",
        "temporal_point_in_time",
        "relational_multi_hop",
        "cross_document_comparative",
    }
    assert categories == expected_categories, f"Categories mismatch: {categories} vs {expected_categories}"
