from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class BenchmarkQuestion:
    """A verified benchmark question instance with unit-level ground truth."""

    question_id: str
    question: str
    category: str
    hop_level: int
    entities: list[str]
    reference_time: str | None
    expected_reasoning_hops: int
    relevant_chunk_ids: list[str]
    relevant_document_ids: list[str]
    expected_graph_path: list[list[str]]
    ground_truth_answer: str
    required_evidence: list[str]
    answerable: bool
    unanswerability_rationale: str | None = None
    verification_status: str = "verified"
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BenchmarkQuestion:
        return cls(
            question_id=data["question_id"],
            question=data["question"],
            category=data["category"],
            hop_level=int(data["hop_level"]),
            entities=list(data.get("entities", [])),
            reference_time=data.get("reference_time"),
            expected_reasoning_hops=int(data.get("expected_reasoning_hops", 1)),
            relevant_chunk_ids=list(data.get("relevant_chunk_ids", [])),
            relevant_document_ids=list(data.get("relevant_document_ids", [])),
            expected_graph_path=[list(p) for p in data.get("expected_graph_path", [])],
            ground_truth_answer=data["ground_truth_answer"],
            required_evidence=list(data.get("required_evidence", [])),
            answerable=bool(data.get("answerable", True)),
            unanswerability_rationale=data.get("unanswerability_rationale"),
            verification_status=data.get("verification_status", "drafted"),
            metadata=data.get("metadata", {}),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_benchmark(
    path: str | Path = "benchmark/questions.json",
    only_verified: bool = True,
) -> list[BenchmarkQuestion]:
    """
    Load benchmark questions from JSON file with optional verification filtering.
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Benchmark file not found: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        raw_items = json.load(f)

    questions = [BenchmarkQuestion.from_dict(item) for item in raw_items]
    if only_verified:
        questions = [q for q in questions if q.verification_status == "verified"]

    return questions


def validate_benchmark_item(item: dict[str, Any]) -> tuple[bool, list[str]]:
    """
    Validate that a dictionary adheres strictly to the benchmark schema.
    """
    errors: list[str] = []
    required_keys = {
        "question_id",
        "question",
        "category",
        "hop_level",
        "entities",
        "reference_time",
        "expected_reasoning_hops",
        "relevant_chunk_ids",
        "relevant_document_ids",
        "expected_graph_path",
        "ground_truth_answer",
        "required_evidence",
        "answerable",
        "verification_status",
    }
    missing = required_keys - set(item.keys())
    if missing:
        errors.append(f"Missing required fields: {sorted(missing)}")

    valid_categories = {
        "factual_single_hop",
        "relational_single_hop",
        "temporal_point_in_time",
        "relational_multi_hop",
        "cross_document_comparative",
    }
    if item.get("category") not in valid_categories:
        errors.append(f"Invalid category: {item.get('category')}")

    if not isinstance(item.get("relevant_chunk_ids"), list):
        errors.append("'relevant_chunk_ids' must be a list of strings")

    if not isinstance(item.get("relevant_document_ids"), list):
        errors.append("'relevant_document_ids' must be a list of strings")

    if item.get("answerable") is False and not item.get("unanswerability_rationale"):
        errors.append("Unanswerable question must include 'unanswerability_rationale'")

    return len(errors) == 0, errors
