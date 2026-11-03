"""
Pydantic schemas for Benchmark endpoints.
"""

from __future__ import annotations

from typing import Any, Optional
from pydantic import BaseModel, Field


class CachedBenchmarkResult(BaseModel):
    decision: str = Field(description="'answer' or 'refuse'")
    token_f1: float
    generated_answer: str
    citations: list[str] = Field(default_factory=list)
    latency_sec: float
    outcome_status: str


class DemoQuestionItem(BaseModel):
    question_id: str
    question: str
    category: str
    answerable: bool
    expected_hops: int
    reference_time: Optional[str] = None
    entities: list[str] = Field(default_factory=list)
    ground_truth_answer: str
    unanswerability_rationale: Optional[str] = None
    cached_phase7_result: CachedBenchmarkResult


class DemoQuestionsResponse(BaseModel):
    total_count: int
    benchmark_baseline: str = "Phase 7 Final Baseline"
    questions: list[DemoQuestionItem]
    note: str = (
        "These 5 demonstration questions represent the verified Phase 7 research benchmark. "
        "The cached_phase7_result shows the audited evaluation run. Use POST /api/v1/query "
        "to run live execution through the active FinGraphPipeline."
    )
