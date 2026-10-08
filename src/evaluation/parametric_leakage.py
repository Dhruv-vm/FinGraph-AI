from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Callable


@dataclass
class ParametricLeakageTestItem:
    """A test instance for evaluating parametric temporal leakage in LLMs."""

    question_id: str
    query: str
    reference_time: str  # Historical cutoff point (e.g. "2024-03-31T00:00:00+00:00")
    ground_truth_answer: str
    future_fact: str  # Factual development that occurred strictly AFTER reference_time
    future_event_date: str  # Date when future_fact occurred/was disclosed
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ParametricLeakageResult:
    """Evaluation output measuring parametric leakage without retrieval."""

    question_id: str
    query: str
    reference_time: str
    future_event_date: str
    llm_answer: str | None
    parametric_leakage_detected: bool
    reason: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ParametricLeakageEvaluator:
    """
    Evaluation interface for testing Parametric (LLM Internal Memory) Temporal Leakage.

    IMPORTANT SCIENTIFIC DISTINCTION:
    - Retriever-Level Temporal Leakage: Future documents or edges retrieved by vector/graph search
      (SOLVED by FinGraph AI's point-in-time filtering and provenance timestamps).
    - Parametric Temporal Leakage: An LLM whose pre-training or post-training cutoff extends beyond
      the historical reference time may recall future events from its weights, even when retrieval
      is strictly constrained or disabled.

    This evaluator tests the 'Retrieval-Disabled' condition (LLM-only baseline):
    Question -> LLM (Weights only, NO RAG) -> Answer -> Compare against post-cutoff knowledge.
    """

    def __init__(self, llm_callable: Callable[[str, str], str] | None = None) -> None:
        self.llm_callable = llm_callable

    def evaluate(
        self,
        item: ParametricLeakageTestItem,
        llm_override: Callable[[str, str], str] | None = None,
    ) -> ParametricLeakageResult:
        """
        Evaluate a single test case for parametric leakage with zero retrieval.

        Args:
            item: Test case containing query, reference cutoff, and future fact.
            llm_override: Optional callable invoking the raw LLM.

        Returns:
            ParametricLeakageResult: Evaluation report indicating whether the model hallucinated
                                     or leaked future information from its weights.
        """
        callable_fn = llm_override or self.llm_callable

        system_prompt = (
            f"You are a financial analyst answering questions as of {item.reference_time}. "
            "Do NOT use any knowledge or events that occurred after this date."
        )

        llm_answer: str | None = None
        leakage_detected = False
        reason = "Evaluation placeholder: LLM inference disabled or offline."

        if callable_fn is not None:
            llm_answer = callable_fn(item.query, system_prompt)
            # Detect whether future fact terms appear in the retrieval-disabled LLM answer
            future_terms = [t.lower() for t in item.future_fact.split() if len(t) > 3]
            answer_lower = llm_answer.lower()
            matches = [t for t in future_terms if t in answer_lower]
            if len(matches) >= 2:
                leakage_detected = True
                reason = f"Model mentioned future post-cutoff facts ({matches}) without retrieval."
            else:
                leakage_detected = False
                reason = "No future parametric facts detected in LLM response."

        return ParametricLeakageResult(
            question_id=item.question_id,
            query=item.query,
            reference_time=item.reference_time,
            future_event_date=item.future_event_date,
            llm_answer=llm_answer,
            parametric_leakage_detected=leakage_detected,
            reason=reason,
        )
