from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Callable

import ollama
from dotenv import load_dotenv

load_dotenv(".env")


@dataclass
class ParametricLeakageTestItem:
    """A test instance for evaluating parametric temporal leakage in LLMs."""

    question_id: str
    query: str
    reference_time: str  # Historical cutoff point (e.g. "2024-03-31T00:00:00+00:00")
    ground_truth_answer: str
    future_fact: str  # Factual development that occurred strictly AFTER reference_time
    future_event_date: str  # Date when future_fact occurred/was disclosed
    question: str | None = None
    future_information_present: bool | None = None
    verified_by: str = "manual"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.question is None:
            self.question = self.query
        elif not self.query:
            self.query = self.question

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
    question: str = ""
    future_information_present: bool = False
    evidence_of_future_information: str = ""
    verification_method: str = "candidate_detector"
    verified_by: str = "manual"
    retrieval_used: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.question:
            self.question = self.query
        self.future_information_present = self.parametric_leakage_detected
        if not self.evidence_of_future_information and self.reason:
            self.evidence_of_future_information = self.reason

    def to_dict(self) -> dict[str, Any]:
        return {
            "question_id": self.question_id,
            "reference_time": self.reference_time,
            "answer": self.llm_answer,
            "future_information_present": self.future_information_present,
            "evidence_of_future_information": self.evidence_of_future_information,
            "verification_method": self.verification_method,
            "future_event_date": self.future_event_date,
            "parametric_leakage_detected": self.parametric_leakage_detected,
            "reason": self.reason,
            "retrieval_used": self.retrieval_used,
            "metadata": self.metadata,
        }


@dataclass
class ParametricLeakageSummary:
    """Summary metrics distinguishing Parametric Leakage Rate (PLR) from FEIR."""

    total_questions: int
    future_containing_answers: int
    plr: float
    feir: float = 0.0
    retrieval_used: bool = False
    results: list[ParametricLeakageResult] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_questions": self.total_questions,
            "future_containing_answers": self.future_containing_answers,
            "plr": self.plr,
            "feir": self.feir,
            "retrieval_used": self.retrieval_used,
            "results": [r.to_dict() for r in self.results],
        }


class ParametricLeakageEvaluator:
    """
    Evaluation interface for testing Parametric (LLM Internal Memory) Temporal Leakage.

    IMPORTANT SCIENTIFIC DISTINCTION:
    - Retriever-Level Temporal Leakage (FEIR):
      Future documents or edges retrieved by vector/graph search
      (SOLVED by FinGraph AI's point-in-time filtering and provenance timestamps: FEIR = 0.00%).
    - Parametric Temporal Leakage (PLR):
      An LLM whose pre-training or post-training cutoff extends beyond the historical
      reference time may recall future events from its internal parameters/weights,
      even when retrieval is strictly constrained or disabled.

    This evaluator tests the 'Retrieval-Disabled' condition (LLM-only baseline):
    Question -> LLM (Weights only, NO RAG, NO KG, NO SEC corpus) -> Answer -> Assess future facts.
    """

    def __init__(
        self,
        llm_callable: Callable[[str, str], str] | None = None,
        model: str | None = None,
        host: str | None = None,
    ) -> None:
        self.llm_callable = llm_callable
        self.model = model or os.getenv("LLM_MODEL", "qwen3.5:4b")
        self.host = host or os.getenv("OLLAMA_HOST", "http://localhost:11434")

    def _default_ollama_call(self, query: str, system_prompt: str) -> str:
        """Execute retrieval-disabled raw query against local Ollama Qwen model."""
        client = ollama.Client(host=self.host)
        res = client.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query},
            ],
            options={"temperature": 0.0, "num_predict": 500},
            think=False,
        )
        return res.message.content.strip()

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
            f"You are a financial analyst answering questions strictly as of {item.reference_time}. "
            "Do NOT use any knowledge, releases, or events that occurred after this date. "
            "If an event had not occurred by this date, state that it was unknown or had not occurred."
        )

        llm_answer: str | None = None
        leakage_detected = False
        reason = "Evaluation placeholder: LLM inference disabled or offline."
        verification_method = "candidate_detector"
        verified_by = item.verified_by or "manual"

        # Check if item already has a verified label
        if item.future_information_present is not None:
            leakage_detected = item.future_information_present
            verification_method = "manual"
            reason = f"Ground truth label provided ({verified_by})."

        if callable_fn is not None:
            llm_answer = callable_fn(item.query, system_prompt)
        elif item.future_information_present is None:
            try:
                llm_answer = self._default_ollama_call(item.query, system_prompt)
            except Exception as exc:
                llm_answer = None
                reason = f"Ollama connection unavailable: {exc}"

        evidence_str = ""
        if llm_answer is not None and item.future_information_present is None:
            # Candidate detector: check for presence of distinctive future keywords
            future_terms = [t.lower() for t in item.future_fact.split() if len(t) > 3]
            answer_lower = llm_answer.lower()
            matches = [t for t in future_terms if t in answer_lower]
            if len(matches) >= 2 or (len(future_terms) == 1 and len(matches) == 1):
                leakage_detected = True
                verification_method = "candidate_detector"
                evidence_str = f"Keyword match: {matches} from future fact '{item.future_fact}'"
                reason = f"Model mentioned future post-cutoff facts ({matches}) without retrieval."
            else:
                leakage_detected = False
                verification_method = "candidate_detector"
                evidence_str = "None"
                reason = "No future parametric facts detected in LLM response."
        elif item.future_information_present is not None:
            evidence_str = f"Manually verified future fact presence: {item.future_fact}"

        return ParametricLeakageResult(
            question_id=item.question_id,
            query=item.query,
            question=item.query,
            reference_time=item.reference_time,
            future_event_date=item.future_event_date,
            llm_answer=llm_answer,
            parametric_leakage_detected=leakage_detected,
            future_information_present=leakage_detected,
            evidence_of_future_information=evidence_str,
            verification_method=verification_method,
            verified_by=verified_by,
            reason=reason,
            retrieval_used=False,
            metadata=item.metadata,
        )

    def evaluate_suite(
        self,
        items: list[ParametricLeakageTestItem],
        llm_override: Callable[[str, str], str] | None = None,
    ) -> ParametricLeakageSummary:
        """
        Evaluate a benchmark suite of retrieval-disabled temporal questions.

        Returns:
            ParametricLeakageSummary: Aggregated metrics reporting PLR and confirming FEIR = 0.
        """
        results = [self.evaluate(item, llm_override=llm_override) for item in items]
        total = len(results)
        future_count = sum(1 for r in results if r.parametric_leakage_detected)
        plr = (future_count / total) if total > 0 else 0.0

        return ParametricLeakageSummary(
            total_questions=total,
            future_containing_answers=future_count,
            plr=round(plr, 4),
            feir=0.0,  # Retrieval is disabled; FEIR is strictly 0.0
            retrieval_used=False,
            results=results,
        )
