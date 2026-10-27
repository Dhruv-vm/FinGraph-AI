from __future__ import annotations

import re
import string
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any


def normalize_answer(text: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""
    text = text.lower()
    # Remove articles
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    # Remove punctuation
    text = "".join(ch for ch in text if ch not in string.punctuation)
    # Normalize whitespace
    return " ".join(text.split())


def exact_match_score(prediction: str, ground_truth: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def token_f1_score(prediction: str, ground_truth: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gt_tokens = normalize_answer(ground_truth).split()

    if not pred_tokens or not gt_tokens:
        return 1.0 if pred_tokens == gt_tokens else 0.0

    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gt_tokens)
    return (2 * precision * recall) / (precision + recall)


@dataclass
class RefusalMetrics:
    total_unanswerable: int
    total_answerable: int
    correct_refusals: int  # Refused when unanswerable (True Positive)
    false_refusals: int    # Refused when answerable (False Positive)
    hallucinations: int    # Answered when unanswerable (False Negative)
    correct_answers: int   # Answered when answerable (True Negative)
    precision: float
    recall: float
    f1: float
    accuracy: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_unanswerable": self.total_unanswerable,
            "total_answerable": self.total_answerable,
            "correct_refusals": self.correct_refusals,
            "false_refusals": self.false_refusals,
            "hallucinations": self.hallucinations,
            "correct_answers": self.correct_answers,
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "accuracy": round(self.accuracy, 4),
        }


@dataclass
class CitationMetrics:
    total_citations: int
    valid_citations: int  # Points to a valid chunk in package
    relevant_citations: int  # Points to ground-truth relevant chunk
    precision: float
    recall: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_citations": self.total_citations,
            "valid_citations": self.valid_citations,
            "relevant_citations": self.relevant_citations,
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
        }


@dataclass
class SingleQAResult:
    question_id: str
    predicted_answer: str
    ground_truth_answer: str
    exact_match: float
    token_f1: float
    is_refusal: bool
    is_answerable: bool
    is_correct_refusal: bool
    confidence: float
    evidence_ids_cited: list[str]
    citations: list[dict[str, Any]]
    validation_passed: bool
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AggregatedQAMetrics:
    total_questions: int
    mean_exact_match: float
    mean_token_f1: float
    refusal_metrics: RefusalMetrics
    citation_metrics: CitationMetrics
    validation_pass_rate: float
    per_question_results: list[SingleQAResult] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_questions": self.total_questions,
            "mean_exact_match": round(self.mean_exact_match, 4),
            "mean_token_f1": round(self.mean_token_f1, 4),
            "refusal_metrics": self.refusal_metrics.to_dict(),
            "citation_metrics": self.citation_metrics.to_dict(),
            "validation_pass_rate": round(self.validation_pass_rate, 4),
        }


class QAEvaluator:
    """
    Evaluates evidence-grounded generative financial QA, including exact match,
    token F1, refusal accuracy, citation quality, and structural validation.
    """

    REFUSAL_PHRASES = (
        "insufficient evidence",
        "insufficient information",
        "not provided in the evidence",
        "cannot be determined",
        "no evidence",
    )

    def is_refusal_text(self, text: str) -> bool:
        low = text.lower()
        return any(phrase in low for phrase in self.REFUSAL_PHRASES)

    def evaluate_single(
        self,
        question_id: str,
        predicted_answer: str,
        ground_truth_answer: str,
        is_answerable: bool,
        confidence: float = 1.0,
        evidence_ids_cited: list[str] | None = None,
        citations: list[dict[str, Any]] | None = None,
        validation_passed: bool = True,
        metadata: dict[str, Any] | None = None,
    ) -> SingleQAResult:
        cited_ids = evidence_ids_cited or []
        pred_refusal = self.is_refusal_text(predicted_answer)

        # Correct refusal: model refused AND question is unanswerable
        correct_refusal = pred_refusal and not is_answerable

        # Compute EM and F1
        em = exact_match_score(predicted_answer, ground_truth_answer)
        f1 = token_f1_score(predicted_answer, ground_truth_answer)

        # If it was unanswerable and model refused, give full credit on F1
        if not is_answerable and pred_refusal:
            f1 = 1.0
            em = 1.0

        return SingleQAResult(
            question_id=question_id,
            predicted_answer=predicted_answer,
            ground_truth_answer=ground_truth_answer,
            exact_match=em,
            token_f1=f1,
            is_refusal=pred_refusal,
            is_answerable=is_answerable,
            is_correct_refusal=correct_refusal,
            confidence=confidence,
            evidence_ids_cited=cited_ids,
            citations=citations or [],
            validation_passed=validation_passed,
            metadata=metadata or {},
        )

    def evaluate_suite(
        self,
        results: list[SingleQAResult],
        relevant_chunks_map: dict[str, list[str]] | None = None,
    ) -> AggregatedQAMetrics:
        total = len(results)
        if total == 0:
            return AggregatedQAMetrics(
                total_questions=0,
                mean_exact_match=0.0,
                mean_token_f1=0.0,
                refusal_metrics=RefusalMetrics(0, 0, 0, 0, 0, 0, 0.0, 0.0, 0.0, 0.0),
                citation_metrics=CitationMetrics(0, 0, 0, 0.0, 0.0),
                validation_pass_rate=0.0,
                per_question_results=[],
            )

        mean_em = sum(r.exact_match for r in results) / total
        mean_f1 = sum(r.token_f1 for r in results) / total
        val_pass = sum(1 for r in results if r.validation_passed) / total

        # Refusal metrics
        unanswerable = [r for r in results if not r.is_answerable]
        answerable = [r for r in results if r.is_answerable]

        tp = sum(1 for r in unanswerable if r.is_refusal)  # correct refusal
        fn = sum(1 for r in unanswerable if not r.is_refusal)  # hallucination
        fp = sum(1 for r in answerable if r.is_refusal)  # false refusal
        tn = sum(1 for r in answerable if not r.is_refusal)  # correct answer

        prec = (tp / (tp + fp)) if (tp + fp) > 0 else (1.0 if not unanswerable else 0.0)
        rec = (tp / (tp + fn)) if (tp + fn) > 0 else 1.0
        f1_ref = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        acc = (tp + tn) / total

        ref_metrics = RefusalMetrics(
            total_unanswerable=len(unanswerable),
            total_answerable=len(answerable),
            correct_refusals=tp,
            false_refusals=fp,
            hallucinations=fn,
            correct_answers=tn,
            precision=prec,
            recall=rec,
            f1=f1_ref,
            accuracy=acc,
        )

        # Citation metrics
        total_cits = 0
        valid_cits = 0
        relevant_cits = 0
        rel_map = relevant_chunks_map or {}

        for r in results:
            q_rel_chunks = set(rel_map.get(r.question_id, []))
            for c in r.citations:
                total_cits += 1
                cid = c.get("chunk_id")
                if cid:
                    valid_cits += 1
                    if cid in q_rel_chunks:
                        relevant_cits += 1

        cit_prec = (relevant_cits / total_cits) if total_cits > 0 else 1.0
        total_rel_chunks_all = sum(len(chunks) for chunks in rel_map.values())
        cit_rec = (relevant_cits / total_rel_chunks_all) if total_rel_chunks_all > 0 else 0.0

        cit_metrics = CitationMetrics(
            total_citations=total_cits,
            valid_citations=valid_cits,
            relevant_citations=relevant_cits,
            precision=cit_prec,
            recall=cit_rec,
        )

        return AggregatedQAMetrics(
            total_questions=total,
            mean_exact_match=mean_em,
            mean_token_f1=mean_f1,
            refusal_metrics=ref_metrics,
            citation_metrics=cit_metrics,
            validation_pass_rate=val_pass,
            per_question_results=results,
        )
