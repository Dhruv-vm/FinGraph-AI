from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from src.evaluation.parametric_leakage import (
    ParametricLeakageEvaluator,
    ParametricLeakageResult,
    ParametricLeakageSummary,
    ParametricLeakageTestItem,
)
from src.evaluation.temporal import TemporalEvaluator, TemporalStageMetrics


@dataclass
class UnifiedLeakageReport:
    """
    Unified report coordinating Retriever-Level Temporal Leakage (FEIR)
    with Parametric LLM-Memory Temporal Leakage (PLR).
    """

    retriever_feir: float
    parametric_plr: float
    feir_stage_metrics: dict[str, Any]
    plr_candidate_positives: int
    plr_manually_verified: int
    plr_confirmed_non_leakage: int
    plr_uncertain: int
    plr_total_probes: int
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class LeakageEvaluator:
    """
    Coordinates evaluation of both forms of temporal leakage:
    1. Retrieval-level leakage (FEIR): evaluated on SEC retrieval pipeline.
    2. Parametric weight leakage (PLR): evaluated on retrieval-disabled LLM probes.
    """

    def __init__(
        self,
        temporal_evaluator: TemporalEvaluator | None = None,
        parametric_evaluator: ParametricLeakageEvaluator | None = None,
    ) -> None:
        self.temporal_evaluator = temporal_evaluator or TemporalEvaluator()
        self.parametric_evaluator = parametric_evaluator or ParametricLeakageEvaluator()

    def evaluate_leakage(
        self,
        stage_metrics: TemporalStageMetrics,
        probe_items: list[ParametricLeakageTestItem],
        llm_override: Any | None = None,
    ) -> UnifiedLeakageReport:
        # Run parametric leakage probe suite
        plr_summary: ParametricLeakageSummary = self.parametric_evaluator.evaluate_suite(
            items=probe_items,
            llm_override=llm_override,
        )

        candidate_pos = 0
        manually_verified = 0
        confirmed_non_leakage = 0
        uncertain = 0

        for r in plr_summary.results:
            if r.verification_method == "manual":
                if r.parametric_leakage_detected:
                    manually_verified += 1
                else:
                    confirmed_non_leakage += 1
            else:
                if r.parametric_leakage_detected:
                    candidate_pos += 1
                else:
                    confirmed_non_leakage += 1

        retriever_feir = (
            round(stage_metrics.feir_post_filter, 4)
            if stage_metrics.feir_post_filter is not None
            else 0.0
        )
        return UnifiedLeakageReport(
            retriever_feir=retriever_feir,
            parametric_plr=plr_summary.plr,
            feir_stage_metrics=stage_metrics.to_dict(),
            plr_candidate_positives=candidate_pos,
            plr_manually_verified=manually_verified,
            plr_confirmed_non_leakage=confirmed_non_leakage,
            plr_uncertain=uncertain,
            plr_total_probes=plr_summary.total_questions,
            details={
                "plr_results": [r.to_dict() for r in plr_summary.results],
            },
        )
