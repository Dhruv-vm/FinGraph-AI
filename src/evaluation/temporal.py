from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Sequence

from src.data.graph.snapshot import is_available, parse_datetime


def calculate_feir(
    evidence_items: Sequence[Any] | None,
    reference_time: str | datetime | None,
) -> float | None:
    """
    Calculate Future Evidence Infiltration Rate (FEIR) for a collection of evidence items.

    FEIR = (Count of evidence items where available_time > reference_time) / (Total evidence items)

    Returns:
        float | None: Infiltration rate in [0.0, 1.0], or None if evidence_items is None,
                      empty (undefined denominator 0/0), or reference_time is None.
    """
    if evidence_items is None or reference_time is None:
        return None

    ref_dt = parse_datetime(reference_time) if isinstance(reference_time, str) else reference_time
    if ref_dt is None:
        return None

    total = len(evidence_items)
    if total == 0:
        return None

    violations = 0
    for ev in evidence_items:
        # Extract available_time whether ev is a RetrievalEvidence object or dict
        avail = None
        if hasattr(ev, "available_time"):
            avail = ev.available_time
        elif isinstance(ev, dict):
            avail = ev.get("available_time")

        if avail is not None:
            if not is_available(avail, ref_dt):
                violations += 1

    return violations / total


@dataclass
class TemporalStageMetrics:
    """Metrics tracking temporal integrity across the four pipeline stages."""

    feir_pre_filter: float | None
    feir_post_filter: float | None
    feir_context: float | None
    feir_cited: float | None
    gate_rejections: int
    total_pre_filter_items: int
    total_post_filter_items: int
    total_context_items: int
    total_cited_items: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "feir_pre_filter": round(self.feir_pre_filter, 4) if self.feir_pre_filter is not None else None,
            "feir_post_filter": round(self.feir_post_filter, 4) if self.feir_post_filter is not None else None,
            "feir_context": round(self.feir_context, 4) if self.feir_context is not None else None,
            "feir_cited": round(self.feir_cited, 4) if self.feir_cited is not None else None,
            "gate_rejections": self.gate_rejections,
            "total_pre_filter_items": self.total_pre_filter_items,
            "total_post_filter_items": self.total_post_filter_items,
            "total_context_items": self.total_context_items,
            "total_cited_items": self.total_cited_items,
        }


@dataclass
class SingleTemporalResult:
    """Temporal evaluation output for a single point-in-time question."""

    question_id: str
    reference_time: str | None
    pre_filter_feir: float | None
    post_filter_feir: float | None
    context_feir: float | None
    cited_feir: float | None
    gate_rejections: int
    is_temporally_valid: bool
    pre_filter_count: int = 0
    post_filter_count: int = 0
    context_count: int = 0
    cited_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        res = asdict(self)
        if self.pre_filter_feir is not None:
            res["pre_filter_feir"] = round(self.pre_filter_feir, 4)
        if self.post_filter_feir is not None:
            res["post_filter_feir"] = round(self.post_filter_feir, 4)
        if self.context_feir is not None:
            res["context_feir"] = round(self.context_feir, 4)
        if self.cited_feir is not None:
            res["cited_feir"] = round(self.cited_feir, 4)
        return res


@dataclass
class AggregatedTemporalMetrics:
    """Aggregated temporal metrics across all evaluated queries."""

    total_queries: int
    temporal_queries: int
    stage_metrics: TemporalStageMetrics
    temporal_compliance_rate: float
    per_question_results: list[SingleTemporalResult] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_queries": self.total_queries,
            "temporal_queries": self.temporal_queries,
            "stage_metrics": self.stage_metrics.to_dict(),
            "temporal_compliance_rate": round(self.temporal_compliance_rate, 4),
        }


@dataclass
class TemporalAblationResult:
    """Ablation isolating causal reduction in FEIR from explicit temporal filtering."""

    with_filtering_feir: float
    without_filtering_feir: float
    absolute_reduction: float
    relative_reduction_pct: float
    gate_rejections: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "with_filtering_feir": round(self.with_filtering_feir, 4),
            "without_filtering_feir": round(self.without_filtering_feir, 4),
            "absolute_reduction": round(self.absolute_reduction, 4),
            "relative_reduction_pct": round(self.relative_reduction_pct, 2),
            "gate_rejections": self.gate_rejections,
        }


class TemporalEvaluator:
    """
    Calculates dynamic FEIR across pipeline stages and evaluates point-in-time gate rejection.
    """

    def evaluate_single(
        self,
        question_id: str,
        reference_time: str | datetime | None,
        pre_filter_evidence: Sequence[Any] | None = None,
        post_filter_evidence: Sequence[Any] | None = None,
        context_evidence: Sequence[Any] | None = None,
        cited_evidence: Sequence[Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> SingleTemporalResult:
        meta = dict(metadata or {})
        ref_dt = parse_datetime(reference_time) if isinstance(reference_time, str) else reference_time

        # Track actual item counts per stage
        pre_count = len(pre_filter_evidence) if pre_filter_evidence is not None else 0
        post_count = len(post_filter_evidence) if post_filter_evidence is not None else 0
        ctx_count = len(context_evidence) if context_evidence is not None else 0
        cited_count = len(cited_evidence) if cited_evidence is not None else 0

        pre_feir = calculate_feir(pre_filter_evidence, reference_time)
        post_feir = calculate_feir(post_filter_evidence, reference_time)
        ctx_feir = calculate_feir(context_evidence, reference_time)
        cited_feir = calculate_feir(cited_evidence, reference_time)

        # Count rejected future items from pre-filter stage if pre_filter_evidence was provided
        rejections = 0
        if pre_filter_evidence is not None and ref_dt is not None:
            for ev in pre_filter_evidence:
                avail = getattr(ev, "available_time", None) or (
                    ev.get("available_time") if isinstance(ev, dict) else None
                )
                if avail is not None and not is_available(avail, ref_dt):
                    rejections += 1
        elif pre_filter_evidence is None:
            meta.setdefault("pre_filter_stage", "unavailable_in_single_pass_retrieval")

        # Temporal validity determination:
        # A temporal query is valid iff at least one evidence item was evaluated across
        # downstream stages (post-filter, context, cited) AND zero temporal violations occurred.
        # Empty evidence across all stages cannot be claimed as empirical proof of compliance.
        has_evaluated_items = (post_count + ctx_count + cited_count) > 0
        has_violations = (
            (post_feir is not None and post_feir > 0.0)
            or (ctx_feir is not None and ctx_feir > 0.0)
            or (cited_feir is not None and cited_feir > 0.0)
        )

        if ref_dt is None:
            is_valid = True
        else:
            is_valid = has_evaluated_items and not has_violations

        return SingleTemporalResult(
            question_id=question_id,
            reference_time=str(reference_time) if reference_time else None,
            pre_filter_feir=pre_feir,
            post_filter_feir=post_feir,
            context_feir=ctx_feir,
            cited_feir=cited_feir,
            gate_rejections=rejections,
            is_temporally_valid=is_valid,
            pre_filter_count=pre_count,
            post_filter_count=post_count,
            context_count=ctx_count,
            cited_count=cited_count,
            metadata=meta,
        )

    def evaluate_suite(
        self,
        results: list[SingleTemporalResult],
        pre_filter_all: Sequence[Any] | None = None,
        post_filter_all: Sequence[Any] | None = None,
        context_all: Sequence[Any] | None = None,
        cited_all: Sequence[Any] | None = None,
    ) -> AggregatedTemporalMetrics:
        total = len(results)
        temporal_qs = [r for r in results if r.reference_time is not None]
        n_temp = len(temporal_qs)

        if n_temp == 0:
            stage_metrics = TemporalStageMetrics(
                feir_pre_filter=None,
                feir_post_filter=None,
                feir_context=None,
                feir_cited=None,
                gate_rejections=0,
                total_pre_filter_items=0,
                total_post_filter_items=0,
                total_context_items=0,
                total_cited_items=0,
            )
            return AggregatedTemporalMetrics(
                total_queries=total,
                temporal_queries=0,
                stage_metrics=stage_metrics,
                temporal_compliance_rate=0.0,
                per_question_results=results,
            )

        pre_vals = [r.pre_filter_feir for r in temporal_qs if r.pre_filter_feir is not None]
        post_vals = [r.post_filter_feir for r in temporal_qs if r.post_filter_feir is not None]
        ctx_vals = [r.context_feir for r in temporal_qs if r.context_feir is not None]
        cited_vals = [r.cited_feir for r in temporal_qs if r.cited_feir is not None]

        mean_pre = (sum(pre_vals) / len(pre_vals)) if pre_vals else None
        mean_post = (sum(post_vals) / len(post_vals)) if post_vals else None
        mean_ctx = (sum(ctx_vals) / len(ctx_vals)) if ctx_vals else None
        mean_cited = (sum(cited_vals) / len(cited_vals)) if cited_vals else None

        total_rejections = sum(r.gate_rejections for r in temporal_qs)
        compliant_count = sum(1 for r in temporal_qs if r.is_temporally_valid)

        # Stage item counts: prioritize explicitly provided sequences if any;
        # otherwise aggregate from SingleTemporalResult item counts.
        total_pre_items = (
            len(pre_filter_all)
            if pre_filter_all is not None
            else sum(getattr(r, "pre_filter_count", 0) for r in temporal_qs)
        )
        total_post_items = (
            len(post_filter_all)
            if post_filter_all is not None
            else sum(getattr(r, "post_filter_count", 0) for r in temporal_qs)
        )
        total_ctx_items = (
            len(context_all)
            if context_all is not None
            else sum(getattr(r, "context_count", 0) for r in temporal_qs)
        )
        total_cited_items = (
            len(cited_all)
            if cited_all is not None
            else sum(getattr(r, "cited_count", 0) for r in temporal_qs)
        )

        stage_metrics = TemporalStageMetrics(
            feir_pre_filter=mean_pre,
            feir_post_filter=mean_post,
            feir_context=mean_ctx,
            feir_cited=mean_cited,
            gate_rejections=total_rejections,
            total_pre_filter_items=total_pre_items,
            total_post_filter_items=total_post_items,
            total_context_items=total_ctx_items,
            total_cited_items=total_cited_items,
        )

        return AggregatedTemporalMetrics(
            total_queries=total,
            temporal_queries=n_temp,
            stage_metrics=stage_metrics,
            temporal_compliance_rate=compliant_count / n_temp,
            per_question_results=results,
        )

    def evaluate_ablation(
        self,
        with_filtering_evidence: Sequence[Any],
        without_filtering_evidence: Sequence[Any],
        reference_time: str | datetime,
    ) -> TemporalAblationResult:
        """
        Ablation study isolating the direct causal impact of temporal filtering.
        """
        feir_with = calculate_feir(with_filtering_evidence, reference_time)
        feir_without = calculate_feir(without_filtering_evidence, reference_time)

        fw = feir_with if feir_with is not None else 0.0
        fwo = feir_without if feir_without is not None else 0.0

        diff = fwo - fw
        rel_pct = (diff / fwo * 100.0) if fwo > 0 else 0.0

        ref_dt = parse_datetime(reference_time) if isinstance(reference_time, str) else reference_time
        rejections = 0
        if ref_dt is not None:
            for ev in without_filtering_evidence:
                avail = getattr(ev, "available_time", None) or (ev.get("available_time") if isinstance(ev, dict) else None)
                if avail is not None and not is_available(avail, ref_dt):
                    rejections += 1

        return TemporalAblationResult(
            with_filtering_feir=fw,
            without_filtering_feir=fwo,
            absolute_reduction=diff,
            relative_reduction_pct=rel_pct,
            gate_rejections=rejections,
        )
