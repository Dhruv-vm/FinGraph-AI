"""
FinGraph AI Pipeline Service.

Executes FinGraphPipeline queries with timing, serialization, and error translation.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import logging
import time
from typing import Any

from api.schemas.query import (
    CitationItem,
    EvidenceItem,
    QueryRequest,
    QueryResponse,
    SufficiencyDetail,
    TemporalValidationDetail,
)
from scripts.run_fingraph import FinGraphPipeline

logger = logging.getLogger("fingraph_api.services.pipeline")


class PipelineService:
    def __init__(self, pipeline: FinGraphPipeline, lock: asyncio.Lock) -> None:
        self.pipeline = pipeline
        self.lock = lock

    async def execute_query(self, request: QueryRequest) -> QueryResponse:
        """
        Execute a query against the FinGraphPipeline in an async thread pool with concurrency locking.
        """
        start_time = time.perf_counter()

        async with self.lock:
            # Run the CPU/IO-bound pipeline.answer inside asyncio.to_thread
            package, answer = await asyncio.to_thread(
                self.pipeline.answer,
                query=request.query,
                reference_time=request.reference_time,
                top_k=request.top_k or 10,
                candidate_pool_size=request.candidate_pool_size or 20,
                mode=request.mode,
            )

        elapsed_ms = round((time.perf_counter() - start_time) * 1000.0, 2)

        # Determine decision ('refuse' vs 'answer')
        is_refusal = (
            answer.insufficient_evidence
            or "Insufficient evidence in the retrieved corpus." in (answer.answer or "")
        )
        decision = "refuse" if is_refusal else "answer"

        # Serialize Evidence
        evidence_items: list[EvidenceItem] = []
        for rank, ev in enumerate(package.all_evidence, start=1):
            evidence_items.append(
                EvidenceItem(
                    evidence_id=ev.evidence_id,
                    source_type=ev.source_type,
                    score=float(ev.score),
                    rank=rank,
                    text=ev.text,
                    document_id=ev.document_id,
                    chunk_id=ev.chunk_id,
                    available_time=ev.available_time,
                    event_time=ev.metadata.get("event_time"),
                    publication_date=ev.metadata.get("publication_date"),
                    source_url=ev.metadata.get("source_url"),
                    source_reference=ev.metadata.get("source_reference"),
                    source_company=ev.metadata.get("source_company"),
                    provenance_verified=ev.metadata.get("provenance_verified", True),
                    attribution_status=ev.metadata.get("attribution_status"),
                    metadata=ev.metadata or {},
                )
            )

        # Serialize Citations
        citations: list[CitationItem] = []
        for cit in answer.citations or []:
            citations.append(
                CitationItem(
                    evidence_id=cit.get("evidence_id", ""),
                    document_id=cit.get("document_id"),
                    chunk_id=cit.get("chunk_id"),
                    source_type=cit.get("source_type"),
                    available_time=cit.get("available_time"),
                    text_snippet=cit.get("text_snippet"),
                )
            )

        # Serialize Sufficiency
        s_res = getattr(package, "sufficiency_result", None)
        sufficiency_detail = SufficiencyDetail(
            sufficient=s_res.sufficient if s_res is not None else package.sufficient,
            reason=s_res.reason if s_res is not None else "",
            missing_entities=s_res.missing_entities if s_res is not None else [],
            missing_relationships=s_res.missing_relationships if s_res is not None else [],
            required_hops=s_res.required_hops if s_res is not None else package.expected_hops,
            observed_hops=s_res.observed_hops if s_res is not None else 1,
            evidence_count=s_res.evidence_count if s_res is not None else len(package.all_evidence),
            temporal_valid=s_res.temporal_valid if s_res is not None else package.temporal_valid,
            provenance_complete=s_res.provenance_complete if s_res is not None else True,
        )

        # Serialize Temporal Validation
        temporal_detail = TemporalValidationDetail(
            temporal_valid=package.temporal_valid,
            reference_time=package.reference_time,
            context_items_count=len(package.primary_evidence),
            future_items_count=0,
            feir=0.0,
        )

        return QueryResponse(
            query=request.query,
            mode=request.mode,
            decision=decision,
            answer=answer.answer,
            insufficient_evidence=answer.insufficient_evidence,
            confidence=answer.confidence,
            reasoning_summary=answer.reasoning_summary,
            citations=citations,
            evidence=evidence_items,
            graph_paths=package.graph_paths or [],
            sufficiency=sufficiency_detail,
            temporal_validation=temporal_detail,
            validation_status=answer.validation_status or {},
            sources=answer.sources or package.sources or [],
            retrieval_rounds=package.retrieval_rounds,
            latency_ms=elapsed_ms,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
