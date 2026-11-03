"""
Evidence inspection read-only endpoints.
"""

from __future__ import annotations

import logging
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status
from qdrant_client.http import models

from api.dependencies import get_pipeline
from api.schemas.evidence import CorpusChunkItem, CorpusChunkResponse
from scripts.run_fingraph import FinGraphPipeline

logger = logging.getLogger("fingraph_api.routes.evidence")

router = APIRouter(prefix="/evidence", tags=["Evidence Inspector"])


@router.get(
    "/chunks",
    response_model=CorpusChunkResponse,
    summary="Browse SEC filings corpus chunks with metadata and provenance filtering",
)
def browse_corpus_chunks(
    company: Optional[str] = Query(None, description="Company ticker filter, e.g. 'MSFT', 'NVDA'"),
    document_id: Optional[str] = Query(None, description="Restrict to one filing, e.g. 'doc:sec:076412151b515b96'"),
    document_type: Optional[str] = Query(None, description="Filing type filter, e.g. '10-K', '10-Q', '8-K'"),
    limit: int = Query(25, ge=1, le=100),
    offset: Optional[str] = Query(None, description="Scroll offset ID"),
    pipeline: FinGraphPipeline = Depends(get_pipeline),
) -> CorpusChunkResponse:
    if not pipeline.vector_store or not pipeline.vector_store.client:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Vector store is not initialized.",
        )

    conditions = []
    if company:
        conditions.append(models.FieldCondition(key="company", match=models.MatchValue(value=company.upper())))
    if document_id:
        conditions.append(models.FieldCondition(key="document_id", match=models.MatchValue(value=document_id)))
    if document_type:
        conditions.append(models.FieldCondition(key="document_type", match=models.MatchValue(value=document_type.upper())))
    scroll_filter = models.Filter(must=conditions) if conditions else None

    try:
        points, next_offset = pipeline.vector_store.client.scroll(
            collection_name=pipeline.vector_store.collection_name,
            scroll_filter=scroll_filter,
            limit=limit,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )

        chunk_items: list[CorpusChunkItem] = []
        for p in points:
            payload = p.payload or {}
            chunk_items.append(
                CorpusChunkItem(
                    chunk_id=payload.get("chunk_id", str(p.id)),
                    document_id=payload.get("document_id", ""),
                    company=payload.get("company", ""),
                    document_type=payload.get("document_type"),
                    publication_date=payload.get("publication_date"),
                    section=payload.get("section"),
                    text=payload.get("text", ""),
                    source_url=payload.get("source_url"),
                    available_time=payload.get("available_time"),
                    event_time=payload.get("event_time"),
                )
            )

        total_matching = None
        try:
            total_matching = pipeline.vector_store.client.count(
                collection_name=pipeline.vector_store.collection_name,
                count_filter=scroll_filter,
                exact=True,
            ).count
        except Exception as count_exc:  # count is informational only
            logger.warning("Could not count matching chunks: %s", count_exc)

        return CorpusChunkResponse(
            total_returned=len(chunk_items),
            total_matching=total_matching,
            offset=str(next_offset) if next_offset else None,
            chunks=chunk_items,
        )
    except Exception as exc:
        logger.exception("Error scrolling vector store chunks: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to scroll vector store chunks: {exc}",
        ) from exc
