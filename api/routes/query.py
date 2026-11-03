"""
Query endpoints for FinGraph AI API.
"""

from __future__ import annotations

import logging
from fastapi import APIRouter, Depends, HTTPException, status

from api.dependencies import get_pipeline_service
from api.schemas.query import QueryRequest, QueryResponse
from api.services.pipeline_service import PipelineService

logger = logging.getLogger("fingraph_api.routes.query")

router = APIRouter(prefix="/query", tags=["Query"])


@router.post(
    "",
    response_model=QueryResponse,
    status_code=status.HTTP_200_OK,
    summary="Answer a financial question using SEC filings and Temporal KG",
    description=(
        "Executes the full FinGraph AI GraphRAG pipeline: query analysis, hybrid retrieval "
        "with entity-balanced RRF fusion, bounded agentic graph exploration, temporal filtering, "
        "evidence sufficiency gating, and grounded answer generation with verified citations."
    ),
)
async def query_fingraph(
    request: QueryRequest,
    service: PipelineService = Depends(get_pipeline_service),
) -> QueryResponse:
    try:
        response = await service.execute_query(request)
        return response
    except ValueError as ve:
        logger.warning("Query validation error: %s", ve)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(ve),
        ) from ve
    except Exception as exc:
        logger.exception("Error executing query: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while executing the query pipeline: {exc}",
        ) from exc
