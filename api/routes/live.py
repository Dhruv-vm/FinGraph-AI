"""
Live SEC filings: fetch a company's latest filings from EDGAR on demand.

Used by the Ask page (Assist mode) when the indexed filings do not answer a question.
Fetched filings are tagged as "live" and never enter benchmark-mode retrieval.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field

from api.dependencies import get_pipeline, get_pipeline_lock
from scripts.run_fingraph import FinGraphPipeline
from src.data.ingestion import live_sec

logger = logging.getLogger("fingraph_api.routes.live")

router = APIRouter(prefix="/live", tags=["Live filings"])

# Issuers in the controlled research corpus.
CORPUS_TICKERS = {"AAPL", "AMZN", "GOOGL", "JNJ", "JPM", "META", "MSFT", "NVDA", "TSLA", "XOM"}


class LiveFetchRequest(BaseModel):
    query: Optional[str] = Field(default=None, description="Question text; the company is resolved from it.")
    ticker: Optional[str] = Field(default=None, description="Explicit ticker (takes precedence over query).")


def _company_payload(company: dict[str, Any] | None) -> dict[str, Any] | None:
    if not company:
        return None
    t = company["ticker"]
    live = [f for f in live_sec._load_manifest().get("filings", []) if f["ticker"] == t]
    return {
        **company,
        "in_research_corpus": t in CORPUS_TICKERS,
        "live_filings": len(live),
    }


@router.get("/resolve", summary="Find the SEC registrant a question refers to")
async def resolve(q: str = Query(..., min_length=2, max_length=1000)) -> dict[str, Any]:
    company = await asyncio.to_thread(live_sec.resolve_company, q)
    return {"company": _company_payload(company)}


@router.get("/filings", summary="Filings fetched on demand (outside the research corpus)")
async def list_live_filings() -> dict[str, Any]:
    return live_sec._load_manifest()


@router.post("/fetch", summary="Fetch a company's latest 10-K, 10-Q and 8-K filings from SEC EDGAR")
async def fetch(
    request: LiveFetchRequest,
    pipeline: FinGraphPipeline = Depends(get_pipeline),
    lock: asyncio.Lock = Depends(get_pipeline_lock),
) -> dict[str, Any]:
    if request.ticker:
        idx = await asyncio.to_thread(live_sec._load_tickers)
        company = idx["by_ticker"].get(request.ticker.strip().upper())
    elif request.query:
        company = await asyncio.to_thread(live_sec.resolve_company, request.query)
    else:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Provide a query or a ticker.")
    if not company:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No SEC-registered company was recognised in the question.")
    if pipeline.vector_store is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Vector store is not available.")
    try:
        # Writes to the local vector index must not interleave with queries.
        async with lock:
            result = await asyncio.to_thread(live_sec.fetch_live_filings, company, pipeline.vector_store)
            pipeline.vector_store.clear_period_cache()
    except Exception as exc:
        logger.exception("Live fetch failed: %s", exc)
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=f"SEC EDGAR fetch failed: {exc}") from exc
    result["company"] = _company_payload(company)
    return result
