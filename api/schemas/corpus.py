"""
Pydantic schemas for the read-only corpus manifest endpoint.
"""

from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field


class FilingItem(BaseModel):
    document_id: str
    company: str
    document_type: Optional[str] = None
    title: Optional[str] = None
    publication_date: Optional[str] = None
    fiscal_period: Optional[str] = None
    available_time: Optional[str] = None
    source_url: Optional[str] = None
    source_reference: Optional[str] = None
    chunk_count: Optional[int] = None


class CompanyCoverage(BaseModel):
    company: str
    filings: int
    chunks: int
    by_type: dict[str, int] = Field(default_factory=dict)
    first_publication_date: Optional[str] = None
    last_publication_date: Optional[str] = None


class CorpusFilingsResponse(BaseModel):
    total_filings: int
    total_chunks: int
    by_type: dict[str, int] = Field(default_factory=dict)
    companies: list[CompanyCoverage]
    filings: list[FilingItem]
    source: str = "data/processed/documents/*.json + data/processed/sec_corpus_manifest.json"
