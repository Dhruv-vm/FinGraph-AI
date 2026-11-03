"""
Pydantic schemas for Evidence Inspector endpoints.
"""

from __future__ import annotations

from typing import Any, Optional
from pydantic import BaseModel, Field


class CorpusChunkItem(BaseModel):
    chunk_id: str
    document_id: str
    company: str
    document_type: Optional[str] = None
    publication_date: Optional[str] = None
    section: Optional[str] = None
    text: str
    source_url: Optional[str] = None
    available_time: Optional[str] = None
    event_time: Optional[str] = None


class CorpusChunkResponse(BaseModel):
    total_returned: int
    total_matching: Optional[int] = None
    offset: Optional[str] = None
    chunks: list[CorpusChunkItem]
