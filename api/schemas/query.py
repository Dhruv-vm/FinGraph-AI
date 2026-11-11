"""
Pydantic schemas for Query endpoints.
"""

from __future__ import annotations

from typing import Any, Literal, Optional
from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=2,
        max_length=1000,
        description="Natural language financial question to answer using SEC corpus and Knowledge Graph.",
        examples=["Did Microsoft acquire Activision Blizzard?", "What risks affected NVIDIA as of June 1, 2025?"],
    )
    reference_time: Optional[str] = Field(
        default=None,
        description="Optional point-in-time ISO 8601 temporal reference date/time cutoff. Zero evidence post-dating this time will reach generation.",
        examples=["2025-05-01T00:00:00+00:00", "2025-06-01T00:00:00+00:00"],
    )
    top_k: Optional[int] = Field(
        default=10,
        ge=1,
        le=50,
        description="Target number of evidence items to consolidate into context window.",
    )
    candidate_pool_size: Optional[int] = Field(
        default=20,
        ge=5,
        le=100,
        description="Candidate pool size entering deterministic reranker.",
    )
    mode: Literal["benchmark", "assist"] = Field(
        default="benchmark",
        description=(
            "'benchmark' runs the evaluated Phase 7 configuration. 'assist' (live use) reads longer passages, "
            "accepts relationships evidenced in filing text, and may give a partial, cited answer instead of refusing. "
            "Both modes answer only from retrieved SEC evidence."
        ),
    )
    allow_general: bool = Field(
        default=True,
        description=(
            "Assist mode only: when the filings cannot answer, reply from the model's general knowledge, "
            "labelled answer_source='general' with no citations. Never used in benchmark mode or with a reference_time."
        ),
    )


class EvidenceItem(BaseModel):
    evidence_id: str
    source_type: str = Field(description="'hybrid', 'graph', or 'vector'")
    score: float
    rank: int
    text: str
    document_id: Optional[str] = None
    chunk_id: Optional[str] = None
    available_time: Optional[str] = None
    event_time: Optional[str] = None
    publication_date: Optional[str] = None
    source_url: Optional[str] = None
    source_reference: Optional[str] = None
    source_company: Optional[str] = None
    provenance_verified: bool = True
    attribution_status: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class CitationItem(BaseModel):
    evidence_id: str
    document_id: Optional[str] = None
    chunk_id: Optional[str] = None
    source_type: Optional[str] = None
    available_time: Optional[str] = None
    text_snippet: Optional[str] = None


class SufficiencyDetail(BaseModel):
    sufficient: bool
    reason: str
    missing_entities: list[str] = Field(default_factory=list)
    missing_relationships: list[str] = Field(default_factory=list)
    required_hops: int = 1
    observed_hops: int = 1
    evidence_count: int = 0
    temporal_valid: bool = True
    provenance_complete: bool = True


class TemporalValidationDetail(BaseModel):
    temporal_valid: bool
    reference_time: Optional[str] = None
    context_items_count: int = 0
    future_items_count: int = 0
    feir: float = 0.0


class QueryResponse(BaseModel):
    query: str
    mode: str = Field(default="benchmark", description="Pipeline mode used for this answer.")
    answer_source: str = Field(
        default="filings",
        description="'filings' (grounded in retrieved SEC evidence), 'general' (model general knowledge, unverified), or 'none' (refused).",
    )
    filings_finding: Optional[str] = Field(
        default=None, description="When answer_source is 'general': what the SEC filing search concluded."
    )
    decision: str = Field(description="'answer' or 'refuse'")
    answer: str
    insufficient_evidence: bool
    confidence: float
    reasoning_summary: Optional[str] = None
    citations: list[CitationItem] = Field(default_factory=list)
    evidence: list[EvidenceItem] = Field(default_factory=list)
    graph_paths: list[Any] = Field(default_factory=list)
    sufficiency: SufficiencyDetail
    temporal_validation: TemporalValidationDetail
    validation_status: dict[str, Any] = Field(default_factory=dict)
    sources: list[str] = Field(default_factory=list)
    retrieval_rounds: int = 1
    latency_ms: float
    timestamp: str
