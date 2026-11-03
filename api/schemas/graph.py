"""
Pydantic schemas for Knowledge Graph Explorer endpoints.
"""

from __future__ import annotations

from typing import Any, Optional
from pydantic import BaseModel, Field


class GraphNodeItem(BaseModel):
    id: str
    label: str
    type: str
    properties: dict[str, Any] = Field(default_factory=dict)
    degree: int = 0


class GraphEdgeItem(BaseModel):
    id: str
    source: str
    target: str
    relationship: str
    available_time: Optional[str] = None
    event_time: Optional[str] = None
    confidence: Optional[float] = None
    document_id: Optional[str] = None
    chunk_id: Optional[str] = None
    source_url: Optional[str] = None
    source_reference: Optional[str] = None
    publication_date: Optional[str] = None
    properties: dict[str, Any] = Field(default_factory=dict)


class SubgraphResponse(BaseModel):
    center_node: str
    node_count: int
    edge_count: int
    nodes: list[GraphNodeItem]
    edges: list[GraphEdgeItem]
    total_direct_edges: Optional[int] = None
    total_connected_companies: Optional[int] = None
    total_matching: Optional[int] = None
    has_more: bool = False
    limit: Optional[int] = None
    offset: int = 0
    filter_type: Optional[str] = "companies"


class CompanyListItem(BaseModel):
    node_id: str
    ticker: str
    canonical_name: str
    edge_count: int


class CompanyListResponse(BaseModel):
    total_companies: int
    companies: list[CompanyListItem]
