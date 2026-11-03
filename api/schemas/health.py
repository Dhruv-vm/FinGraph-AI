"""
Pydantic schemas for Health check endpoints.
"""

from __future__ import annotations

from typing import Any, Optional
from pydantic import BaseModel, Field


class ComponentHealth(BaseModel):
    status: str = Field(description="'healthy', 'degraded', or 'unavailable'")
    message: Optional[str] = None
    details: dict[str, Any] = Field(default_factory=dict)


class HealthResponse(BaseModel):
    status: str = Field(description="'healthy', 'degraded', or 'unavailable'")
    version: str
    timestamp: str
    pipeline: ComponentHealth
    qdrant: ComponentHealth
    knowledge_graph: ComponentHealth
    llm: ComponentHealth
