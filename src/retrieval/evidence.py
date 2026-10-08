from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class GraphEvidence:
    """A retrievable path or relational assertion extracted from the Knowledge Graph."""

    evidence_id: str
    source_node: str
    target_node: str
    relationship: str
    path: list[str]
    hop_count: int
    text: str
    available_time: str | None = None
    event_time: str | None = None
    confidence: float = 1.0
    document_id: str | None = None
    chunk_id: str | None = None
    source: str | None = None
    source_url: str | None = None
    source_reference: str | None = None
    publication_date: str | None = None
    fiscal_period: str | None = None
    properties: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RetrievalEvidence:
    """A unified evidence representation across vector, graph, and hybrid retrieval."""

    evidence_id: str
    source_type: str  # "vector", "graph", or "hybrid"
    score: float
    rank: int
    text: str
    document_id: str | None = None
    chunk_id: str | None = None
    available_time: str | None = None
    event_time: str | None = None
    publication_date: str | None = None
    source_url: str | None = None
    source_reference: str | None = None
    retrieval_sources: list[str] = field(default_factory=list)
    original_scores: dict[str, float] = field(default_factory=dict)
    original_ranks: dict[str, int] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
