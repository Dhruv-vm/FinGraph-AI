from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Document:
    """
    Canonical source document used by the FinGraph AI pipeline.

    Temporal fields are deliberately kept separate:
    - event_time: when the underlying event occurred, if known
    - publication_date: when the source published the information
    - available_time: when the information became publicly available
    - ingested_at: when FinGraph obtained the source
    - fiscal_period: reporting period, when applicable
    """

    document_id: str
    company: str | None
    source: str
    document_type: str

    title: str | None = None
    text: str = ""

    publication_date: str | None = None
    fiscal_period: str | None = None
    section: str | None = None

    event_time: str | None = None
    available_time: str | None = None
    ingested_at: str | None = None

    source_url: str | None = None
    source_reference: str | None = None

    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the document into a JSON-compatible dictionary."""
        return asdict(self)


@dataclass
class DocumentChunk:
    """
    A retrievable chunk derived from a canonical Document.

    Chunk-level temporal and provenance metadata is retained so that
    retrieval can enforce point-in-time constraints without losing
    source traceability.
    """

    chunk_id: str
    document_id: str
    text: str

    chunk_index: int = 0
    section: str | None = None

    company: str | None = None
    source: str | None = None
    document_type: str | None = None

    publication_date: str | None = None
    fiscal_period: str | None = None
    event_time: str | None = None
    available_time: str | None = None
    ingested_at: str | None = None

    source_url: str | None = None
    source_reference: str | None = None

    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the chunk into a JSON-compatible dictionary."""
        return asdict(self)
