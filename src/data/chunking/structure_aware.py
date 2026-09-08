from __future__ import annotations

import re

from src.data.documents.schema import Document, DocumentChunk


def _normalize_text(text: str) -> str:
    """Normalize whitespace while preserving paragraph boundaries."""

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def _split_paragraphs(text: str) -> list[str]:
    """Split normalized text into paragraph-level units."""

    return [
        paragraph.strip()
        for paragraph in re.split(r"\n\s*\n", text)
        if paragraph.strip()
    ]


def _make_chunk(
    document: Document,
    text: str,
    index: int,
) -> DocumentChunk:
    """Create a chunk while preserving document metadata."""

    return DocumentChunk(
        chunk_id=f"{document.document_id}:chunk:{index}",
        document_id=document.document_id,
        text=text.strip(),
        chunk_index=index,
        section=document.section,
        company=document.company,
        source=document.source,
        document_type=document.document_type,
        publication_date=document.publication_date,
        fiscal_period=document.fiscal_period,
        event_time=document.event_time,
        available_time=document.available_time,
        ingested_at=document.ingested_at,
        source_url=document.source_url,
        source_reference=document.source_reference,
        metadata=dict(document.metadata),
    )


def _split_long_paragraph(
    paragraph: str,
    max_chars: int,
    overlap_chars: int,
) -> list[str]:
    """Split an oversized paragraph using deterministic overlapping windows."""

    chunks: list[str] = []

    start = 0

    while start < len(paragraph):
        end = min(start + max_chars, len(paragraph))
        chunks.append(paragraph[start:end].strip())

        if end >= len(paragraph):
            break

        start = end - overlap_chars

    return chunks


def chunk_document(
    document: Document,
    max_chars: int = 1800,
    overlap_chars: int = 250,
) -> list[DocumentChunk]:
    """
    Create structure-aware chunks from a canonical document.

    Paragraph boundaries are preferred. Oversized paragraphs are split
    using deterministic character windows with bounded overlap.
    """

    if max_chars <= 0:
        raise ValueError("max_chars must be greater than zero")

    if overlap_chars < 0:
        raise ValueError("overlap_chars cannot be negative")

    if overlap_chars >= max_chars:
        raise ValueError(
            "overlap_chars must be smaller than max_chars"
        )

    text = _normalize_text(document.text)

    if not text:
        return []

    paragraphs = _split_paragraphs(text)

    chunks: list[DocumentChunk] = []
    current_parts: list[str] = []
    current_length = 0

    def flush() -> None:
        nonlocal current_parts, current_length

        if not current_parts:
            return

        chunk_text = "\n\n".join(current_parts)

        chunks.append(
            _make_chunk(
                document=document,
                text=chunk_text,
                index=len(chunks),
            )
        )

        current_parts = []
        current_length = 0

    for paragraph in paragraphs:
        if len(paragraph) > max_chars:
            flush()

            for window in _split_long_paragraph(
                paragraph,
                max_chars=max_chars,
                overlap_chars=overlap_chars,
            ):
                chunks.append(
                    _make_chunk(
                        document=document,
                        text=window,
                        index=len(chunks),
                    )
                )

            continue

        separator_length = 2 if current_parts else 0
        required_length = current_length + separator_length + len(paragraph)

        if current_parts and required_length > max_chars:
            previous_text = "\n\n".join(current_parts)
            flush()

            if overlap_chars:
                overlap = previous_text[-overlap_chars:].strip()

                if overlap:
                    current_parts = [overlap]
                    current_length = len(overlap)

        separator_length = 2 if current_parts else 0

        if current_length + separator_length + len(paragraph) > max_chars:
            flush()

        current_parts.append(paragraph)
        current_length = (
            current_length
            + (2 if len(current_parts) > 1 else 0)
            + len(paragraph)
        )

    flush()

    return chunks
