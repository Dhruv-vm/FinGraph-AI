from __future__ import annotations

from datetime import datetime

from src.data.documents.schema import Document, DocumentChunk


def parse_datetime(value: str | None) -> datetime | None:
    """Parse an ISO-8601 timestamp."""

    if not value:
        return None

    try:
        return datetime.fromisoformat(
            value.replace("Z", "+00:00")
        )
    except ValueError:
        return None


def is_available_at(
    item: Document | DocumentChunk,
    reference_time: str,
) -> bool:
    """
    Determine whether evidence is admissible at a question's
    reference time.

    Unknown availability is rejected conservatively.
    """

    available_time = parse_datetime(item.available_time)
    reference = parse_datetime(reference_time)

    if available_time is None or reference is None:
        return False

    return available_time <= reference
