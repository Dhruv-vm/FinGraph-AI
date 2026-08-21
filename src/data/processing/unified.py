from __future__ import annotations

from datetime import datetime
from typing import Any


def parse_datetime(value: str | None) -> datetime | None:
    """Parse an ISO date or datetime."""

    if not value:
        return None

    try:
        return datetime.fromisoformat(
            value.replace("Z", "+00:00")
        )
    except ValueError:
        return None


def build_market_event(record: dict[str, Any]) -> dict[str, Any]:
    """Convert a market record into a unified temporal event."""

    return {
        "event_type": "market",
        "ticker": record["ticker"],
        "event_time": record["date"],
        "available_time": record["date"],
        "source": "market",
        "data": record,
    }


def build_news_event(record: dict[str, Any]) -> dict[str, Any]:
    """Convert a news article into a unified temporal event."""

    return {
        "event_type": "news",
        "ticker": record["ticker"],
        "event_time": record["published_at"],
        "available_time": record["published_at"],
        "source": record.get("source", "news"),
        "data": record,
    }


def build_sec_event(record: dict[str, Any]) -> dict[str, Any]:
    """Convert an SEC filing into a unified temporal event."""

    return {
        "event_type": "sec_filing",
        "ticker": record.get("ticker"),
        "event_time": record.get("filing_date"),
        "available_time": record.get("acceptance_datetime"),
        "source": "sec",
        "data": record,
    }


def build_macro_event(record: dict[str, Any]) -> dict[str, Any]:
    """Convert a macroeconomic observation into a unified temporal event."""

    return {
        "event_type": "macro",
        "ticker": record.get("ticker"),
        "event_time": record["date"],
        "available_time": record["date"],
        "source": record.get("source", "macro"),
        "data": record,
    }


def build_unified_events(
    market: list[dict[str, Any]] | None = None,
    news: list[dict[str, Any]] | None = None,
    sec: list[dict[str, Any]] | None = None,
    macro: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """
    Convert all source-specific records into one temporal event format.
    """

    events: list[dict[str, Any]] = []

    for record in market or []:
        events.append(build_market_event(record))

    for record in news or []:
        events.append(build_news_event(record))

    for record in sec or []:
        events.append(build_sec_event(record))

    for record in macro or []:
        events.append(build_macro_event(record))

    events.sort(
        key=lambda event: event.get("available_time") or ""
    )

    return events
