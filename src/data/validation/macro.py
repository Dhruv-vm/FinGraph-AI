from __future__ import annotations

from datetime import datetime
from typing import Any


def parse_date(value: str | None) -> datetime | None:
    """Parse an ISO date or datetime."""

    if not value:
        return None

    try:
        return datetime.fromisoformat(
            value.replace("Z", "+00:00")
        )
    except ValueError:
        return None


def validate_macro_record(
    record: dict[str, Any],
) -> dict[str, Any]:
    """Validate a normalized macro observation."""

    errors: list[str] = []

    if not record.get("series_id"):
        errors.append("missing_series_id")

    if not record.get("series_name"):
        errors.append("missing_series_name")

    if not record.get("date"):
        errors.append("missing_date")
    elif parse_date(record.get("date")) is None:
        errors.append("invalid_date")

    if record.get("value") is None:
        errors.append("missing_value")
    else:
        try:
            float(record["value"])
        except (TypeError, ValueError):
            errors.append("invalid_value")

    if not record.get("source"):
        errors.append("missing_source")

    if not record.get("source_url"):
        errors.append("missing_source_url")

    if not record.get("ingested_at"):
        errors.append("missing_ingested_at")
    elif parse_date(record.get("ingested_at")) is None:
        errors.append("invalid_ingested_at")

    return {
        "macro_valid": len(errors) == 0,
        "macro_errors": errors,
    }


def validate_macro(
    records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Validate a collection of macro observations."""

    validated: list[dict[str, Any]] = []

    stats = {
        "total": len(records),
        "valid": 0,
        "invalid": 0,
    }

    seen: set[tuple[str, str]] = set()
    duplicates = 0

    for record in records:
        key = (
            record.get("series_id", ""),
            record.get("date", ""),
        )

        if key in seen:
            duplicates += 1

        seen.add(key)

        result = validate_macro_record(record)

        enriched = {
            **record,
            **result,
        }

        validated.append(enriched)

        if result["macro_valid"]:
            stats["valid"] += 1
        else:
            stats["invalid"] += 1

    stats["duplicates"] = duplicates

    return validated, stats
