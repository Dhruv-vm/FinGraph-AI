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


def validate_filing_temporality(
    filing: dict[str, Any],
) -> dict[str, Any]:
    """
    Validate whether a filing has sufficient temporal metadata.

    Filing date represents the SEC filing date.
    Acceptance datetime provides the precise SEC timestamp
    used later for information-availability checks.
    """

    filing_date = parse_date(
        filing.get("filing_date")
    )

    report_date = parse_date(
        filing.get("report_date")
    )

    acceptance_datetime = parse_date(
        filing.get("acceptance_datetime")
    )

    errors: list[str] = []

    if filing_date is None:
        errors.append("missing_filing_date")

    if acceptance_datetime is None:
        errors.append("missing_acceptance_datetime")

    if report_date is None:
        errors.append("missing_report_date")

    if (
        filing_date
        and report_date
        and report_date > filing_date
    ):
        errors.append(
            "report_date_after_filing_date"
        )

    return {
        "temporal_valid": len(errors) == 0,
        "temporal_errors": errors,
    }

def validate_filings(
    filings: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Validate temporal metadata for a collection of filings."""

    validated: list[dict[str, Any]] = []

    stats = {
        "total": len(filings),
        "valid": 0,
        "invalid": 0,
    }

    for filing in filings:
        result = validate_filing_temporality(filing)

        enriched = {
            **filing,
            **result,
        }

        validated.append(enriched)

        if result["temporal_valid"]:
            stats["valid"] += 1
        else:
            stats["invalid"] += 1

    return validated, stats
