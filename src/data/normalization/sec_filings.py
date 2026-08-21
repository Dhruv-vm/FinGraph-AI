from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any


RAW_SEC_DIR = Path("data/raw/sec")
PROCESSED_SEC_DIR = Path("data/processed/sec")


TARGET_FORMS = {
    "10-K",
    "10-Q",
    "8-K",
}


def parse_datetime(value: str | None) -> datetime | None:
    """Parse SEC ISO timestamps into timezone-aware datetimes."""

    if not value:
        return None

    try:
        return datetime.fromisoformat(
            value.replace("Z", "+00:00")
        )
    except ValueError:
        return None


def normalize_filing(
    ticker: str,
    cik: str,
    record: dict[str, Any],
) -> dict[str, Any]:
    """Convert one SEC filing record into FinGraph's normalized schema."""

    acceptance_time = parse_datetime(
        record.get("acceptanceDateTime")
    )

    return {
        "ticker": ticker.upper(),
        "cik": cik.zfill(10),
        "form": record.get("form"),
        "filing_date": record.get("filingDate"),
        "report_date": record.get("reportDate"),
        "acceptance_datetime": (
            acceptance_time.isoformat()
            if acceptance_time
            else None
        ),
        "accession_number": record.get("accessionNumber"),
        "primary_document": record.get("primaryDocument"),
        "primary_doc_description": record.get(
            "primaryDocDescription"
        ),
        "act": record.get("act"),
        "file_number": record.get("fileNumber"),
        "items": record.get("items"),
        "is_xbrl": record.get("isXBRL"),
        "is_inline_xbrl": record.get("isInlineXBRL"),
    }


def load_recent_filings(
    ticker: str,
    cik: str,
) -> list[dict[str, Any]]:
    """Load and normalize recent SEC filings for one company."""

    ticker = ticker.upper()

    path = RAW_SEC_DIR / f"{ticker.lower()}_submissions.json"

    if not path.exists():
        raise FileNotFoundError(
            f"SEC raw file not found: {path}"
        )

    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    recent = data.get("filings", {}).get("recent", {})

    if not recent:
        return []

    forms = recent.get("form", [])

    records: list[dict[str, Any]] = []

    for index, form in enumerate(forms):
        record = {
            key: values[index]
            for key, values in recent.items()
            if index < len(values)
        }

        if form not in TARGET_FORMS:
            continue

        records.append(
            normalize_filing(
                ticker=ticker,
                cik=cik,
                record=record,
            )
        )

    return records


def save_json(
    records: list[dict[str, Any]],
    ticker: str,
) -> Path:
    """Save normalized records as JSON."""

    PROCESSED_SEC_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output = (
        PROCESSED_SEC_DIR
        / f"{ticker.lower()}_filings.json"
    )

    with output.open("w", encoding="utf-8") as file:
        json.dump(
            records,
            file,
            indent=2,
            ensure_ascii=False,
        )

    return output


def normalize_company(
    ticker: str,
    cik: str,
) -> Path:
    """Normalize one company's SEC submissions."""

    records = load_recent_filings(
        ticker=ticker,
        cik=cik,
    )

    return save_json(
        records=records,
        ticker=ticker,
    )


if __name__ == "__main__":
    output = normalize_company(
        ticker="AAPL",
        cik="0000320193",
    )

    print(f"Saved: {output}")
