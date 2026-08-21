from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Any


DEFAULT_COMPANY_UNIVERSE = Path(
    "configs/company_universe.csv"
)


def normalize_text(value: str | None) -> str:
    """Normalize text for deterministic entity matching."""

    if not value:
        return ""

    value = value.strip().lower()

    value = re.sub(
        r"[^a-z0-9]+",
        " ",
        value,
    )

    return " ".join(value.split())


def normalize_cik(value: str | None) -> str:
    """Normalize an SEC CIK to its zero-padded form."""

    if not value:
        return ""

    digits = re.sub(r"\D", "", value)

    return digits.zfill(10)


def load_company_universe(
    path: str | Path = DEFAULT_COMPANY_UNIVERSE,
) -> list[dict[str, str]]:
    """Load the canonical company universe."""

    path = Path(path)

    with path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        reader = csv.DictReader(handle)

        records = []

        for row in reader:
            ticker = (row.get("ticker") or "").strip().upper()
            company = (row.get("company") or "").strip()
            sector = (row.get("sector") or "").strip()
            cik = normalize_cik(row.get("cik"))

            if not ticker or not company or not cik:
                continue

            records.append(
                {
                    "entity_id": f"company:{ticker}",
                    "ticker": ticker,
                    "company": company,
                    "sector": sector,
                    "cik": cik,
                }
            )

    return records


def build_entity_indexes(
    companies: list[dict[str, str]],
) -> dict[str, dict[str, dict[str, str]]]:
    """Build deterministic lookup indexes."""

    by_ticker: dict[str, dict[str, str]] = {}
    by_cik: dict[str, dict[str, str]] = {}
    by_name: dict[str, dict[str, str]] = {}

    for company in companies:
        by_ticker[company["ticker"]] = company
        by_cik[company["cik"]] = company
        by_name[normalize_text(company["company"])] = company

    return {
        "ticker": by_ticker,
        "cik": by_cik,
        "name": by_name,
    }


def resolve_company(
    *,
    ticker: str | None = None,
    cik: str | None = None,
    company: str | None = None,
    indexes: dict[str, dict[str, dict[str, str]]],
) -> dict[str, Any] | None:
    """
    Resolve a company using deterministic identifiers.

    Resolution priority:
        1. CIK
        2. Ticker
        3. Company name
    """

    if cik:
        normalized = normalize_cik(cik)

        if normalized in indexes["cik"]:
            entity = indexes["cik"][normalized]

            return {
                **entity,
                "resolution_method": "cik",
                "resolution_confidence": 1.0,
            }

    if ticker:
        normalized = ticker.strip().upper()

        if normalized in indexes["ticker"]:
            entity = indexes["ticker"][normalized]

            return {
                **entity,
                "resolution_method": "ticker",
                "resolution_confidence": 1.0,
            }

    if company:
        normalized = normalize_text(company)

        if normalized in indexes["name"]:
            entity = indexes["name"][normalized]

            return {
                **entity,
                "resolution_method": "company_name",
                "resolution_confidence": 1.0,
            }

    return None


def resolve_event_company(
    event: dict[str, Any],
    indexes: dict[str, dict[str, dict[str, str]]],
) -> dict[str, Any]:
    """Resolve the company associated with a unified event."""

    data = event.get("data", {})

    resolved = resolve_company(
        ticker=event.get("ticker") or data.get("ticker"),
        cik=data.get("cik"),
        company=data.get("company"),
        indexes=indexes,
    )

    if resolved is None:
        return {
            **event,
            "entity_id": None,
            "entity_resolution_method": None,
            "entity_resolution_confidence": 0.0,
            "entity_resolved": False,
        }

    return {
        **event,
        "entity_id": resolved["entity_id"],
        "entity_resolution_method": resolved[
            "resolution_method"
        ],
        "entity_resolution_confidence": resolved[
            "resolution_confidence"
        ],
        "entity_resolved": True,
    }