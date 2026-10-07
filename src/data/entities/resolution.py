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


LEGAL_SUFFIXES = {
    "corporation",
    "corp",
    "incorporated",
    "inc",
    "company",
    "co",
    "limited",
    "ltd",
    "plc",
    "llc",
    "holdings",
    "group",
    "class a",
    "class b",
    "class c",
    "n a",
}

KNOWN_COMPANY_ALIASES = {
    "google": "GOOGL",
    "alphabet": "GOOGL",
    "facebook": "META",
    "meta": "META",
    "meta platforms": "META",
    "jpmorgan": "JPM",
    "jpmorganchase": "JPM",
    "jp morgan": "JPM",
    "jpmorgan chase": "JPM",
    "exxonmobil": "XOM",
    "exxon mobil": "XOM",
    "johnson johnson": "JNJ",
    "amazon com": "AMZN",
}


def strip_company_legal_suffix(name: str | None) -> str:
    """Strip common legal corporate suffixes for fuzzy canonical matching."""
    normalized = normalize_text(name)
    if not normalized:
        return ""

    tokens = normalized.split()
    while tokens:
        if len(tokens) >= 2 and f"{tokens[-2]} {tokens[-1]}" in LEGAL_SUFFIXES:
            tokens = tokens[:-2]
            continue
        if tokens[-1] in LEGAL_SUFFIXES:
            tokens = tokens[:-1]
            continue
        break

    return " ".join(tokens)


def build_entity_indexes(
    companies: list[dict[str, str]],
) -> dict[str, dict[str, dict[str, str]]]:
    """Build deterministic lookup indexes."""

    by_ticker: dict[str, dict[str, str]] = {}
    by_cik: dict[str, dict[str, str]] = {}
    by_name: dict[str, dict[str, str]] = {}

    for company in companies:
        ticker = company["ticker"]
        by_ticker[ticker] = company
        by_cik[company["cik"]] = company

        exact_name = normalize_text(company["company"])
        by_name[exact_name] = company

        stripped_name = strip_company_legal_suffix(company["company"])
        if stripped_name:
            by_name[stripped_name] = company

        by_name[ticker.lower()] = company

    # Map known canonical corporate aliases to universe records
    ticker_lookup = {c["ticker"]: c for c in companies}
    for alias, ticker in KNOWN_COMPANY_ALIASES.items():
        if ticker in ticker_lookup:
            by_name[alias] = ticker_lookup[ticker]

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
        3. Company name (exact, stripped legal suffix, or alias)
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

        # 1. Exact name or registered alias
        if normalized in indexes["name"]:
            entity = indexes["name"][normalized]

            return {
                **entity,
                "resolution_method": "company_name",
                "resolution_confidence": 1.0,
            }

        # 2. Name with legal corporate suffix stripped
        stripped = strip_company_legal_suffix(company)
        if stripped and stripped in indexes["name"]:
            entity = indexes["name"][stripped]

            return {
                **entity,
                "resolution_method": "company_name_normalized",
                "resolution_confidence": 0.95,
            }

        # 3. Check if company string is a ticker symbol
        ticker_candidate = company.strip().upper()
        if ticker_candidate in indexes["ticker"]:
            entity = indexes["ticker"][ticker_candidate]

            return {
                **entity,
                "resolution_method": "company_as_ticker",
                "resolution_confidence": 0.95,
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