from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from yfinance import ticker


PROJECT_ROOT = Path(__file__).resolve().parents[3]
COMPANY_UNIVERSE = PROJECT_ROOT / "configs" / "company_universe.csv"


def load_company_universe(
    path: Path = COMPANY_UNIVERSE,
) -> list[dict[str, str]]:
    """Load the configured company universe."""

    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)

        companies = []

        for row in reader:
            companies.append(
                {
                    "ticker": row["ticker"].strip(),
                    "company": row["company"].strip(),
                    "sector": row["sector"].strip(),
                    "cik": row["cik"].strip().zfill(10),
                }
            )

    return companies


def get_company(
    ticker: str,
    companies: list[dict[str, str]],
) -> dict[str, str]:
    """Return a company configuration by ticker."""

    ticker = ticker.upper()

    for company in companies:
        if company["ticker"] == ticker:
            return company

    raise ValueError(
        f"Ticker '{ticker}' not found in company universe."
    )


def build_company_output_paths(
    ticker: str,
) -> dict[str, Path]:
    """Build standard processed-data paths for a company."""

    base = PROJECT_ROOT / "data" / "processed"

    return {
        "market": base / "market" / f"{ticker.lower()}_market.json",
        "news": base / "news" / f"{ticker.lower()}_news.json",
        "sec": base / "sec" / f"{ticker.lower()}_filings.json",
    }


def validate_company_universe(
    companies: list[dict[str, str]],
) -> dict[str, Any]:
    """Validate the company universe configuration."""

    tickers = [company["ticker"] for company in companies]

    duplicate_tickers = sorted(
        {
            ticker
            for ticker in tickers
            if tickers.count(ticker) > 1
        }
    )

    missing_fields = []

    required_fields = {
        "ticker",
        "company",
        "sector",
        "cik",
    }

    for company in companies:
        missing = sorted(
            field
            for field in required_fields
            if not company.get(field)
        )

        if missing:
            missing_fields.append(
                {
                    "ticker": company.get("ticker", ""),
                    "fields": missing,
                }
            )

    return {
        "total_companies": len(companies),
        "unique_tickers": len(set(tickers)),
        "duplicate_tickers": duplicate_tickers,
        "missing_fields": missing_fields,
        "valid": (
            len(companies) > 0
            and not duplicate_tickers
            and not missing_fields
        ),
    }
from data.ingestion import sec
from src.data.ingestion.market import ingest_market_data
from src.data.ingestion.news import ingest_news
from src.data.ingestion.sec import ingest_company
from src.data.storage.local import save_json

def ingest_company_data(
    company: dict[str, str],
    start: str = "2025-01-01",
) -> dict[str, Any]:
    """
    Ingest all company-level datasets.

    Covers:
    - Market data
    - Financial news
    - SEC filings
    """

    ticker = company["ticker"]
    cik = company["cik"]

    paths = build_company_output_paths(ticker)

    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 60}")
    print(f"Ingesting {ticker} | {company['company']}")
    print(f"{'=' * 60}")

    # -------------------------
    # Market
    # -------------------------

    print("→ Market data")

    market = ingest_market_data(
        ticker,
        start=start,
    )

    save_json(market, paths["market"])

    print(f"  Saved: {len(market)} market records")

    # -------------------------
    # News
    # -------------------------

    print("→ News")

    news = ingest_news(
        ticker,
        query=ticker,
    )

    save_json(news, paths["news"])

    print(f"  Saved: {len(news)} news records")

    # -------------------------
    # SEC
    # -------------------------

    print("→ SEC filings")

    sec_result = ingest_company(
        ticker,
        cik,
    )

    # ingest_company() already saves SEC data
    # and returns the output path.
    if isinstance(sec_result, (str, Path)):
        from src.data.storage.local import load_json

        sec = load_json(sec_result)
    else:
        sec = sec_result

    save_json(sec, paths["sec"])

    print(f"  Saved: {len(sec)} SEC records")

    return {
        "ticker": ticker,
        "company": company["company"],
        "sector": company["sector"],
        "market_records": len(market),
        "news_records": len(news),
        "sec_records": len(sec),
        "paths": {
            key: str(value)
            for key, value in paths.items()
        },
    }

def ingest_companies(
    tickers: list[str] | None = None,
    start: str = "2025-01-01",
) -> list[dict[str, Any]]:
    """
    Ingest data for multiple companies.

    If tickers is None, the complete configured
    company universe is used.
    """

    companies = load_company_universe()

    validation = validate_company_universe(companies)

    if not validation["valid"]:
        raise ValueError(
            f"Invalid company universe: {validation}"
        )

    if tickers is not None:
        requested = {
            ticker.upper()
            for ticker in tickers
        }

        companies = [
            company
            for company in companies
            if company["ticker"] in requested
        ]

        found = {
            company["ticker"]
            for company in companies
        }

        missing = sorted(requested - found)

        if missing:
            raise ValueError(
                f"Tickers not found in company universe: {missing}"
            )

    results = []

    for company in companies:
        result = ingest_company_data(
            company,
            start=start,
        )

        results.append(result)

    return results