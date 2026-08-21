from __future__ import annotations

from datetime import datetime, timezone

import yfinance as yf

from src.data.storage.local import save_json


MACRO_SERIES = {
    "^TNX": {
        "series_id": "DGS10",
        "series_name": "10-Year Treasury Yield",
        "unit": "percent",
    },
    "^IRX": {
        "series_id": "DGS3MO",
        "series_name": "3-Month Treasury Yield",
        "unit": "percent",
    },
}


def fetch_macro_market_data(
    ticker: str,
    start: str = "2025-01-01",
) -> list[dict]:
    """Fetch a market-observable macro series using Yahoo Finance."""

    data = yf.download(
        ticker,
        start=start,
        progress=False,
        auto_adjust=False,
    )

    if data.empty:
        return []

    metadata = MACRO_SERIES[ticker]
    records: list[dict] = []

    for date, row in data.iterrows():
        close = row["Close"]

        if hasattr(close, "iloc"):
            close = close.iloc[0]

        if close is None:
            continue

        try:
            value = float(close)
        except (TypeError, ValueError):
            continue

        records.append(
            {
                "series_id": metadata["series_id"],
                "series_name": metadata["series_name"],
                "ticker": ticker,
                "date": date.strftime("%Y-%m-%d"),
                "value": value,
                "unit": metadata["unit"],
                "source": "Yahoo Finance",
                "source_url": "https://finance.yahoo.com/",
                "ingested_at": datetime.now(
                    timezone.utc
                ).isoformat(),
            }
        )

    return records


def ingest_macro_series(
    ticker: str,
    start: str = "2025-01-01",
    output_path: str | None = None,
) -> list[dict]:
    """Ingest one market-observable macro series."""

    if ticker not in MACRO_SERIES:
        raise ValueError(
            f"Unsupported macro ticker: {ticker}. "
            f"Supported: {list(MACRO_SERIES)}"
        )

    records = fetch_macro_market_data(
        ticker=ticker,
        start=start,
    )

    if output_path:
        save_json(records, output_path)

    return records


def ingest_macro_data(
    start: str = "2025-01-01",
    output_path: str | None = None,
) -> list[dict]:
    """Ingest all configured macro market series."""

    records: list[dict] = []

    for ticker in MACRO_SERIES:
        records.extend(
            fetch_macro_market_data(
                ticker=ticker,
                start=start,
            )
        )

    records.sort(
        key=lambda item: (
            item["date"],
            item["series_id"],
        )
    )

    if output_path:
        save_json(records, output_path)

    return records
