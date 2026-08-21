from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any
import pandas as pd
import yfinance as yf


DEFAULT_START_DATE = "2020-01-01"


def fetch_market_data(
    ticker: str,
    start: str = DEFAULT_START_DATE,
    end: str | None = None,
) -> list[dict[str, Any]]:
    """
    Fetch historical daily OHLCV market data for a ticker.

    Returns normalized records suitable for downstream
    temporal validation, NLP/market alignment, and prediction.
    """

    data = yf.download(
        ticker,
        start=start,
        end=end,
        auto_adjust=False,
        progress=False,
    )
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    if data.empty:
        return []

    records: list[dict[str, Any]] = []

    for timestamp, row in data.iterrows():
        records.append(
            {
                "ticker": ticker,
                "date": timestamp.strftime("%Y-%m-%d"),
                "open": float(row["Open"]),
                "high": float(row["High"]),
                "low": float(row["Low"]),
                "close": float(row["Close"]),
                "adj_close": float(row["Adj Close"]),
                "volume": int(row["Volume"]),
            }
        )

    return records


def add_market_features(
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Add daily return and rolling volatility features.

    Return:
        Percentage price change from the previous trading day.

    Volatility:
        20-day rolling standard deviation of daily returns.
    """

    if not records:
        return []

    ordered = sorted(
        records,
        key=lambda record: record["date"],
    )

    previous_close: float | None = None
    returns: list[float | None] = []

    for record in ordered:
        current_close = record["close"]

        if previous_close is None:
            daily_return = None
        else:
            daily_return = (
                current_close - previous_close
            ) / previous_close

        returns.append(daily_return)
        previous_close = current_close

    for index, record in enumerate(ordered):
        record["daily_return"] = returns[index]

        if index < 20:
            record["volatility_20d"] = None
        else:
            window = [
                value
                for value in returns[index - 19 : index + 1]
                if value is not None
            ]

            if len(window) < 2:
                record["volatility_20d"] = None
            else:
                mean = sum(window) / len(window)

                variance = sum(
                    (value - mean) ** 2
                    for value in window
                ) / (len(window) - 1)

                record["volatility_20d"] = variance**0.5

    return ordered


def ingest_market_data(
    ticker: str,
    start: str = DEFAULT_START_DATE,
    end: str | None = None,
) -> list[dict[str, Any]]:
    """Fetch and enrich historical market data."""

    records = fetch_market_data(
        ticker=ticker,
        start=start,
        end=end,
    )

    return add_market_features(records)
