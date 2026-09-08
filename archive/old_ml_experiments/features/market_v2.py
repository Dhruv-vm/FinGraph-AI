from __future__ import annotations

from typing import Any

from src.data.graph.schema import TemporalGraph


def _market_nodes(
    snapshot: TemporalGraph,
    company_id: str,
) -> list[Any]:
    """Return market observations for the company."""

    ticker = company_id.replace("company:", "").upper()

    nodes = [
        node
        for node in snapshot.nodes
        if node.node_type == "market"
        and node.properties.get("ticker", "").upper() == ticker
    ]

    return sorted(
        nodes,
        key=lambda node: node.event_time or "",
    )


def _close_values(
    nodes: list[Any],
) -> list[float]:
    """Return valid closing prices."""

    values = []

    for node in nodes:
        close = node.properties.get("close")

        if close is not None:
            values.append(float(close))

    return values


def _return_over_period(
    closes: list[float],
    periods: int,
) -> float | None:
    """Calculate return over the requested number of observations."""

    if len(closes) <= periods:
        return None

    current = closes[-1]
    previous = closes[-(periods + 1)]

    if previous == 0:
        return None

    return (current / previous) - 1


def _moving_average(
    closes: list[float],
    periods: int,
) -> float | None:
    """Calculate simple moving average."""

    if len(closes) < periods:
        return None

    values = closes[-periods:]

    return sum(values) / len(values)


def _volatility(
    nodes: list[Any],
    periods: int = 20,
) -> float | None:
    """Calculate rolling volatility from daily returns."""

    returns = []

    for node in nodes:
        value = node.properties.get("daily_return")

        if value is not None:
            returns.append(float(value))

    if len(returns) < periods:
        return None

    recent = returns[-periods:]

    mean = sum(recent) / len(recent)

    variance = sum(
        (value - mean) ** 2
        for value in recent
    ) / len(recent)

    return variance ** 0.5


def extract_market_features_v2(
    snapshot: TemporalGraph,
    company_id: str,
) -> dict[str, Any]:
    """
    Extract time-relative market features.

    All features are calculated only from observations
    available inside the temporal snapshot.
    """

    nodes = _market_nodes(
        snapshot,
        company_id,
    )

    closes = _close_values(nodes)

    if not nodes or not closes:
        return {
            "company_id": company_id,
            "market_event_count": 0,
            "latest_close": None,
            "return_5d": None,
            "return_20d": None,
            "return_60d": None,
            "price_vs_20d_ma": None,
            "price_vs_50d_ma": None,
            "momentum_20d": None,
            "volatility_20d": None,
        }

    latest_close = closes[-1]

    ma_20 = _moving_average(
        closes,
        20,
    )

    ma_50 = _moving_average(
        closes,
        50,
    )

    price_vs_20d_ma = None

    if ma_20 is not None and ma_20 != 0:
        price_vs_20d_ma = (
            latest_close / ma_20
        ) - 1

    price_vs_50d_ma = None

    if ma_50 is not None and ma_50 != 0:
        price_vs_50d_ma = (
            latest_close / ma_50
        ) - 1

    return_5d = _return_over_period(
        closes,
        5,
    )

    return_20d = _return_over_period(
        closes,
        20,
    )

    return_60d = _return_over_period(
        closes,
        60,
    )

    return {
        "company_id": company_id,
        "market_event_count": len(nodes),
        "latest_close": latest_close,
        "return_5d": return_5d,
        "return_20d": return_20d,
        "return_60d": return_60d,
        "price_vs_20d_ma": price_vs_20d_ma,
        "price_vs_50d_ma": price_vs_50d_ma,
        "momentum_20d": return_20d,
        "volatility_20d": _volatility(
            nodes,
            20,
        ),
    }