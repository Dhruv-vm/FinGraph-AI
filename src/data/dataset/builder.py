from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any
from src.data.storage.local import save_json
from src.data.features.unified import extract_unified_features
from src.data.graph.snapshot import build_temporal_snapshot
from src.data.graph.schema import TemporalGraph


def _parse_date(value: str) -> datetime:
    return datetime.fromisoformat(value)


def _future_market_return(
    graph: TemporalGraph,
    ticker: str,
    as_of: str,
    horizon_days: int,
) -> float | None:
    """
    Calculate future return after the snapshot date.

    This function is intentionally used only for the target.
    It must never be used during feature extraction.
    """

    cutoff = _parse_date(as_of)

    future_nodes = [
        node
        for node in graph.nodes
        if node.node_type == "market"
        and node.properties.get("ticker") == ticker
        and node.event_time
        and _parse_date(node.event_time[:10]) > cutoff
    ]

    future_nodes.sort(
        key=lambda node: node.event_time or ""
    )

    if not future_nodes:
        return None

    current_nodes = [
        node
        for node in graph.nodes
        if node.node_type == "market"
        and node.properties.get("ticker") == ticker
        and node.event_time
        and node.event_time[:10] <= as_of
    ]

    current_nodes.sort(
        key=lambda node: node.event_time or ""
    )

    if not current_nodes:
        return None

    current_close = current_nodes[-1].properties.get("close")

    if current_close is None or current_close == 0:
        return None

    target_date = cutoff + timedelta(days=horizon_days)

    eligible = [
        node
        for node in future_nodes
        if _parse_date(node.event_time[:10]) <= target_date
    ]

    if not eligible:
        return None

    target_node = eligible[-1]
    future_close = target_node.properties.get("close")

    if future_close is None:
        return None

    return (future_close / current_close) - 1

def _flatten_features(
    features: dict[str, Any],
) -> dict[str, Any]:
    """Convert nested feature groups into ML-friendly columns."""

    market = features.get("market", {})
    market_v2 = features.get("market_v2", {})
    sec = features.get("sec", {})
    news = features.get("news", {})
    macro = features.get("macro", {})

    macro_values = macro.get(
        "macro_values",
        {},
    )

    dgs10 = macro_values.get(
        "DGS10",
        {},
    )

    dgs3mo = macro_values.get(
        "DGS3MO",
        {},
    )

    return {
        "entity_id": features.get(
            "entity_id"
        ),

        "ticker": features.get(
            "entity_id",
            "",
        ).replace(
            "company:",
            "",
        ),

        "as_of": features.get(
            "as_of"
        ),

        # =========================
        # Original Market Features
        # =========================

        "market_event_count": market.get(
            "market_event_count"
        ),

        "latest_close": market.get(
            "latest_close"
        ),

        "latest_daily_return": market.get(
            "latest_daily_return"
        ),

        "latest_volatility_20d": market.get(
            "latest_volatility_20d"
        ),

        "cumulative_return": market.get(
            "cumulative_return"
        ),

        # =========================
        # Market V2 Features
        # =========================

        "return_5d": market_v2.get(
            "return_5d"
        ),

        "return_20d": market_v2.get(
            "return_20d"
        ),

        "return_60d": market_v2.get(
            "return_60d"
        ),

        "price_vs_20d_ma": market_v2.get(
            "price_vs_20d_ma"
        ),

        "price_vs_50d_ma": market_v2.get(
            "price_vs_50d_ma"
        ),

        "momentum_20d": market_v2.get(
            "momentum_20d"
        ),

        "volatility_20d": market_v2.get(
            "volatility_20d"
        ),

        # =========================
        # SEC V1 Features
        # =========================

        "sec_filing_count": sec.get(
            "sec_filing_count"
        ),

        "sec_10k_count": sec.get(
            "sec_10k_count"
        ),

        "sec_10q_count": sec.get(
            "sec_10q_count"
        ),

        "sec_8k_count": sec.get(
            "sec_8k_count"
        ),

        "sec_other_count": sec.get(
            "sec_other_count"
        ),

        # =========================
        # SEC V3 Recent Activity
        # =========================

        "sec_filings_5d": sec.get(
            "sec_filings_5d"
        ),

        "sec_filings_20d": sec.get(
            "sec_filings_20d"
        ),

        "sec_filings_60d": sec.get(
            "sec_filings_60d"
        ),

        "sec_10k_recent": sec.get(
            "sec_10k_recent"
        ),

        "sec_10q_recent": sec.get(
            "sec_10q_recent"
        ),

        "sec_8k_recent": sec.get(
            "sec_8k_recent"
        ),

        # =========================
        # SEC V3 Dynamics
        # =========================

        "sec_filing_velocity": sec.get(
            "sec_filing_velocity"
        ),

        "sec_filing_acceleration": sec.get(
            "sec_filing_acceleration"
        ),

        "sec_8k_ratio": sec.get(
            "sec_8k_ratio"
        ),

        "sec_10q_ratio": sec.get(
            "sec_10q_ratio"
        ),

        # =========================
        # SEC V3 Recency
        # =========================

        "days_since_last_filing": sec.get(
            "days_since_last_filing"
        ),

        "days_since_last_10k": sec.get(
            "days_since_last_10k"
        ),

        "days_since_last_10q": sec.get(
            "days_since_last_10q"
        ),

        "days_since_last_8k": sec.get(
            "days_since_last_8k"
        ),

        # =========================
        # News V1 Features
        # =========================

        "news_event_count": news.get(
            "news_event_count"
        ),

        "average_sentiment": news.get(
            "average_sentiment"
        ),

        "positive_news_count": news.get(
            "positive_news_count"
        ),

        "negative_news_count": news.get(
            "negative_news_count"
        ),

        "neutral_news_count": news.get(
            "neutral_news_count"
        ),

        "unique_news_sources": news.get(
            "unique_news_sources"
        ),

        # =========================
        # News V2 Features
        # =========================

        "sentiment_std": news.get(
            "sentiment_std"
        ),

        "positive_ratio": news.get(
            "positive_ratio"
        ),

        "negative_ratio": news.get(
            "negative_ratio"
        ),

        "neutral_ratio": news.get(
            "neutral_ratio"
        ),

        "sentiment_strength": news.get(
            "sentiment_strength"
        ),

        "source_diversity": news.get(
            "source_diversity"
        ),

        # =========================
        # Macro Features
        # =========================

        "macro_series_count": macro.get(
            "macro_series_count"
        ),

        "dgs10": dgs10.get(
            "value"
        ),

        "dgs3mo": dgs3mo.get(
            "value"
        ),
    }
def build_company_dataset(
    graph: TemporalGraph | dict[str, Any],
    ticker: str,
    snapshot_dates: list[str],
    horizon_days: int = 5,
) -> list[dict[str, Any]]:
    """
    Build ML-ready point-in-time examples for one company.

    Features come strictly from information available at each
    snapshot date.

    The future return is used only as the prediction target.
    """

    if isinstance(graph, dict):
        from src.data.graph.schema import (
            GraphEdge,
            GraphNode,
        )

        graph = TemporalGraph(
            nodes=[
                GraphNode(
                    node_id=node["node_id"],
                    node_type=node["node_type"],
                    properties=node.get("properties", {}),
                    event_time=node.get("event_time"),
                    available_time=node.get("available_time"),
                )
                for node in graph["nodes"]
            ],
            edges=[
                GraphEdge(
                    edge_id=edge["edge_id"],
                    source=edge["source"],
                    target=edge["target"],
                    relationship=edge["relationship"],
                    event_time=edge.get("event_time"),
                    available_time=edge.get("available_time"),
                    properties=edge.get("properties", {}),
                )
                for edge in graph["edges"]
            ],
        )

    company_id = f"company:{ticker.upper()}"

    V2_FEATURE_COLUMNS = [
        "return_5d",
        "return_20d",
        "return_60d",
        "price_vs_20d_ma",
        "price_vs_50d_ma",
        "momentum_20d",
        "volatility_20d",
    ]

    dataset = []

    for as_of in sorted(snapshot_dates):
        snapshot = build_temporal_snapshot(
            graph,
            as_of,
        )

        try:
            features = extract_unified_features(
                snapshot,
                company_id,
            )
        except ValueError:
            continue

        future_return = _future_market_return(
            graph,
            ticker.upper(),
            as_of,
            horizon_days,
        )

        if future_return is None:
            continue

        row = _flatten_features(features)

        # Always store the dataset snapshot date as YYYY-MM-DD.
        # Prevents mixed timezone-aware / timezone-naive values.
        row["as_of"] = as_of[:10]

        # Skip warm-up rows where V2 features
        # do not have enough historical data.
        if any(
            row.get(column) is None
            for column in V2_FEATURE_COLUMNS
        ):
            continue

        row["target_return"] = future_return

        row["target_direction"] = (
            1
            if future_return > 0
            else 0
        )

        dataset.append(row)

    return dataset


def save_company_dataset(
    dataset: list[dict[str, Any]],
    path: str = "data/processed/unified/aapl_training_dataset.json",
) -> None:
    """Save a generated company dataset to JSON."""

    save_json(
        dataset,
        path,
    )