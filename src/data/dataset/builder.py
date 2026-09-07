from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

from src.data.features.unified import extract_unified_features
from src.data.graph.schema import TemporalGraph
from src.data.graph.snapshot import build_temporal_snapshot
from src.data.storage.local import save_json


# ============================================================
# DATE UTILITIES
# ============================================================


def _parse_date(value: str) -> datetime:
    """Parse an ISO-formatted date/time string."""

    return datetime.fromisoformat(value)


# ============================================================
# TARGET GENERATION
# ============================================================


def _future_market_return(
    graph: TemporalGraph,
    ticker: str,
    as_of: str,
    horizon_days: int,
) -> float | None:
    """
    Calculate the forward market return used as the prediction
    target for a point-in-time snapshot.

    Temporal semantics
    ------------------
    Reference price:
        Latest market closing price available on or before
        the snapshot date.

    Target price:
        Latest available market closing price within the
        forward horizon.

    IMPORTANT
    ---------
    Future market observations are used ONLY to construct
    target_return and target_direction.

    They must never be passed to feature extraction.
    """

    ticker = ticker.upper()

    cutoff = _parse_date(
        as_of[:10]
    )

    # --------------------------------------------------------
    # Find the latest market observation available at snapshot
    # --------------------------------------------------------

    current_nodes = []

    for node in graph.nodes:

        if node.node_type != "market":
            continue

        node_ticker = str(
            node.properties.get(
                "ticker",
                "",
            )
        ).upper()

        if node_ticker != ticker:
            continue

        if not node.event_time:
            continue

        event_date = _parse_date(
            node.event_time[:10]
        )

        if event_date <= cutoff:
            current_nodes.append(
                node
            )

    if not current_nodes:
        return None

    current_nodes.sort(
        key=lambda node: node.event_time or ""
    )

    current_node = current_nodes[-1]

    current_close = (
        current_node.properties.get(
            "close"
        )
    )

    if (
        current_close is None
        or current_close <= 0
    ):
        return None

    # --------------------------------------------------------
    # Define forward horizon
    # --------------------------------------------------------

    target_date = (
        cutoff
        + timedelta(
            days=horizon_days
        )
    )

    # --------------------------------------------------------
    # Find future market observations inside horizon
    # --------------------------------------------------------

    future_nodes = []

    for node in graph.nodes:

        if node.node_type != "market":
            continue

        node_ticker = str(
            node.properties.get(
                "ticker",
                "",
            )
        ).upper()

        if node_ticker != ticker:
            continue

        if not node.event_time:
            continue

        event_date = _parse_date(
            node.event_time[:10]
        )

        if (
            event_date > cutoff
            and event_date <= target_date
        ):
            future_nodes.append(
                node
            )

    if not future_nodes:
        return None

    future_nodes.sort(
        key=lambda node: node.event_time or ""
    )

    # --------------------------------------------------------
    # Use the final available trading observation inside the
    # requested forecast horizon.
    # --------------------------------------------------------

    target_node = future_nodes[-1]

    future_close = (
        target_node.properties.get(
            "close"
        )
    )

    if (
        future_close is None
        or future_close <= 0
    ):
        return None

    # --------------------------------------------------------
    # Forward return
    # --------------------------------------------------------

    return (
        float(future_close)
        / float(current_close)
    ) - 1.0


# ============================================================
# FEATURE FLATTENING
# ============================================================


def _flatten_features(
    features: dict[str, Any],
) -> dict[str, Any]:
    """
    Convert nested unified feature groups into flat ML columns.

    The resulting dictionary is directly suitable for conversion
    into a pandas DataFrame.
    """

    market = features.get(
        "market",
        {},
    )

    market_v2 = features.get(
        "market_v2",
        {},
    )

    sec = features.get(
        "sec",
        {},
    )

    news = features.get(
        "news",
        {},
    )

    macro = features.get(
        "macro",
        {},
    )

    # --------------------------------------------------------
    # Macro values
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Flatten feature groups
    # --------------------------------------------------------

    return {

        # ====================================================
        # Entity Metadata
        # ====================================================

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

        # ====================================================
        # Original Market Features
        # ====================================================

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

        # ====================================================
        # Market V2 Features
        # ====================================================

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

        # ====================================================
        # SEC V1 Features
        # ====================================================

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

        # ====================================================
        # SEC V3 Recent Activity
        # ====================================================

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

        # ====================================================
        # SEC V3 Dynamics
        # ====================================================

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

        # ====================================================
        # SEC V3 Recency
        # ====================================================

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

        # ====================================================
        # News V1 Features
        # ====================================================

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

        # ====================================================
        # News V2 Features
        # ====================================================

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

        # ====================================================
        # Macro Features
        # ====================================================

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


# ============================================================
# DATASET CONSTRUCTION
# ============================================================


def build_company_dataset(
    graph: TemporalGraph | dict[str, Any],
    ticker: str,
    snapshot_dates: list[str],
    horizon_days: int = 5,
) -> list[dict[str, Any]]:
    """
    Build ML-ready point-in-time examples for one company.

    For each snapshot:

        historical information
                |
                v
        temporal graph snapshot
                |
                v
        feature extraction
                |
                v
        ML feature vector

    Future market information is used exclusively for:

        target_return
        target_direction

    This preserves the temporal separation required for
    point-in-time financial prediction.
    """

    # --------------------------------------------------------
    # Convert dictionary graph to TemporalGraph
    # --------------------------------------------------------

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
                    properties=node.get(
                        "properties",
                        {},
                    ),
                    event_time=node.get(
                        "event_time"
                    ),
                    available_time=node.get(
                        "available_time"
                    ),
                )
                for node in graph["nodes"]
            ],
            edges=[
                GraphEdge(
                    edge_id=edge["edge_id"],
                    source=edge["source"],
                    target=edge["target"],
                    relationship=edge["relationship"],
                    event_time=edge.get(
                        "event_time"
                    ),
                    available_time=edge.get(
                        "available_time"
                    ),
                    properties=edge.get(
                        "properties",
                        {},
                    ),
                )
                for edge in graph["edges"]
            ],
        )

    # --------------------------------------------------------
    # Company identifier
    # --------------------------------------------------------

    ticker = ticker.upper()

    company_id = (
        f"company:{ticker}"
    )

    # --------------------------------------------------------
    # Market V2 warm-up requirements
    # --------------------------------------------------------

    required_market_features = [
        "return_5d",
        "return_20d",
        "return_60d",
        "price_vs_20d_ma",
        "price_vs_50d_ma",
        "momentum_20d",
        "volatility_20d",
    ]

    dataset: list[dict[str, Any]] = []

    # --------------------------------------------------------
    # Normalize snapshot dates
    # --------------------------------------------------------

    normalized_dates = sorted(
        {
            str(date)[:10]
            for date in snapshot_dates
        }
    )

    # --------------------------------------------------------
    # Process snapshots chronologically
    # --------------------------------------------------------

    for as_of in normalized_dates:

        # ====================================================
        # Build point-in-time graph snapshot
        # ====================================================

        snapshot = build_temporal_snapshot(
            graph,
            as_of,
        )

        # ====================================================
        # Extract point-in-time features
        # ====================================================

        try:

            features = extract_unified_features(
                snapshot,
                company_id,
            )

        except ValueError:

            # Insufficient historical information.
            continue

        # ====================================================
        # Flatten features
        # ====================================================

        row = _flatten_features(
            features
        )

        # ----------------------------------------------------
        # Explicit metadata
        # ----------------------------------------------------

        row["entity_id"] = company_id

        row["ticker"] = ticker

        row["as_of"] = as_of

        # ====================================================
        # Market warm-up validation
        # ====================================================

        if any(
            row.get(feature) is None
            for feature in required_market_features
        ):
            continue

        # ====================================================
        # Future target generation
        # ====================================================

        future_return = _future_market_return(
            graph=graph,
            ticker=ticker,
            as_of=as_of,
            horizon_days=horizon_days,
        )

        if future_return is None:
            continue

        # ====================================================
        # Prediction targets
        # ====================================================

        row["target_return"] = float(
            future_return
        )

        row["target_direction"] = int(
            future_return > 0
        )

        # ====================================================
        # Store valid example
        # ====================================================

        dataset.append(
            row
        )

    return dataset


# ============================================================
# DATASET STORAGE
# ============================================================


def save_company_dataset(
    dataset: list[dict[str, Any]],
    path: str = (
        "data/processed/unified/"
        "aapl_training_dataset.json"
    ),
) -> None:
    """Save a generated company dataset to JSON."""

    save_json(
        dataset,
        path,
    )