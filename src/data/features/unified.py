from __future__ import annotations

from typing import Any

from src.data.graph.schema import TemporalGraph
from src.data.features.market import extract_market_features
from src.data.features.market_v2 import extract_market_features_v2
from src.data.features.sec import extract_sec_features
from src.data.features.news import extract_news_features
from src.data.features.macro import extract_macro_features


def extract_unified_features(
    snapshot: TemporalGraph,
    company_id: str,
) -> dict[str, Any]:
    """
    Combine all point-in-time feature groups.

    V1 market features are retained for comparison.
    V2 market features provide time-relative signals.
    """

    market = extract_market_features(
        snapshot,
        company_id,
    )

    market_v2 = extract_market_features_v2(
        snapshot,
        company_id,
    )

    sec = extract_sec_features(
        snapshot,
        company_id,
        as_of=snapshot.as_of,
    )

    news = extract_news_features(
        snapshot,
        company_id,
    )

    macro = extract_macro_features(
        snapshot,
        company_id,
    )

    return {
        "entity_id": company_id,
        "as_of": snapshot.as_of,
        "market": market,
        "market_v2": market_v2,
        "sec": sec,
        "news": news,
        "macro": macro,
    }