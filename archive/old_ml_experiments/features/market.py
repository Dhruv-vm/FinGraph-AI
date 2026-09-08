from __future__ import annotations

from typing import Any

from src.data.graph.schema import TemporalGraph


def extract_market_features(
    snapshot: TemporalGraph,
    company_id: str,
) -> dict[str, Any]:
    """
    Extract point-in-time market features for one company.

    Only market nodes already present in the temporal snapshot
    are considered, preventing future-data leakage.
    """

    ticker = company_id.replace("company:", "")

    market_nodes = [
        node
        for node in snapshot.nodes
        if node.node_type == "market"
        and node.properties.get("ticker") == ticker
    ]

    market_nodes.sort(
        key=lambda node: (
            node.event_time or "",
            node.available_time or "",
        )
    )

    if not market_nodes:
        return {
            "company_id": company_id,
            "market_event_count": 0,
            "latest_close": None,
            "latest_daily_return": None,
            "latest_volatility_20d": None,
            "cumulative_return": None,
        }

    latest = market_nodes[-1]
    latest_data = latest.properties

    first_close = market_nodes[0].properties.get("close")
    latest_close = latest_data.get("close")

    cumulative_return = None

    if (
        first_close is not None
        and latest_close is not None
        and first_close != 0
    ):
        cumulative_return = (
            latest_close / first_close
        ) - 1

    return {
        "company_id": company_id,
        "market_event_count": len(market_nodes),
        "latest_close": latest_close,
        "latest_daily_return": latest_data.get(
            "daily_return"
        ),
        "latest_volatility_20d": latest_data.get(
            "volatility_20d"
        ),
        "cumulative_return": cumulative_return,
    }
