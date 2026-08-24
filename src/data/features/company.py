from __future__ import annotations

from collections import Counter
from typing import Any

from src.data.graph.schema import TemporalGraph


def _event_nodes(
    graph: TemporalGraph,
    company_id: str,
) -> list[Any]:
    """Return event nodes directly connected to a company."""

    event_ids = set()

    for edge in graph.edges:
        if edge.source == company_id:
            event_ids.add(edge.target)
        elif edge.target == company_id:
            event_ids.add(edge.source)

    return [
        node
        for node in graph.nodes
        if node.node_id in event_ids
        and node.node_type != "company"
    ]


def _latest_market_event(
    events: list[Any],
) -> Any | None:
    """Return the latest market event."""

    market = [
        event
        for event in events
        if event.node_type == "market"
    ]

    if not market:
        return None

    return max(
        market,
        key=lambda x: x.event_time or "",
    )


def _market_features(
    events: list[Any],
) -> dict[str, Any]:
    """Extract market features."""

    market = [
        event
        for event in events
        if event.node_type == "market"
    ]

    if not market:
        return {
            "market_event_count": 0,
            "latest_close": None,
            "latest_daily_return": None,
            "latest_volatility_20d": None,
            "cumulative_return": None,
        }

    market = sorted(
        market,
        key=lambda x: x.event_time or "",
    )

    latest = market[-1]
    latest_data = latest.properties

    first_close = market[0].properties.get("close")
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
        "market_event_count": len(market),
        "latest_close": latest_close,
        "latest_daily_return": latest_data.get(
            "daily_return"
        ),
        "latest_volatility_20d": latest_data.get(
            "volatility_20d"
        ),
        "cumulative_return": cumulative_return,
    }


def _sec_features(
    events: list[Any],
) -> dict[str, Any]:
    """Extract SEC filing features."""

    filings = [
        event
        for event in events
        if event.node_type == "sec_filing"
    ]

    forms = Counter(
        event.properties.get("form")
        for event in filings
    )

    return {
        "sec_filing_count": len(filings),
        "sec_10k_count": forms.get("10-K", 0),
        "sec_10q_count": forms.get("10-Q", 0),
        "sec_8k_count": forms.get("8-K", 0),
        "sec_other_count": sum(
            count
            for form, count in forms.items()
            if form not in {"10-K", "10-Q", "8-K"}
        ),
    }


def _news_features(
    events: list[Any],
) -> dict[str, Any]:
    """Extract news features."""

    news = [
        event
        for event in events
        if event.node_type == "news"
    ]

    return {
        "news_event_count": len(news),
    }


def _macro_features(
    events: list[Any],
) -> dict[str, Any]:
    """Extract macro features."""

    macro = [
        event
        for event in events
        if event.node_type == "macro"
    ]

    return {
        "macro_event_count": len(macro),
    }


def extract_company_features(
    graph: TemporalGraph,
    company_id: str,
) -> dict[str, Any]:
    """
    Build a point-in-time feature vector for a company.

    The graph passed here should already be a temporal snapshot.
    Therefore no future information can enter the feature vector.
    """

    company = next(
        (
            node
            for node in graph.nodes
            if node.node_id == company_id
            and node.node_type == "company"
        ),
        None,
    )

    if company is None:
        raise ValueError(
            f"Company not found: {company_id}"
        )

    events = _event_nodes(
        graph,
        company_id,
    )

    features = {
        "entity_id": company.node_id,
        "ticker": company.properties.get("ticker"),
        "company": company.properties.get("company"),
        "sector": company.properties.get("sector"),
        "as_of": max(
            (
                node.available_time
                for node in graph.nodes
                if node.available_time
            ),
            default=None,
        ),
    }

    features.update(
        _market_features(events)
    )

    features.update(
        _sec_features(events)
    )

    features.update(
        _news_features(events)
    )

    features.update(
        _macro_features(events)
    )

    features["total_connected_events"] = len(events)

    return features