from __future__ import annotations

import re
from collections import Counter
from typing import Any

from src.data.graph.schema import TemporalGraph


POSITIVE_WORDS = {
    "beat",
    "beats",
    "growth",
    "strong",
    "surge",
    "surges",
    "profit",
    "profits",
    "positive",
    "gain",
    "gains",
    "upgrade",
    "upgraded",
    "bullish",
    "record",
    "revenue",
    "success",
    "outperform",
    "outperformed",
    "higher",
    "increase",
    "increased",
    "expansion",
    "optimistic",
}

NEGATIVE_WORDS = {
    "fall",
    "falls",
    "drop",
    "drops",
    "decline",
    "declines",
    "loss",
    "losses",
    "negative",
    "weak",
    "downgrade",
    "downgraded",
    "bearish",
    "risk",
    "risks",
    "warning",
    "warns",
    "miss",
    "misses",
    "lower",
    "decrease",
    "decreased",
    "cut",
    "cuts",
    "lawsuit",
    "investigation",
    "uncertainty",
}


def _score_text(text: str) -> float:
    """
    Calculate a deterministic lexicon sentiment score.

    Score is normalized to [-1, 1].
    """

    words = re.findall(
        r"[a-zA-Z]+",
        text.lower(),
    )

    if not words:
        return 0.0

    positive = sum(
        1
        for word in words
        if word in POSITIVE_WORDS
    )

    negative = sum(
        1
        for word in words
        if word in NEGATIVE_WORDS
    )

    total = positive + negative

    if total == 0:
        return 0.0

    return (positive - negative) / total


def _sentiment_label(score: float) -> str:
    """Convert sentiment score into a categorical label."""

    if score > 0.2:
        return "positive"

    if score < -0.2:
        return "negative"

    return "neutral"


def extract_news_features(
    snapshot: TemporalGraph,
    company_id: str,
) -> dict[str, Any]:
    """
    Extract point-in-time News V2 features.

    Features describe the news information available
    at the snapshot date only.
    """

    news_nodes = []

    for node in snapshot.nodes:

        if node.node_type != "news":
            continue

        ticker = node.properties.get("ticker")

        if company_id != f"company:{ticker}":
            continue

        news_nodes.append(node)

    scores: list[float] = []
    sources: list[str] = []

    for node in news_nodes:

        title = node.properties.get(
            "title",
            "",
        )

        description = node.properties.get(
            "description",
            "",
        )

        source = node.properties.get(
            "source"
        )

        text = f"{title} {description}"

        score = _score_text(text)

        scores.append(score)

        if source:
            sources.append(source)

    news_count = len(scores)

    if news_count:

        average_sentiment = (
            sum(scores) / news_count
        )

        positive_count = sum(
            1
            for score in scores
            if score > 0.2
        )

        negative_count = sum(
            1
            for score in scores
            if score < -0.2
        )

        neutral_count = (
            news_count
            - positive_count
            - negative_count
        )

        mean_absolute_sentiment = (
            sum(abs(score) for score in scores)
            / news_count
        )

        variance = sum(
            (score - average_sentiment) ** 2
            for score in scores
        ) / news_count

        sentiment_std = variance ** 0.5

    else:

        average_sentiment = 0.0
        positive_count = 0
        negative_count = 0
        neutral_count = 0
        mean_absolute_sentiment = 0.0
        sentiment_std = 0.0

    source_counts = Counter(sources)

    unique_sources = len(source_counts)

    return {
        "company_id": company_id,

        # Existing features
        "news_event_count": news_count,
        "average_sentiment": average_sentiment,
        "sentiment_label": _sentiment_label(
            average_sentiment
        ),
        "positive_news_count": positive_count,
        "negative_news_count": negative_count,
        "neutral_news_count": neutral_count,
        "unique_news_sources": unique_sources,

        # News V2
        "sentiment_std": sentiment_std,

        "positive_ratio": (
            positive_count / news_count
            if news_count
            else 0.0
        ),

        "negative_ratio": (
            negative_count / news_count
            if news_count
            else 0.0
        ),

        "neutral_ratio": (
            neutral_count / news_count
            if news_count
            else 0.0
        ),

        "sentiment_strength": mean_absolute_sentiment,

        "source_diversity": (
            unique_sources / news_count
            if news_count
            else 0.0
        ),

        "top_news_sources": [
            {
                "source": source,
                "count": count,
            }
            for source, count
            in source_counts.most_common(5)
        ],
    }