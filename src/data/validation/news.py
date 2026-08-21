from __future__ import annotations

from datetime import datetime
from hashlib import sha256
from typing import Any
from urllib.parse import urlparse


REQUIRED_FIELDS = (
    "ticker",
    "title",
    "url",
    "source",
    "published_at",
    "ingested_at",
)


def parse_timestamp(value: str | None) -> datetime | None:
    """Parse an ISO-8601 timestamp."""

    if not value:
        return None

    try:
        return datetime.fromisoformat(
            value.replace("Z", "+00:00")
        )
    except ValueError:
        return None


def is_valid_url(value: str | None) -> bool:
    """Check whether a URL has a valid HTTP(S) structure."""

    if not value:
        return False

    try:
        parsed = urlparse(value)
    except ValueError:
        return False

    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def article_fingerprint(article: dict[str, Any]) -> str:
    """Create a stable fingerprint for duplicate detection."""

    normalized = "|".join(
        [
            str(article.get("ticker", "")).strip().lower(),
            str(article.get("title", "")).strip().lower(),
            str(article.get("published_at", "")).strip(),
        ]
    )

    return sha256(
        normalized.encode("utf-8")
    ).hexdigest()


def validate_news_article(
    article: dict[str, Any],
) -> dict[str, Any]:
    """Validate one normalized news article."""

    errors: list[str] = []

    for field in REQUIRED_FIELDS:
        value = article.get(field)

        if value is None or not str(value).strip():
            errors.append(f"missing_{field}")

    ticker = str(article.get("ticker", "")).strip()

    if ticker and (
        len(ticker) > 10
        or not ticker.replace("-", "").isalnum()
    ):
        errors.append("invalid_ticker")

    if not is_valid_url(article.get("url")):
        errors.append("invalid_url")

    published_at = parse_timestamp(
        article.get("published_at")
    )

    ingested_at = parse_timestamp(
        article.get("ingested_at")
    )

    if published_at is None:
        errors.append("invalid_published_at")

    if ingested_at is None:
        errors.append("invalid_ingested_at")

    if (
        published_at is not None
        and ingested_at is not None
        and published_at > ingested_at
    ):
        errors.append(
            "published_after_ingestion"
        )

    return {
        "news_valid": len(errors) == 0,
        "news_errors": errors,
        "article_fingerprint": article_fingerprint(article),
    }


def validate_news(
    articles: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Validate a collection of normalized news articles."""

    validated: list[dict[str, Any]] = []

    seen_urls: set[str] = set()
    seen_fingerprints: set[str] = set()

    stats = {
        "total": len(articles),
        "valid": 0,
        "invalid": 0,
        "duplicate_urls": 0,
        "duplicate_articles": 0,
    }

    for article in articles:
        result = validate_news_article(article)

        url = str(
            article.get("url", "")
        ).strip()

        fingerprint = result["article_fingerprint"]

        if url and url in seen_urls:
            result["news_errors"].append(
                "duplicate_url"
            )
            stats["duplicate_urls"] += 1

        if fingerprint in seen_fingerprints:
            result["news_errors"].append(
                "duplicate_article"
            )
            stats["duplicate_articles"] += 1

        if url:
            seen_urls.add(url)

        seen_fingerprints.add(fingerprint)

        result["news_valid"] = (
            len(result["news_errors"]) == 0
        )

        enriched = {
            **article,
            **result,
        }

        validated.append(enriched)

        if result["news_valid"]:
            stats["valid"] += 1
        else:
            stats["invalid"] += 1

    return validated, stats
