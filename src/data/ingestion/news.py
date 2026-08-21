from __future__ import annotations

from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from urllib.parse import quote
from urllib.request import Request, urlopen
from xml.etree import ElementTree as ET

from src.data.storage.local import save_json


USER_AGENT = "FinGraph-AI/0.1 research-project"


def parse_published_date(value: str | None) -> datetime | None:
    """Parse an RSS publication timestamp."""

    if not value:
        return None

    try:
        return parsedate_to_datetime(value).astimezone(timezone.utc)
    except (TypeError, ValueError, OverflowError):
        return None


def build_news_url(query: str) -> str:
    """Build a Google News RSS search URL."""

    encoded_query = quote(query)
    return f"https://news.google.com/rss/search?q={encoded_query}"


def fetch_rss(url: str) -> bytes:
    """Fetch an RSS feed with an explicit research user-agent."""

    request = Request(
        url,
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "application/rss+xml, application/xml, text/xml",
        },
    )

    with urlopen(request, timeout=20) as response:
        return response.read()


def parse_rss(
    xml_data: bytes,
    ticker: str,
    query: str,
) -> list[dict]:
    """Parse RSS articles into normalized FinGraph records."""

    root = ET.fromstring(xml_data)

    records: list[dict] = []

    for item in root.findall(".//item"):
        title = item.findtext("title")
        link = item.findtext("link")
        description = item.findtext("description")
        published_raw = item.findtext("pubDate")
        source = item.findtext("source")

        published_at = parse_published_date(published_raw)

        if not title or not link or not published_at:
            continue

        records.append(
            {
                "ticker": ticker,
                "query": query,
                "title": title.strip(),
                "description": (description or "").strip(),
                "url": link.strip(),
                "source": (source or "").strip(),
                "published_at": published_at.isoformat(),
                "ingested_at": datetime.now(timezone.utc).isoformat(),
            }
        )

    return records


def ingest_news(
    ticker: str,
    query: str | None = None,
    output_path: str | None = None,
) -> list[dict]:
    """
    Ingest financial news for a ticker.

    The query defaults to the ticker symbol.
    """

    query = query or ticker

    url = build_news_url(query)
    xml_data = fetch_rss(url)

    records = parse_rss(
        xml_data,
        ticker=ticker,
        query=query,
    )

    if output_path:
        save_json(records, output_path)

    return records