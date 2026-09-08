from __future__ import annotations

import hashlib
import json
from typing import Any

from src.data.documents.schema import Document


def _stable_document_id(
    source: str,
    company: str | None,
    document_type: str,
    source_reference: str | None,
    source_url: str | None,
) -> str:
    """Create a deterministic document identifier."""

    payload = {
        "source": source,
        "company": company,
        "document_type": document_type,
        "source_reference": source_reference,
        "source_url": source_url,
    }

    digest = hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()[:16]

    return f"doc:{source}:{digest}"


def build_news_document(record: dict[str, Any]) -> Document:
    """Convert a normalized news record into a canonical Document."""

    ticker = record.get("ticker")
    source = record.get("source") or "news"
    source_url = record.get("url")

    title = (record.get("title") or "").strip()
    description = (record.get("description") or "").strip()

    text = title
    if description:
        text = f"{title}\n\n{description}" if title else description

    published_at = record.get("published_at")

    document_id = _stable_document_id(
        source=source,
        company=ticker,
        document_type="news_article",
        source_reference=source_url,
        source_url=source_url,
    )

    return Document(
        document_id=document_id,
        company=ticker.upper() if ticker else None,
        source=source,
        document_type="news_article",
        title=title or None,
        text=text,
        publication_date=published_at,
        available_time=published_at,
        ingested_at=record.get("ingested_at"),
        source_url=source_url,
        source_reference=source_url,
        metadata={
            "query": record.get("query"),
        },
    )


def build_sec_document(record: dict[str, Any]) -> Document:
    """Convert a normalized SEC filing record into a canonical Document."""

    ticker = record.get("ticker")
    accession = record.get("accession_number")

    form = record.get("form") or "SEC filing"

    filing_date = record.get("filing_date")
    acceptance_datetime = record.get("acceptance_datetime")
    report_date = record.get("report_date")

    source_url = record.get("source_url")

    document_id = _stable_document_id(
        source="sec",
        company=ticker,
        document_type=form,
        source_reference=accession,
        source_url=source_url,
    )

    return Document(
        document_id=document_id,
        company=ticker.upper() if ticker else None,
        source="sec",
        document_type=form,
        title=record.get("primary_doc_description"),
        text=record.get("text", ""),
        publication_date=filing_date,
        fiscal_period=report_date,
        available_time=acceptance_datetime,
        ingested_at=record.get("ingested_at"),
        source_url=source_url,
        source_reference=accession,
        metadata={
            "cik": record.get("cik"),
            "primary_document": record.get("primary_document"),
            "act": record.get("act"),
            "file_number": record.get("file_number"),
            "items": record.get("items"),
            "is_xbrl": record.get("is_xbrl"),
            "is_inline_xbrl": record.get("is_inline_xbrl"),
        },
    )
