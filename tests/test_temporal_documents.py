from src.data.documents import (
    Document,
    build_news_document,
    build_sec_document,
    is_available_at,
)
from src.data.chunking.structure_aware import chunk_document


def test_news_document_preserves_temporal_metadata():
    record = {
        "ticker": "NVDA",
        "source": "Example News",
        "title": "NVIDIA announces new product",
        "description": "Example description.",
        "url": "https://example.com/article",
        "published_at": "2025-01-10T12:00:00+00:00",
        "ingested_at": "2025-01-10T12:05:00+00:00",
    }

    document = build_news_document(record)

    assert document.document_id.startswith("doc:")
    assert document.company == "NVDA"
    assert document.publication_date == record["published_at"]
    assert document.available_time == record["published_at"]
    assert document.ingested_at == record["ingested_at"]
    assert document.source_url == record["url"]


def test_sec_document_preserves_distinct_temporal_fields():
    record = {
        "ticker": "NVDA",
        "cik": "0001045810",
        "form": "10-Q",
        "filing_date": "2025-05-20",
        "report_date": "2025-04-27",
        "acceptance_datetime": "2025-05-20T16:30:00+00:00",
        "accession_number": "0000000000-25-000001",
        "primary_document": "example.htm",
    }

    document = build_sec_document(record)

    assert document.publication_date == "2025-05-20"
    assert document.fiscal_period == "2025-04-27"
    assert document.available_time == (
        "2025-05-20T16:30:00+00:00"
    )


def test_future_evidence_is_rejected():
    document = Document(
        document_id="doc:test",
        company="NVDA",
        source="test",
        document_type="news_article",
        text="future evidence",
        available_time="2025-01-10T12:00:00+00:00",
    )

    assert not is_available_at(
        document,
        "2025-01-10T11:59:59+00:00",
    )

    assert is_available_at(
        document,
        "2025-01-10T12:00:00+00:00",
    )


def test_unknown_availability_is_rejected():
    document = Document(
        document_id="doc:test",
        company="NVDA",
        source="test",
        document_type="news_article",
        text="unknown timing",
    )

    assert not is_available_at(
        document,
        "2025-01-10T12:00:00+00:00",
    )


def test_chunk_metadata_is_preserved():
    document = Document(
        document_id="doc:test",
        company="NVDA",
        source="test",
        document_type="news_article",
        text="Paragraph one.\n\nParagraph two.",
        publication_date="2025-01-10T12:00:00+00:00",
        available_time="2025-01-10T12:00:00+00:00",
        source_url="https://example.com",
    )

    chunks = chunk_document(document)

    assert len(chunks) == 1
    assert chunks[0].document_id == document.document_id
    assert chunks[0].available_time == document.available_time
    assert chunks[0].source_url == document.source_url
