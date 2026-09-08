from src.data.documents.schema import Document
from src.data.chunking.structure_aware import chunk_document


def make_document(text: str) -> Document:
    return Document(
        document_id="doc:test:001",
        company="AAPL",
        source="sec",
        document_type="10-Q",
        title="Test Filing",
        text=text,
        publication_date="2026-07-31",
        fiscal_period="2026-06-27",
        section="Item 1",
        event_time=None,
        available_time="2026-07-31T10:01:02+00:00",
        ingested_at="2026-08-01T00:00:00+00:00",
        source_url="https://example.com/filing",
        source_reference="0000320193-26-000020",
        metadata={"test": True},
    )


def test_chunk_document_returns_chunks():
    document = make_document(
        "Apple reported strong results.\n\n"
        "Revenue increased during the quarter.\n\n"
        "Services revenue also increased."
    )

    chunks = chunk_document(document)

    assert len(chunks) > 0


def test_chunk_size_is_bounded():
    text = "A" * 6000
    document = make_document(text)

    chunks = chunk_document(document)

    assert all(len(chunk.text) <= 1800 for chunk in chunks)


def test_chunk_metadata_is_preserved():
    document = make_document(
        "Revenue increased significantly during the quarter."
    )

    chunks = chunk_document(document)

    chunk = chunks[0]

    assert chunk.document_id == document.document_id
    assert chunk.company == document.company
    assert chunk.source == document.source
    assert chunk.document_type == document.document_type
    assert chunk.publication_date == document.publication_date
    assert chunk.fiscal_period == document.fiscal_period
    assert chunk.available_time == document.available_time
    assert chunk.source_url == document.source_url
    assert chunk.source_reference == document.source_reference


def test_chunk_ids_are_deterministic():
    document = make_document(
        "First paragraph.\n\n"
        "Second paragraph.\n\n"
        "Third paragraph."
    )

    chunks_a = chunk_document(document)
    chunks_b = chunk_document(document)

    assert [c.chunk_id for c in chunks_a] == [
        c.chunk_id for c in chunks_b
    ]


def test_chunk_indexes_are_sequential():
    document = make_document("A " * 2000)

    chunks = chunk_document(document)

    assert [chunk.chunk_index for chunk in chunks] == list(
        range(len(chunks))
    )
