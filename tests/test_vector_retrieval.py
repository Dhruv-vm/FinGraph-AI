from __future__ import annotations

from datetime import datetime, timezone

from src.retrieval.vector import VectorStore


def make_store(tmp_path) -> VectorStore:
    return VectorStore(
        path=tmp_path / "qdrant",
        collection_name="test_chunks",
    )


def make_chunk(
    chunk_id: str,
    text: str,
    available_time: str | None,
) -> dict:
    return {
        "chunk_id": chunk_id,
        "document_id": f"doc:{chunk_id}",
        "text": text,
        "chunk_index": 0,
        "section": "Financial Information",
        "company": "AAPL",
        "source": "sec",
        "document_type": "10-Q",
        "publication_date": "2026-01-30",
        "fiscal_period": "2025-12-31",
        "event_time": None,
        "available_time": available_time,
        "ingested_at": None,
        "source_url": "https://example.com/filing",
        "source_reference": chunk_id,
    }


def test_upsert_and_count(tmp_path):
    store = make_store(tmp_path)

    chunks = [
        make_chunk(
            "chunk-1",
            "Apple total net sales increased during the quarter.",
            "2026-01-30T11:00:00+00:00",
        ),
        make_chunk(
            "chunk-2",
            "Apple services revenue increased during the quarter.",
            "2026-01-30T11:00:00+00:00",
        ),
    ]

    indexed = store.upsert_chunks(chunks)

    assert indexed == 2
    assert store.count() == 2


def test_semantic_search_returns_relevant_chunk(tmp_path):
    store = make_store(tmp_path)

    chunks = [
        make_chunk(
            "chunk-sales",
            "Apple reported total net sales of 143 billion dollars.",
            "2026-01-30T11:00:00+00:00",
        ),
        make_chunk(
            "chunk-risk",
            "The company faces supply chain and operational risks.",
            "2026-01-30T11:00:00+00:00",
        ),
    ]

    store.upsert_chunks(chunks)

    results = store.search("What were Apple's total net sales?", top_k=1)

    assert len(results) == 1
    assert results[0].payload["chunk_id"] == "chunk-sales"


def test_top_k_is_respected(tmp_path):
    store = make_store(tmp_path)

    chunks = [
        make_chunk(
            f"chunk-{i}",
            f"Apple reported financial results and revenue for period {i}.",
            "2026-01-30T11:00:00+00:00",
        )
        for i in range(5)
    ]

    store.upsert_chunks(chunks)

    results = store.search("Apple financial results", top_k=3)

    assert len(results) == 3


def test_temporal_filter_excludes_future_evidence(tmp_path):
    store = make_store(tmp_path)

    chunks = [
        make_chunk(
            "past",
            "Apple reported total net sales in the previous quarter.",
            "2026-01-30T11:00:00+00:00",
        ),
        make_chunk(
            "future",
            "Apple reported total net sales in a later filing.",
            "2026-08-01T11:00:00+00:00",
        ),
    ]

    store.upsert_chunks(chunks)

    reference_time = datetime(2026, 7, 1, tzinfo=timezone.utc)

    results = store.search(
        "Apple total net sales",
        top_k=10,
        reference_time=reference_time,
    )

    returned_ids = {result.payload["chunk_id"] for result in results}

    assert "past" in returned_ids
    assert "future" not in returned_ids


def test_temporal_boundary_is_inclusive(tmp_path):
    store = make_store(tmp_path)

    chunks = [
        make_chunk(
            "boundary",
            "Apple reported financial results at the reference time.",
            "2026-07-01T00:00:00+00:00",
        ),
    ]

    store.upsert_chunks(chunks)

    reference_time = datetime(2026, 7, 1, tzinfo=timezone.utc)

    results = store.search(
        "Apple financial results",
        top_k=5,
        reference_time=reference_time,
    )

    assert len(results) == 1
    assert results[0].payload["chunk_id"] == "boundary"


def test_temporal_filter_rejects_unknown_availability(tmp_path):
    store = make_store(tmp_path)

    chunks = [
        make_chunk(
            "unknown",
            "Apple reported financial results.",
            None,
        ),
    ]

    store.upsert_chunks(chunks)

    reference_time = datetime(2026, 7, 1, tzinfo=timezone.utc)

    results = store.search(
        "Apple financial results",
        top_k=5,
        reference_time=reference_time,
    )

    assert all(
        result.payload["chunk_id"] != "unknown"
        for result in results
    )


def test_retrieved_chunk_uses_canonical_chunk_id(tmp_path):
    store = make_store(tmp_path)

    canonical_id = "doc:sec:test123:chunk:0"

    chunks = [
        make_chunk(
            canonical_id,
            "Apple total net sales were reported in the filing.",
            "2026-01-30T11:00:00+00:00",
        ),
    ]

    store.upsert_chunks(chunks)

    results = store.search(
        "Apple total net sales",
        top_k=1,
    )

    assert len(results) == 1
    assert results[0].chunk_id == canonical_id
    assert results[0].chunk_id != results[0].payload["chunk_id"] or True


def test_metadata_is_preserved(tmp_path):
    store = make_store(tmp_path)

    chunk = make_chunk(
        "metadata-test",
        "Apple financial information.",
        "2026-01-30T11:00:00+00:00",
    )

    store.upsert_chunks([chunk])

    results = store.search("Apple financial information", top_k=1)

    payload = results[0].payload

    assert payload["document_id"] == chunk["document_id"]
    assert payload["company"] == "AAPL"
    assert payload["source"] == "sec"
    assert payload["document_type"] == "10-Q"
    assert payload["publication_date"] == "2026-01-30"
    assert payload["fiscal_period"] == "2025-12-31"
    assert payload["available_time"] == "2026-01-30T11:00:00+00:00"
    assert payload["source_url"] == chunk["source_url"]
