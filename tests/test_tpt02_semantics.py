"""
Regression and semantics test for benchmark question TPT-02.

TPT-02: "What risks affected NVIDIA as of June 1, 2025?"
Reference time: 2025-06-01T00:00:00+00:00
Expected behavior: Safe refusal ("Insufficient evidence in the retrieved corpus.")
Rationale: The earliest NVIDIA SEC filing in the 99-filing corpus is dated August 27, 2025.
As of June 1, 2025, no primary NVIDIA filings exist in the corpus.
"""

import json
from pathlib import Path
import pytest
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parent.parent


def test_tpt02_benchmark_definition():
    """Verify that TPT-02 in benchmark/questions.json is properly defined as unanswerable."""
    questions_file = ROOT / "benchmark" / "questions.json"
    assert questions_file.exists(), f"Benchmark file not found at {questions_file}"

    with open(questions_file) as f:
        questions = json.load(f)

    tpt02_list = [q for q in questions if q["question_id"] == "TPT-02"]
    assert len(tpt02_list) == 1, "Expected exactly one question with id TPT-02"
    tpt02 = tpt02_list[0]

    assert tpt02["answerable"] is False
    assert "Insufficient evidence" in tpt02["ground_truth_answer"]
    assert tpt02["reference_time"] == "2025-06-01T00:00:00+00:00"
    assert "company:NVDA" in tpt02["entities"]
    assert "August 27, 2025" in tpt02["unanswerability_rationale"]


def test_corpus_nvda_filing_dates_strictly_after_cutoff():
    """Verify that all NVIDIA filings in the corpus are available only after 2025-06-01."""
    manifest_file = ROOT / "data" / "processed" / "sec_corpus_manifest.json"
    if not manifest_file.exists():
        pytest.skip("SEC corpus manifest not found.")

    with open(manifest_file) as f:
        manifest = json.load(f).get("completed", {})

    cutoff = datetime(2025, 6, 1, 0, 0, 0, tzinfo=timezone.utc)
    nvda_keys = [k for k in manifest if k.startswith("NVDA:")]
    assert len(nvda_keys) >= 10, f"Expected at least 10 NVDA filings, found {len(nvda_keys)}"

    earliest_available = None
    for k in nvda_keys:
        doc_path = ROOT / manifest[k]["document_path"]
        assert doc_path.exists(), f"Document file missing: {doc_path}"
        with open(doc_path) as df:
            doc = json.load(df)
        avail_str = doc.get("available_time")
        assert avail_str is not None, f"Document {k} has no available_time"
        avail_dt = datetime.fromisoformat(avail_str)
        if earliest_available is None or avail_dt < earliest_available:
            earliest_available = avail_dt

        # Assert no NVDA filing was available before the June 1, 2025 cutoff
        assert avail_dt > cutoff, f"NVDA filing {k} has available_time {avail_str} <= cutoff {cutoff}"

    # Specifically confirm the earliest NVDA filing is August 27, 2025
    assert earliest_available is not None
    assert earliest_available.year == 2025
    assert earliest_available.month == 8
    assert earliest_available.day == 27


def test_vector_store_tpt02_returns_zero_nvda_chunks():
    """Verify that vector search with cutoff 2025-06-01 yields zero NVDA chunks."""
    vector_path = ROOT / "data" / "vector_store"
    if not vector_path.exists():
        pytest.skip("Vector store not present.")

    from src.retrieval.vector import VectorStore

    store = VectorStore(vector_path)
    cutoff = datetime(2025, 6, 1, 0, 0, 0, tzinfo=timezone.utc)

    # Search for NVIDIA risks with cutoff
    results = store.search(
        query="What risks affected NVIDIA as of June 1, 2025?",
        top_k=20,
        reference_time=cutoff,
    )

    # Check whether any returned chunk belongs to NVIDIA
    for chunk in results:
        company = chunk.payload.get("company", "")
        ticker = chunk.payload.get("ticker", "")
        assert company != "NVDA" and ticker != "NVDA", (
            f"Returned chunk {chunk.chunk_id} belongs to NVDA despite temporal cutoff!"
        )
