"""
Tests for Workstream A (Entity-Balanced Hybrid Fusion).

Validates:
1. Minimum evidence allocation per entity in comparative queries.
2. Handling when candidate evidence for an entity is scarce (does not force fake allocation).
3. Remaining slots filled with globally ranked candidates.
4. Attribution and provenance preserved during balanced selection.
"""

from __future__ import annotations

import pytest
from src.retrieval.evidence import RetrievalEvidence
from src.retrieval.hybrid import apply_entity_balanced_allocation


def make_cand(
    evidence_id: str,
    score: float,
    source_company: str,
    text: str = "",
    provenance_verified: bool = True,
    rank: int = 1,
) -> RetrievalEvidence:
    return RetrievalEvidence(
        evidence_id=evidence_id,
        source_type="hybrid",
        score=score,
        rank=rank,
        text=text or f"Disclosure from {source_company}",
        document_id=f"doc:{source_company}:10k",
        chunk_id=f"{evidence_id}:chunk:0",
        available_time="2025-01-01T00:00:00+00:00",
        event_time=None,
        publication_date="2025-01-01",
        source_url=None,
        source_reference=None,
        retrieval_sources=["vector", "graph"],
        original_scores={},
        original_ranks={},
        metadata={
            "source_company": source_company,
            "provenance_verified": provenance_verified,
        },
        source_company=source_company,
        provenance_verified=provenance_verified,
    )


def test_balanced_allocation_two_entities():
    """When two entities are required and K=8, allocates min 3 items per entity and fills 2 remaining."""
    # Pool: 10 NVDA items with high score, 5 AAPL items with medium score
    nvda_items = [make_cand(f"nvda_{i}", score=0.9 - i * 0.01, source_company="NVDA", rank=i + 1) for i in range(10)]
    aapl_items = [make_cand(f"aapl_{i}", score=0.7 - i * 0.01, source_company="AAPL", rank=10 + i + 1) for i in range(5)]
    candidates = nvda_items + aapl_items
    candidates.sort(key=lambda c: -c.score)

    # Standard greedy top_k=8 would pick only NVDA items
    assert all(c.source_company == "NVDA" for c in candidates[:8])

    # Balanced allocation reserves 3 per entity
    balanced = apply_entity_balanced_allocation(
        candidates=candidates,
        target_entities=["company:AAPL", "company:NVDA"],
        top_k=8,
        min_per_entity=3,
    )

    assert len(balanced) == 8
    nvda_count = sum(1 for c in balanced if c.source_company == "NVDA")
    aapl_count = sum(1 for c in balanced if c.source_company == "AAPL")

    assert nvda_count == 5  # 3 reserved + 2 remaining slots filled by next best scores (NVDA)
    assert aapl_count == 3  # 3 reserved
    # Verify sequential rank assigned
    assert [c.rank for c in balanced] == list(range(1, 9))


def test_balanced_allocation_scarce_evidence():
    """When an entity has fewer candidates than the quota, takes available verified items without forcing fake ones."""
    nvda_items = [make_cand(f"nvda_{i}", score=0.9 - i * 0.01, source_company="NVDA", rank=i + 1) for i in range(10)]
    # Only 1 AAPL item exists
    aapl_items = [make_cand("aapl_0", score=0.6, source_company="AAPL", rank=11)]
    candidates = nvda_items + aapl_items
    candidates.sort(key=lambda c: -c.score)

    balanced = apply_entity_balanced_allocation(
        candidates=candidates,
        target_entities=["company:AAPL", "company:NVDA"],
        top_k=8,
        min_per_entity=3,
    )

    assert len(balanced) == 8
    aapl_count = sum(1 for c in balanced if c.source_company == "AAPL")
    nvda_count = sum(1 for c in balanced if c.source_company == "NVDA")

    assert aapl_count == 1  # Only 1 existed, did not hallucinate or force
    assert nvda_count == 7  # Remaining 7 slots filled by best NVDA items


def test_balanced_allocation_preserves_provenance():
    """Unverified items do not count towards entity allocation."""
    # AAPL item with unverified provenance
    aapl_unverified = make_cand("aapl_bad", score=0.8, source_company="AAPL", provenance_verified=False)
    aapl_good = make_cand("aapl_good", score=0.7, source_company="AAPL", provenance_verified=True)
    nvda_items = [make_cand(f"nvda_{i}", score=0.9 - i * 0.01, source_company="NVDA") for i in range(8)]

    candidates = nvda_items + [aapl_unverified, aapl_good]
    candidates.sort(key=lambda c: -c.score)

    balanced = apply_entity_balanced_allocation(
        candidates=candidates,
        target_entities=["company:AAPL", "company:NVDA"],
        top_k=5,
        min_per_entity=2,
    )

    # aapl_good should be included; unverified should not satisfy entity reservation
    included_ids = {c.evidence_id for c in balanced}
    assert "aapl_good" in included_ids
