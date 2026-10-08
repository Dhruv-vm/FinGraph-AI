from __future__ import annotations

from typing import Any

from src.retrieval.evidence import RetrievalEvidence


class BaseReranker:
    """Base interface for retrieval rerankers in FinGraph AI."""

    def rerank(
        self,
        query: str,
        candidates: list[RetrievalEvidence],
        top_k: int | None = None,
    ) -> list[RetrievalEvidence]:
        raise NotImplementedError


class IdentityReranker(BaseReranker):
    """Pass-through reranker preserving candidate order."""

    def rerank(
        self,
        query: str,
        candidates: list[RetrievalEvidence],
        top_k: int | None = None,
    ) -> list[RetrievalEvidence]:
        k = top_k if top_k is not None else len(candidates)
        return candidates[:k]
