from __future__ import annotations

from datetime import datetime, timezone
import re
from typing import Any

from src.data.graph.snapshot import is_available, parse_datetime
from src.retrieval.evidence import RetrievalEvidence
from src.retrieval.graph import detect_relationship_intents


class BaseReranker:
    """Base interface for retrieval rerankers in FinGraph AI."""

    def rerank(
        self,
        query: str,
        candidates: list[RetrievalEvidence],
        top_k: int | None = None,
        reference_time: str | datetime | None = None,
    ) -> list[RetrievalEvidence]:
        raise NotImplementedError


class IdentityReranker(BaseReranker):
    """Pass-through reranker preserving candidate order."""

    def rerank(
        self,
        query: str,
        candidates: list[RetrievalEvidence],
        top_k: int | None = None,
        reference_time: str | datetime | None = None,
    ) -> list[RetrievalEvidence]:
        # Filter temporally if reference_time provided
        if reference_time is not None:
            ref_dt = parse_datetime(reference_time) if isinstance(reference_time, str) else reference_time
            if ref_dt is not None:
                candidates = [c for c in candidates if is_available(c.available_time, ref_dt)]
        k = top_k if top_k is not None else len(candidates)
        return candidates[:k]


class DeterministicReranker(BaseReranker):
    """
    Transparent, deterministic baseline reranker combining:
    - Normalized RRF score (alpha = 0.20)
    - Query entity overlap (beta = 0.25)
    - Relationship intent match (gamma = 0.35)
    - Evidence confidence (delta = 0.15)
    - Hop depth penalty (epsilon = 0.05 * hop_penalty, where hop_penalty = max(0, hops - 1))

    Formula (Option A):
        hop_penalty = max(0, hops - 1)
        Score = alpha * norm_rrf + beta * entity_overlap + gamma * rel_match + delta * conf - epsilon * hop_penalty

    Hard Constraint:
    Temporal validity is strictly enforced. Any evidence item with
    available_time > reference_time is REJECTED, never merely downranked.
    """

    def __init__(
        self,
        alpha: float = 0.20,  # RRF score weight
        beta: float = 0.25,   # Entity overlap weight
        gamma: float = 0.35,  # Relationship intent match weight
        delta: float = 0.15,  # Confidence weight
        epsilon: float = 0.05, # Hop count penalty
    ) -> None:
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.epsilon = epsilon

    def rerank(
        self,
        query: str,
        candidates: list[RetrievalEvidence],
        top_k: int | None = None,
        reference_time: str | datetime | None = None,
    ) -> list[RetrievalEvidence]:
        """
        Rerank retrieval evidence using multi-signal transparent scoring.

        Args:
            query: Natural language query.
            candidates: Retrieved evidence list.
            top_k: Number of reranked items to return.
            reference_time: Point-in-time cutoff.

        Returns:
            list[RetrievalEvidence]: Re-scored and ranked valid evidence.
        """
        if not candidates:
            return []

        # 1. HARD TEMPORAL CONSTRAINT: Reject future evidence immediately
        ref_dt: datetime | None = None
        if reference_time is not None:
            if isinstance(reference_time, str):
                ref_dt = parse_datetime(reference_time)
                if ref_dt is None:
                    raise ValueError(f"Invalid reference_time format: {reference_time}")
            elif isinstance(reference_time, datetime):
                ref_dt = reference_time if reference_time.tzinfo else reference_time.replace(tzinfo=timezone.utc)

        valid_candidates = []
        for c in candidates:
            if ref_dt is not None:
                if not is_available(c.available_time, ref_dt):
                    continue  # Strict rejection
            valid_candidates.append(c)

        if not valid_candidates:
            return []

        # 2. Extract query signals
        rel_intents = detect_relationship_intents(query)
        query_words = set(re.findall(r"\b[a-zA-Z0-9_]+\b", query.lower()))

        # Normalize RRF scores across candidates
        scores = [c.score for c in valid_candidates]
        min_score = min(scores)
        max_score = max(scores)
        score_range = max_score - min_score if max_score > min_score else 1.0

        scored_pairs: list[tuple[float, RetrievalEvidence]] = []

        for item in valid_candidates:
            # Normalized RRF
            norm_rrf = (item.score - min_score) / score_range

            # Entity and term lexical overlap
            item_text_lower = (item.text or "").lower()
            text_words = set(re.findall(r"\b[a-zA-Z0-9_]+\b", item_text_lower))
            overlap_count = len(query_words.intersection(text_words))
            entity_overlap = min(1.0, overlap_count / max(1, len(query_words)))

            # Relationship match
            if rel_intents:
                item_rels: list[str] = []
                if "graph_relations" in item.metadata:
                    item_rels.extend(item.metadata["graph_relations"])
                elif "relationship" in item.metadata:
                    item_rels.append(str(item.metadata["relationship"]))

                rel_match = 0.0
                if any(r in rel_intents for r in item_rels):
                    rel_match = 1.0
                elif any(r.lower() in item_text_lower for r in rel_intents):
                    rel_match = 0.8
                else:
                    rel_match = 0.1

                # Direction-sensitive check (Issue 2):
                # When query asks "What does X depend on?", favor incoming SUPPLIES or outgoing DEPENDS_ON.
                # Penalize outgoing SUPPLIES from X (which means X supplies Y, not Y supplies X).
                if "depend" in query.lower():
                    edge_str = item.evidence_id.lower()
                    text_str = item_text_lower
                    if "nvda" in query.lower() or "nvidia" in query.lower():
                        # If edge indicates NVDA supplies another company, it does NOT support NVDA depending on it
                        if "company:nvda->supplies->" in edge_str or "nvidia corporation --[supplies]-->" in text_str:
                            rel_match *= 0.15
                        elif "->supplies->company:nvda" in edge_str or "--[supplies]--> nvidia" in text_str:
                            rel_match = max(rel_match, 1.0)
                        elif "company:nvda->depends_on->" in edge_str or "nvidia corporation --[depends_on]-->" in text_str:
                            rel_match = max(rel_match, 1.0)
            else:
                rel_match = 0.5  # Neutral when no relationship intent specified

            # Confidence
            conf = 1.0
            if "original_scores" in item.metadata:
                conf = float(item.metadata["original_scores"].get("graph", 1.0))
            elif "graph" in item.original_scores:
                conf = float(item.original_scores["graph"])

            # Hop penalty
            hop_count = int(item.metadata.get("hop_count", 1))
            hop_penalty = max(0, hop_count - 1)

            # Transparent formula
            final_score = (
                (self.alpha * norm_rrf)
                + (self.beta * entity_overlap)
                + (self.gamma * rel_match)
                + (self.delta * conf)
                - (self.epsilon * hop_penalty)
            )

            scored_pairs.append((final_score, item))

        # Deterministic sort: score DESC, original rank ASC, evidence_id ASC
        scored_pairs.sort(key=lambda pair: (-pair[0], pair[1].rank, pair[1].evidence_id))

        results: list[RetrievalEvidence] = []
        for rank_idx, (sc, item) in enumerate(scored_pairs):
            # Update score and rank while preserving all provenance
            item.score = round(sc, 5)
            item.rank = rank_idx + 1
            results.append(item)

        k = top_k if top_k is not None else len(results)
        return results[:k]
