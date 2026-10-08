from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from src.data.graph.snapshot import is_available, parse_datetime
from src.retrieval.evidence import GraphEvidence, RetrievalEvidence
from src.retrieval.graph import GraphRetriever
from src.retrieval.vector import RetrievedChunk, VectorStore


class HybridRetriever:
    """
    Hybrid Retrieval engine combining semantic vector search and
    temporal knowledge graph traversal using Reciprocal Rank Fusion (RRF).
    """

    def __init__(
        self,
        vector_store: VectorStore | None = None,
        graph_retriever: GraphRetriever | None = None,
        rrf_k: int = 60,
    ) -> None:
        self.vector_store = vector_store
        self.graph_retriever = graph_retriever
        self.rrf_k = rrf_k

    def retrieve(
        self,
        query: str,
        reference_time: str | datetime | None = None,
        top_k: int = 10,
        max_hops: int = 2,
        vector_weight: float = 1.0,
        graph_weight: float = 1.0,
    ) -> list[RetrievalEvidence]:
        """
        Execute hybrid retrieval across vector store and knowledge graph.

        Args:
            query: Natural language query.
            reference_time: Point-in-time cutoff (str or datetime).
            top_k: Number of final fused evidence records to return.
            max_hops: Maximum graph traversal depth.
            vector_weight: Multiplier weight for vector search RRF rank.
            graph_weight: Multiplier weight for graph traversal RRF rank.

        Returns:
            list[RetrievalEvidence]: Ranked, point-in-time fused evidence objects.
        """
        # Parse reference datetime for vector search
        ref_dt: datetime | None = None
        if reference_time is not None:
            if isinstance(reference_time, str):
                ref_dt = parse_datetime(reference_time)
                if ref_dt is None:
                    raise ValueError(f"Invalid reference_time format: {reference_time}")
            elif isinstance(reference_time, datetime):
                ref_dt = (
                    reference_time
                    if reference_time.tzinfo is not None
                    else reference_time.replace(tzinfo=timezone.utc)
                )

        candidate_fetch_k = max(top_k * 2, 20)

        # 1. Vector Search
        vector_chunks: list[RetrievedChunk] = []
        if self.vector_store is not None:
            vector_chunks = self.vector_store.search(
                query=query,
                top_k=candidate_fetch_k,
                reference_time=ref_dt,
            )

        # 2. Strict Point-in-Time Temporal Filtering on Vector Chunks
        if ref_dt is not None:
            vector_chunks = [
                chunk
                for chunk in vector_chunks
                if is_available(chunk.payload.get("available_time"), ref_dt)
            ]

        # 3. Graph Search
        graph_items: list[GraphEvidence] = []
        if self.graph_retriever is not None:
            graph_items = self.graph_retriever.retrieve(
                query=query,
                reference_time=ref_dt,
                max_hops=max_hops,
                top_k=candidate_fetch_k,
            )

        # 4. Strict Point-in-Time Temporal Filtering on Graph Evidence
        if ref_dt is not None:
            graph_items = [
                item
                for item in graph_items
                if is_available(item.available_time, ref_dt)
            ]

        # Build rank lookups
        # vector: chunk_id -> (rank, chunk)
        vector_by_chunk: dict[str, tuple[int, RetrievedChunk]] = {
            chunk.chunk_id: (idx + 1, chunk)
            for idx, chunk in enumerate(vector_chunks)
        }

        # graph by chunk_id (map to best graph rank and items)
        graph_by_chunk: dict[str, list[tuple[int, GraphEvidence]]] = {}
        for idx, item in enumerate(graph_items):
            if item.chunk_id:
                graph_by_chunk.setdefault(item.chunk_id, []).append((idx + 1, item))

        fused_items: list[RetrievalEvidence] = []
        handled_graph_evidence_ids: set[str] = set()

        # Process all vector results
        for chunk_id, (v_rank, v_chunk) in vector_by_chunk.items():
            payload = v_chunk.payload or {}

            if chunk_id in graph_by_chunk:
                # Hybrid match: present in both vector and graph
                g_matches = graph_by_chunk[chunk_id]
                best_g_rank, best_g_item = min(g_matches, key=lambda x: x[0])
                for _, g_item in g_matches:
                    handled_graph_evidence_ids.add(g_item.evidence_id)

                rrf_score = (
                    (vector_weight / (self.rrf_k + v_rank))
                    + (graph_weight / (self.rrf_k + best_g_rank))
                )

                fused_items.append(
                    RetrievalEvidence(
                        evidence_id=f"hybrid:{chunk_id}",
                        source_type="hybrid",
                        score=rrf_score,
                        rank=0,
                        text=v_chunk.text,
                        document_id=payload.get("document_id") or best_g_item.document_id,
                        chunk_id=chunk_id,
                        available_time=payload.get("available_time") or best_g_item.available_time,
                        event_time=payload.get("event_time") or best_g_item.event_time,
                        publication_date=payload.get("publication_date") or best_g_item.publication_date,
                        source_url=payload.get("source_url") or best_g_item.source_url,
                        source_reference=payload.get("source_reference") or best_g_item.source_reference,
                        retrieval_sources=["vector", "graph"],
                        original_scores={
                            "vector": v_chunk.score,
                            "graph": best_g_item.confidence,
                        },
                        original_ranks={
                            "vector": v_rank,
                            "graph": best_g_rank,
                        },
                        metadata={
                            "vector_score": v_chunk.score,
                            "graph_relations": [g.relationship for _, g in g_matches],
                            "graph_paths": [g.path for _, g in g_matches],
                        },
                    )
                )
            else:
                # Vector only
                rrf_score = vector_weight / (self.rrf_k + v_rank)
                fused_items.append(
                    RetrievalEvidence(
                        evidence_id=f"vector:{chunk_id}",
                        source_type="vector",
                        score=rrf_score,
                        rank=0,
                        text=v_chunk.text,
                        document_id=payload.get("document_id"),
                        chunk_id=chunk_id,
                        available_time=payload.get("available_time"),
                        event_time=payload.get("event_time"),
                        publication_date=payload.get("publication_date"),
                        source_url=payload.get("source_url"),
                        source_reference=payload.get("source_reference"),
                        retrieval_sources=["vector"],
                        original_scores={"vector": v_chunk.score},
                        original_ranks={"vector": v_rank},
                        metadata=payload,
                    )
                )

        # Process graph results not already fused with a vector chunk
        for g_rank_0, g_item in enumerate(graph_items):
            g_rank = g_rank_0 + 1
            if g_item.evidence_id in handled_graph_evidence_ids:
                continue

            rrf_score = graph_weight / (self.rrf_k + g_rank)
            fused_items.append(
                RetrievalEvidence(
                    evidence_id=f"graph:{g_item.evidence_id}",
                    source_type="graph",
                    score=rrf_score,
                    rank=0,
                    text=g_item.text,
                    document_id=g_item.document_id,
                    chunk_id=g_item.chunk_id,
                    available_time=g_item.available_time,
                    event_time=g_item.event_time,
                    publication_date=g_item.publication_date,
                    source_url=g_item.source_url,
                    source_reference=g_item.source_reference,
                    retrieval_sources=["graph"],
                    original_scores={"graph": g_item.confidence},
                    original_ranks={"graph": g_rank},
                    metadata={
                        "path": g_item.path,
                        "relationship": g_item.relationship,
                        "hop_count": g_item.hop_count,
                        "properties": g_item.properties,
                    },
                )
            )

        # Defensive post-fusion temporal check
        if ref_dt is not None:
            fused_items = [
                ev for ev in fused_items
                if is_available(ev.available_time, ref_dt)
            ]

        # Sort deterministically by RRF score DESC, then evidence_id ASC
        fused_items.sort(key=lambda ev: (-ev.score, ev.evidence_id))

        # Assign sequential rank (1-indexed)
        for idx, ev in enumerate(fused_items):
            ev.rank = idx + 1

        return fused_items[:top_k]
