from __future__ import annotations

import re

from datetime import datetime, timezone
from typing import Any

from src.data.graph.snapshot import is_available, parse_datetime
from src.retrieval.evidence import GraphEvidence, RetrievalEvidence
from src.retrieval.graph import GraphRetriever
from src.retrieval.vector import RetrievedChunk, VectorStore


def apply_entity_balanced_allocation(
    candidates: list[RetrievalEvidence],
    target_entities: list[str] | None,
    top_k: int,
    min_per_entity: int = 3,
) -> list[RetrievalEvidence]:
    """
    Apply entity-balanced evidence allocation for comparative queries.

    Reserves up to `min_per_entity` verified candidate items for each required entity,
    and fills remaining slots from globally ranked candidates in score order.
    Does not force allocations if sufficient verified evidence does not exist.
    """
    if not candidates:
        return []
    if not target_entities or len(target_entities) < 2:
        return candidates[:top_k]

    # Normalize target entity identifiers (e.g. "company:AAPL" -> "AAPL", "regulator:FTC" -> "FTC")
    clean_targets = []
    for ent in target_entities:
        clean = ent.split(":", 1)[1] if ":" in ent else ent
        clean_targets.append((ent, clean.upper(), clean.lower()))

    selected_ids: set[str] = set()
    entity_allocated: dict[str, list[RetrievalEvidence]] = {ent: [] for ent, _, _ in clean_targets}

    def candidate_matches_entity(item: RetrievalEvidence, ent: str, clean_upper: str, clean_lower: str) -> bool:
        # Check source company
        src_co = (item.source_company or item.metadata.get("source_company") or item.metadata.get("company") or "").upper()
        if src_co and src_co == clean_upper:
            return True
        # Check graph edge nodes
        for e in item.metadata.get("graph_edges", []) or []:
            if isinstance(e, dict):
                src = str(e.get("source", ""))
                tgt = str(e.get("target", ""))
                if ent in (src, tgt) or clean_upper in (src.upper(), tgt.upper()):
                    return True
        # Check text mentions with verified provenance
        text_lower = (item.text or "").lower()
        if clean_lower in text_lower:
            if getattr(item, "provenance_verified", True) is not False:
                return True
        return False

    # 1. Allocate up to min_per_entity for each required entity
    for ent, clean_upper, clean_lower in clean_targets:
        count = 0
        for item in candidates:
            if item.evidence_id in selected_ids:
                continue
            if candidate_matches_entity(item, ent, clean_upper, clean_lower):
                entity_allocated[ent].append(item)
                selected_ids.add(item.evidence_id)
                count += 1
                if count >= min_per_entity:
                    break

    # Flatten allocated items
    balanced_selection: list[RetrievalEvidence] = []
    for ent, _, _ in clean_targets:
        balanced_selection.extend(entity_allocated[ent])

    # 2. Fill remaining slots from globally ranked candidates
    for item in candidates:
        if len(balanced_selection) >= top_k:
            break
        if item.evidence_id not in selected_ids:
            balanced_selection.append(item)
            selected_ids.add(item.evidence_id)

    # 3. Sort final selection deterministically by score DESC, rank ASC
    balanced_selection.sort(key=lambda ev: (-ev.score, ev.rank, ev.evidence_id))
    for idx, ev in enumerate(balanced_selection):
        ev.rank = idx + 1

    return balanced_selection[:top_k]


_MONTHS = {m: i for i, m in enumerate(
    ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december"], 1
)}


# Topics whose answers sit in a specific, consistently-titled filing section. A question that
# matches a topic also searches with that section's wording (assist mode only), because users'
# phrasing ("who owns most of NVIDIA") rarely resembles the filing's ("Security Ownership of
# Certain Beneficial Owners").
_TOPIC_EXPANSIONS: list[tuple[re.Pattern[str], str, str | None]] = [
    (
        re.compile(r"\b(own|owns|owned|owner\w*|ownership|stake\w*|shareholder\w*|stockholder\w*|majority|largest holder\w*|major holder\w*|institutional|holds? (?:the )?most|share\w* in)\b", re.I),
        "Security Ownership of Certain Beneficial Owners and Management 5% stockholders beneficially owned percent of class shares outstanding",
        "DEF 14A",
    ),
    (
        re.compile(r"\b(ceo pay|compensation|salary|salaries|paid|pay package|bonus\w*|stock awards?|executive pay)\b", re.I),
        "Summary Compensation Table named executive officers salary bonus stock awards total compensation",
        "DEF 14A",
    ),
    (
        re.compile(r"\b(board of directors|directors?|board members?|independent director\w*|chair\w*)\b", re.I),
        "Board of Directors nominees director independence committees",
        "DEF 14A",
    ),
]


def _topic_queries(query: str) -> list[tuple[str, str | None]]:
    return [(q, dt) for pat, q, dt in _TOPIC_EXPANSIONS if pat.search(query)]


def _query_periods(query: str) -> list[str]:
    """Extract ISO period-end dates ("September 30, 2025", "2025-09-30") mentioned in a query."""
    out: list[str] = []
    for m in re.finditer(r"\b(" + "|".join(_MONTHS) + r")\s+(\d{1,2}),?\s+(\d{4})\b", query.lower()):
        out.append(f"{m.group(3)}-{_MONTHS[m.group(1)]:02d}-{int(m.group(2)):02d}")
    out.extend(re.findall(r"\b\d{4}-\d{2}-\d{2}\b", query))
    return list(dict.fromkeys(out))


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
        candidate_fetch_k: int | None = None,
        balanced_fusion: bool = False,
        min_per_entity: int = 3,
        target_entities: list[str] | None = None,
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
            candidate_fetch_k: Optional explicit candidate pool size for vector and graph search.

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

        fetch_k = candidate_fetch_k if candidate_fetch_k is not None else max(top_k * 2, 20)

        # 1. Vector Search
        vector_chunks: list[RetrievedChunk] = []
        if self.vector_store is not None:
            vector_chunks = self.vector_store.search(
                query=query,
                top_k=fetch_k,
                reference_time=ref_dt,
            )

        # 1b. Issuer-scoped search (live assist mode only; off for benchmark runs).
        # Searches inside each named company's own filings, and its matching fiscal period
        # when the question states a period-end date, then puts those hits first.
        if self.vector_store is not None and getattr(self, "issuer_scoped", False):
            tickers = [e.split(":", 1)[1].upper() for e in (target_entities or []) if e.startswith("company:") and e.split(":", 1)[1].isalpha()]
            for t in getattr(self, "extra_scoped_tickers", []) or []:
                if t not in tickers:
                    tickers.append(t)
            tickers = tickers[:2]
            periods = _query_periods(query)
            wants_latest = bool(re.search(r"\b(latest|most recent|current|recent quarter|last quarter|newest)\b", query.lower()))
            scoped: list[RetrievedChunk] = []
            for t in tickers:
                t_periods = list(periods)
                if wants_latest and not t_periods and hasattr(self.vector_store, "fiscal_periods"):
                    try:
                        # Newest periodic report (10-Q/10-K) first, then the newest filing of any kind.
                        t_periods = self.vector_store.fiscal_periods(t, ref_dt, ("10-Q", "10-K"))[:1]
                        t_periods += [p for p in self.vector_store.fiscal_periods(t, ref_dt)[:1] if p not in t_periods]
                    except Exception:
                        t_periods = []
                for fp in t_periods[:2]:
                    scoped.extend(self.vector_store.search(query=query, top_k=6, reference_time=ref_dt, company=t, fiscal_period=fp))
                scoped.extend(self.vector_store.search(query=query, top_k=8, reference_time=ref_dt, company=t))
            # Section-targeted searches for topic questions go first.
            topical: list[RetrievedChunk] = []
            for t in tickers:
                for tq, dt in _topic_queries(query):
                    hits = self.vector_store.search(query=tq, top_k=4, reference_time=ref_dt, company=t, document_type=dt)
                    if not hits and dt:
                        hits = self.vector_store.search(query=tq, top_k=4, reference_time=ref_dt, company=t)
                    for h in hits:
                        # Marked so the pipeline can keep these in front of the generator.
                        h.payload["topical_match"] = tq
                    topical.extend(hits)
            scoped = topical + scoped
            seen_ids: set[str] = set()
            merged: list[RetrievedChunk] = []
            for c in scoped + vector_chunks:
                if c.chunk_id in seen_ids:
                    continue
                seen_ids.add(c.chunk_id)
                merged.append(c)
            vector_chunks = merged[: max(fetch_k, len(scoped))]

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
                top_k=fetch_k,
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

                graph_edge_paths: list[list[dict[str, Any]]] = []
                for _, g in g_matches:
                    if "graph_edges" in g.properties and g.properties["graph_edges"]:
                        graph_edge_paths.append(list(g.properties["graph_edges"]))
                    elif g.source_node and g.target_node and g.relationship and " -> " not in g.relationship:
                        graph_edge_paths.append([{
                            "source": g.source_node,
                            "relationship": g.relationship,
                            "target": g.target_node,
                        }])

                rrf_score = (
                    (vector_weight / (self.rrf_k + v_rank))
                    + (graph_weight / (self.rrf_k + best_g_rank))
                )

                v_co = payload.get("company")
                g_co = best_g_item.source_company
                src_co = v_co or g_co
                prov_ver = bool(best_g_item.provenance_verified)
                attr_stat = best_g_item.attribution_status or "verified"

                if v_co and best_g_item.source_node.startswith("company:"):
                    g_src_ticker = best_g_item.source_node.split(":", 1)[1].upper()
                    if v_co.upper() != g_src_ticker and not best_g_item.provenance_verified:
                        prov_ver = False
                        attr_stat = "conflicting_attribution"

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
                            "graph_edge_paths": graph_edge_paths,
                            "graph_edges": [e for path in graph_edge_paths for e in path],
                            "source_company": src_co,
                            "provenance_verified": prov_ver,
                            "attribution_status": attr_stat,
                        },
                        source_company=src_co,
                        provenance_verified=prov_ver,
                        attribution_status=attr_stat,
                    )
                )
            else:
                # Vector only
                v_co = payload.get("company")
                doc_id = payload.get("document_id")
                prov_ver = bool(v_co and doc_id)
                attr_stat = "verified_issuer" if prov_ver else "missing_provenance"
                rrf_score = vector_weight / (self.rrf_k + v_rank)
                fused_items.append(
                    RetrievalEvidence(
                        evidence_id=f"vector:{chunk_id}",
                        source_type="vector",
                        score=rrf_score,
                        rank=0,
                        text=v_chunk.text,
                        document_id=doc_id,
                        chunk_id=chunk_id,
                        available_time=payload.get("available_time"),
                        event_time=payload.get("event_time"),
                        publication_date=payload.get("publication_date"),
                        source_url=payload.get("source_url"),
                        source_reference=payload.get("source_reference"),
                        retrieval_sources=["vector"],
                        original_scores={"vector": v_chunk.score},
                        original_ranks={"vector": v_rank},
                        metadata=dict(payload, source_company=v_co, provenance_verified=prov_ver, attribution_status=attr_stat),
                        source_company=v_co,
                        provenance_verified=prov_ver,
                        attribution_status=attr_stat,
                    )
                )

        # Process graph results not already fused with a vector chunk
        for g_rank_0, g_item in enumerate(graph_items):
            g_rank = g_rank_0 + 1
            if g_item.evidence_id in handled_graph_evidence_ids:
                continue

            edge_paths: list[list[dict[str, Any]]] = []
            if "graph_edges" in g_item.properties and g_item.properties["graph_edges"]:
                edge_paths.append(list(g_item.properties["graph_edges"]))
            elif g_item.source_node and g_item.target_node and g_item.relationship and " -> " not in g_item.relationship:
                edge_paths.append([{
                    "source": g_item.source_node,
                    "relationship": g_item.relationship,
                    "target": g_item.target_node,
                }])

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
                        "graph_edge_paths": edge_paths,
                        "graph_edges": edge_paths[0] if edge_paths else [],
                        "source_company": g_item.source_company,
                        "provenance_verified": g_item.provenance_verified,
                        "attribution_status": g_item.attribution_status,
                    },
                    source_company=g_item.source_company,
                    provenance_verified=g_item.provenance_verified,
                    attribution_status=g_item.attribution_status,
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

        if balanced_fusion and target_entities and len(target_entities) >= 2:
            return apply_entity_balanced_allocation(
                candidates=fused_items,
                target_entities=target_entities,
                top_k=top_k,
                min_per_entity=min_per_entity,
            )

        return fused_items[:top_k]
