"""
Knowledge Graph explorer read-only endpoints.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from fastapi import APIRouter, Depends, HTTPException, Query, status

from api.dependencies import get_pipeline
from api.schemas.graph import (
    CompanyListItem,
    CompanyListResponse,
    GraphEdgeItem,
    GraphNodeItem,
    SubgraphResponse,
)
from scripts.run_fingraph import FinGraphPipeline

logger = logging.getLogger("fingraph_api.routes.graph")

router = APIRouter(prefix="/graph", tags=["Knowledge Graph"])

# Tickers to highlight prominently at the top
CORE_TICKERS = [
    "NVDA",
    "MSFT",
    "AAPL",
    "AMZN",
    "GOOGL",
    "META",
    "TSLA",
    "JPM",
    "XOM",
    "JNJ",
    "NFLX",
    "V",
]


@router.get(
    "/companies",
    response_model=CompanyListResponse,
    summary="List all prominent companies present in the Knowledge Graph",
)
def get_kg_companies(
    pipeline: FinGraphPipeline = Depends(get_pipeline),
) -> CompanyListResponse:
    if not pipeline.graph_retriever or not pipeline.graph_retriever.graph:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Knowledge graph is not initialized.",
        )

    graph = pipeline.graph_retriever.graph
    edge_counts: dict[str, int] = defaultdict(int)
    for edge in graph.edges:
        edge_counts[edge.source] += 1
        edge_counts[edge.target] += 1

    core_items: list[CompanyListItem] = []
    other_items: list[CompanyListItem] = []

    for node in graph.nodes:
        if node.node_type == "Company" or node.node_id.startswith("company:"):
            raw_id = node.node_id.replace("company:", "")
            canonical_name = (
                node.properties.get("canonical_name")
                or node.properties.get("name")
                or raw_id
            )

            # Detect if ticker is a known symbol or hex hash
            is_hex = len(raw_id) == 16 and all(c in "0123456789abcdef" for c in raw_id)
            ticker_display = raw_id.upper() if not is_hex else canonical_name[:12]

            item = CompanyListItem(
                node_id=node.node_id,
                ticker=ticker_display,
                canonical_name=canonical_name,
                edge_count=edge_counts.get(node.node_id, 0),
            )

            if raw_id.upper() in CORE_TICKERS:
                core_items.append(item)
            else:
                other_items.append(item)

    # Sort core items by CORE_TICKERS priority order, others by edge count descending
    core_items.sort(
        key=lambda c: (
            CORE_TICKERS.index(c.ticker)
            if c.ticker in CORE_TICKERS
            else 999,
            -c.edge_count,
        )
    )
    other_items.sort(key=lambda c: c.edge_count, reverse=True)

    all_companies = core_items + other_items

    return CompanyListResponse(
        total_companies=len(all_companies),
        companies=all_companies,
    )


@router.get(
    "/subgraph",
    response_model=SubgraphResponse,
    summary="Get 1-hop or bounded subgraph surrounding a target company/entity node",
)
def get_kg_subgraph(
    node_id: str = Query(..., description="Target node identifier, e.g. 'company:MSFT' or 'MSFT'"),
    limit: int = Query(100, ge=5, le=500, description="Max edges to return"),
    offset: int = Query(0, ge=0, description="Offset for pagination"),
    filter_type: str = Query("companies", description="Filter mode: 'companies' (corporate network) or 'all'"),
    pipeline: FinGraphPipeline = Depends(get_pipeline),
) -> SubgraphResponse:
    if not pipeline.graph_retriever or not pipeline.graph_retriever.graph:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Knowledge graph is not initialized.",
        )

    graph = pipeline.graph_retriever.graph
    node_map = {n.node_id: n for n in graph.nodes}

    # Normalize target_id search
    clean_id = node_id.strip()
    target_id = clean_id

    if target_id not in node_map:
        # Try prepending company:
        candidate_company = f"company:{clean_id.upper()}"
        if candidate_company in node_map:
            target_id = candidate_company
        elif f"regulator:{clean_id.upper()}" in node_map:
            target_id = f"regulator:{clean_id.upper()}"
        else:
            # Fuzzy match on canonical_name or lower-case node_id
            lower_clean = clean_id.lower()
            for n in graph.nodes:
                cand_name = (
                    n.properties.get("canonical_name")
                    or n.properties.get("name")
                    or ""
                ).lower()
                if (
                    lower_clean == n.node_id.lower()
                    or lower_clean == cand_name
                    or lower_clean in cand_name
                ):
                    target_id = n.node_id
                    break

    if target_id not in node_map:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Node '{node_id}' not found in Knowledge Graph.",
        )

    # 1. Collect ALL direct edges connected to target_id
    all_direct_edges = [
        edge for edge in graph.edges
        if edge.source == target_id or edge.target == target_id
    ]
    total_direct_edges = len(all_direct_edges)

    # 2. Identify all connected corporate entities
    corporate_types = {"company", "supplier", "competitor", "regulator"}
    connected_company_ids = set()
    for edge in all_direct_edges:
        other_id = edge.target if edge.source == target_id else edge.source
        other_node = node_map.get(other_id)
        if other_node and other_node.node_type.lower() in corporate_types:
            connected_company_ids.add(other_id)
    total_connected_companies = len(connected_company_ids)

    # 3. Filter edges based on filter_type
    if filter_type == "companies":
        matching_edges = [
            edge for edge in all_direct_edges
            if (edge.target if edge.source == target_id else edge.source) in connected_company_ids
        ]
    else:
        matching_edges = all_direct_edges

    total_matching = len(matching_edges)

    # 4. Paginate or slice
    paginated_edges = matching_edges[offset : offset + limit]
    has_more = (offset + limit) < total_matching

    # 5. Collect connected node IDs (always include center target_id)
    connected_node_ids = {target_id}
    for edge in paginated_edges:
        connected_node_ids.add(edge.source)
        connected_node_ids.add(edge.target)

    # Calculate degrees within whole graph
    degrees: dict[str, int] = defaultdict(int)
    for edge in graph.edges:
        if edge.source in connected_node_ids:
            degrees[edge.source] += 1
        if edge.target in connected_node_ids:
            degrees[edge.target] += 1

    # Build node and edge items
    result_nodes: list[GraphNodeItem] = []
    for nid in connected_node_ids:
        n = node_map.get(nid)
        if n:
            label = (
                n.properties.get("canonical_name")
                or n.properties.get("name")
                or nid
            )
            result_nodes.append(
                GraphNodeItem(
                    id=n.node_id,
                    label=label,
                    type=n.node_type,
                    properties=n.properties or {},
                    degree=degrees.get(n.node_id, 0),
                )
            )

    result_edges: list[GraphEdgeItem] = []
    for e in paginated_edges:
        props = getattr(e, "properties", {}) or {}
        confidence = getattr(e, "confidence", None)
        if confidence is None:
            # Report the stored value only; never substitute a default score.
            confidence = props.get("confidence")
        result_edges.append(
            GraphEdgeItem(
                id=getattr(e, "edge_id", f"{e.source}->{e.relationship}->{e.target}"),
                source=e.source,
                target=e.target,
                relationship=e.relationship,
                available_time=getattr(e, "available_time", None),
                event_time=getattr(e, "event_time", None),
                confidence=float(confidence) if confidence is not None else None,
                document_id=props.get("document_id"),
                chunk_id=props.get("chunk_id"),
                source_url=props.get("source_url"),
                source_reference=props.get("source_reference"),
                publication_date=props.get("publication_date"),
                properties=props,
            )
        )

    return SubgraphResponse(
        center_node=target_id,
        node_count=len(result_nodes),
        edge_count=len(result_edges),
        nodes=result_nodes,
        edges=result_edges,
        total_direct_edges=total_direct_edges,
        total_connected_companies=total_connected_companies,
        total_matching=total_matching,
        has_more=has_more,
        limit=limit,
        offset=offset,
        filter_type=filter_type,
    )


@router.get(
    "/search",
    response_model=list[GraphNodeItem],
    summary="Search nodes in Knowledge Graph by name or ID",
)
def search_kg_nodes(
    q: str = Query(..., min_length=2, description="Search term"),
    limit: int = Query(20, ge=1, le=50),
    pipeline: FinGraphPipeline = Depends(get_pipeline),
) -> list[GraphNodeItem]:
    if not pipeline.graph_retriever or not pipeline.graph_retriever.graph:
        return []

    graph = pipeline.graph_retriever.graph
    lower_q = q.lower()

    # Pre-calculate degrees for matches
    edge_counts: dict[str, int] = defaultdict(int)
    for edge in graph.edges:
        edge_counts[edge.source] += 1
        edge_counts[edge.target] += 1

    matches: list[GraphNodeItem] = []

    for n in graph.nodes:
        name = (
            n.properties.get("canonical_name")
            or n.properties.get("name")
            or ""
        ).lower()
        if lower_q in n.node_id.lower() or lower_q in name:
            label = (
                n.properties.get("canonical_name")
                or n.properties.get("name")
                or n.node_id
            )
            matches.append(
                GraphNodeItem(
                    id=n.node_id,
                    label=label,
                    type=n.node_type,
                    properties=n.properties or {},
                    degree=edge_counts.get(n.node_id, 0),
                )
            )
            if len(matches) >= limit:
                break

    # Sort matches by degree descending
    matches.sort(key=lambda m: m.degree, reverse=True)
    return matches
