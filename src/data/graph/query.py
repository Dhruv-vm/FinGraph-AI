from __future__ import annotations

from typing import Any

from src.data.graph.schema import TemporalGraph
from src.data.graph.snapshot import build_temporal_snapshot


def query_company_snapshot(
    graph: TemporalGraph | dict[str, Any],
    ticker: str,
    as_of: str,
) -> dict[str, Any]:
    """
    Return all information available for a company
    at a specific point in time.
    """

    snapshot = build_temporal_snapshot(
        graph,
        as_of,
    )

    company_id = f"company:{ticker.upper()}"

    company = next(
        (
            node
            for node in snapshot.nodes
            if node.node_id == company_id
        ),
        None,
    )

    if company is None:
        raise ValueError(
            f"Company {ticker} not found in snapshot"
        )

    connected_node_ids = set()

    for edge in snapshot.edges:
        if edge.source == company_id:
            connected_node_ids.add(edge.target)

        elif edge.target == company_id:
            connected_node_ids.add(edge.source)

    related_nodes = [
        node
        for node in snapshot.nodes
        if node.node_id in connected_node_ids
    ]

    return {
        "ticker": ticker.upper(),
        "as_of": as_of,
        "company": company.to_dict(),
        "related_nodes": [
            node.to_dict()
            for node in related_nodes
        ],
        "related_edges": [
            edge.to_dict()
            for edge in snapshot.edges
            if edge.source == company_id
            or edge.target == company_id
        ],
        "node_count": len(related_nodes),
        "edge_count": sum(
            1
            for edge in snapshot.edges
            if edge.source == company_id
            or edge.target == company_id
        ),
    }