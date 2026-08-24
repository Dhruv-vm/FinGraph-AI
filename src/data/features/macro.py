from __future__ import annotations

from typing import Any

from src.data.graph.schema import TemporalGraph


def extract_macro_features(
    snapshot: TemporalGraph,
    company_id: str,
) -> dict[str, Any]:
    """
    Extract point-in-time macroeconomic features.

    Macro nodes are global economic observations, so they
    are not directly connected to a company. We therefore
    select the latest available observation for each macro
    series as of the snapshot.
    """

    macro_nodes = [
        node
        for node in snapshot.nodes
        if node.node_type == "macro"
    ]

    series: dict[str, list[Any]] = {}

    for node in macro_nodes:
        data = node.properties

        series_id = data.get("series_id")

        if not series_id:
            continue

        series.setdefault(series_id, []).append(node)

    latest_values: dict[str, Any] = {}

    for series_id, nodes in series.items():
        nodes.sort(
            key=lambda node: (
                node.event_time or "",
                node.available_time or "",
            )
        )

        latest = nodes[-1]
        data = latest.properties

        latest_values[series_id] = {
            "series_name": data.get("series_name"),
            "ticker": data.get("ticker"),
            "date": data.get("date"),
            "value": data.get("value"),
            "unit": data.get("unit"),
            "source": data.get("source"),
        }

    return {
        "company_id": company_id,
        "macro_series_count": len(latest_values),
        "macro_values": latest_values,
    }