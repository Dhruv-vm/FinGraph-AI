from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# ============================================================
# CONFIGURATION
# ============================================================

GRAPH_PATH = Path(
    "data/processed/unified/aapl_temporal_graph.json"
)

SNAPSHOT_TIME = datetime(
    2026,
    8,
    21,
    tzinfo=timezone.utc,
)


# ============================================================
# DATETIME PARSER
# ============================================================

def parse_datetime(
    value: str | None,
) -> datetime | None:
    """
    Parse ISO/date strings into timezone-aware datetimes.
    """

    if not value:
        return None

    value = value.strip()

    if len(value) == 10:
        value = f"{value}T00:00:00+00:00"

    dt = datetime.fromisoformat(
        value.replace("Z", "+00:00")
    )

    if dt.tzinfo is None:
        dt = dt.replace(
            tzinfo=timezone.utc
        )

    return dt


# ============================================================
# LOAD GRAPH
# ============================================================

def load_graph(
    path: Path,
) -> dict[str, Any]:

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


# ============================================================
# TEMPORAL LEAKAGE VALIDATION
# ============================================================

def validate_temporal_leakage(
    graph: dict[str, Any],
    snapshot_time: datetime,
) -> None:

    nodes = graph.get(
        "nodes",
        [],
    )

    edges = graph.get(
        "edges",
        [],
    )

    print("=" * 70)
    print("TEMPORAL LEAKAGE VALIDATION")
    print("=" * 70)

    print(
        f"Snapshot time: {snapshot_time.isoformat()}"
    )

    print(
        f"Original nodes : {len(nodes)}"
    )

    print(
        f"Original edges : {len(edges)}"
    )

    # --------------------------------------------------------
    # FUTURE NODES
    # --------------------------------------------------------

    future_nodes = []

    for node in nodes:

        node_type = node.get(
            "node_type"
        )

        available_time = parse_datetime(
            node.get("available_time")
        )

        # Company/entity nodes are always available.
        if node_type == "company":
            continue

        if (
            available_time is not None
            and available_time > snapshot_time
        ):
            future_nodes.append(node)

    # --------------------------------------------------------
    # FUTURE EDGES
    # --------------------------------------------------------

    future_edges = []

    for edge in edges:

        available_time = parse_datetime(
            edge.get("available_time")
        )

        if (
            available_time is not None
            and available_time > snapshot_time
        ):
            future_edges.append(edge)

    # --------------------------------------------------------
    # SNAPSHOT NODE SET
    # --------------------------------------------------------

    snapshot_nodes = []

    for node in nodes:

        node_type = node.get(
            "node_type"
        )

        available_time = parse_datetime(
            node.get("available_time")
        )

        if node_type == "company":
            snapshot_nodes.append(node)
            continue

        if (
            available_time is not None
            and available_time <= snapshot_time
        ):
            snapshot_nodes.append(node)

    snapshot_node_ids = {
        node["node_id"]
        for node in snapshot_nodes
    }

    # --------------------------------------------------------
    # SNAPSHOT EDGES
    # --------------------------------------------------------

    snapshot_edges = []

    for edge in edges:

        source = edge.get(
            "source"
        )

        target = edge.get(
            "target"
        )

        available_time = parse_datetime(
            edge.get("available_time")
        )

        if source not in snapshot_node_ids:
            continue

        if target not in snapshot_node_ids:
            continue

        if available_time is None:
            continue

        if available_time > snapshot_time:
            continue

        snapshot_edges.append(edge)

    # --------------------------------------------------------
    # DANGLING EDGES
    # --------------------------------------------------------

    dangling_edges = []

    for edge in snapshot_edges:

        if (
            edge["source"]
            not in snapshot_node_ids
            or
            edge["target"]
            not in snapshot_node_ids
        ):
            dangling_edges.append(edge)

    # --------------------------------------------------------
    # INVALID AVAILABILITY ORDER
    # --------------------------------------------------------
    #
    # IMPORTANT:
    #
    # We DO NOT check:
    #
    #     event_time <= available_time
    #
    # because SEC filing event dates and SEC acceptance/
    # publication timestamps can legitimately fall on adjacent
    # calendar dates due to timestamp/time-zone semantics.
    #
    # The actual leakage invariant is:
    #
    #     available_time <= snapshot_time
    #
    # --------------------------------------------------------

    invalid_availability_nodes = []

    for node in snapshot_nodes:

        node_type = node.get(
            "node_type"
        )

        if node_type == "company":
            continue

        available_time = parse_datetime(
            node.get("available_time")
        )

        if available_time is None:
            invalid_availability_nodes.append(
                node
            )
            continue

        if available_time > snapshot_time:
            invalid_availability_nodes.append(
                node
            )

    invalid_availability_edges = []

    for edge in snapshot_edges:

        available_time = parse_datetime(
            edge.get("available_time")
        )

        if available_time is None:
            invalid_availability_edges.append(
                edge
            )
            continue

        if available_time > snapshot_time:
            invalid_availability_edges.append(
                edge
            )

    # --------------------------------------------------------
    # COMPANY NODES
    # --------------------------------------------------------

    company_nodes = [
        node
        for node in snapshot_nodes
        if node.get("node_type") == "company"
    ]

    # --------------------------------------------------------
    # RESULTS
    # --------------------------------------------------------

    print()
    print("1. FUTURE NODES INCLUDED")
    print("-" * 70)
    print(
        f"Count: {len(future_nodes)}"
    )

    print()
    print("2. FUTURE EDGES INCLUDED")
    print("-" * 70)
    print(
        f"Count: {len(future_edges)}"
    )

    print()
    print("3. DANGLING EDGES")
    print("-" * 70)
    print(
        f"Count: {len(dangling_edges)}"
    )

    print()
    print("4. COMPANY NODES")
    print("-" * 70)
    print(
        f"Count: {len(company_nodes)}"
    )

    print()
    print("5. INVALID AVAILABILITY")
    print("-" * 70)

    print(
        f"Nodes: {len(invalid_availability_nodes)}"
    )

    print(
        f"Edges: {len(invalid_availability_edges)}"
    )

    # --------------------------------------------------------
    # SNAPSHOT SUMMARY
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("SNAPSHOT SUMMARY")
    print("=" * 70)

    print(
        f"Snapshot nodes : {len(snapshot_nodes)}"
    )

    print(
        f"Snapshot edges : {len(snapshot_edges)}"
    )

    print()
    print("NODE TYPES")
    print("-" * 70)

    print(
        Counter(
            node.get("node_type")
            for node in snapshot_nodes
        )
    )

    print()
    print("EDGE RELATIONSHIPS")
    print("-" * 70)

    print(
        Counter(
            edge.get("relationship")
            for edge in snapshot_edges
        )
    )

    # --------------------------------------------------------
    # VALIDATION RESULT
    # --------------------------------------------------------

    failed = (
        len(future_nodes) > 0
        or
        len(future_edges) > 0
        or
        len(dangling_edges) > 0
        or
        len(invalid_availability_nodes) > 0
        or
        len(invalid_availability_edges) > 0
    )

    print()
    print("=" * 70)

    if failed:

        print(
            "TEMPORAL LEAKAGE VALIDATION: FAILED"
        )

        print("=" * 70)

        raise RuntimeError(
            "Temporal leakage validation failed."
        )

    print(
        "TEMPORAL LEAKAGE VALIDATION: PASSED"
    )

    print("=" * 70)


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    graph = load_graph(
        GRAPH_PATH
    )

    validate_temporal_leakage(
        graph,
        SNAPSHOT_TIME,
    )


if __name__ == "__main__":
    main()