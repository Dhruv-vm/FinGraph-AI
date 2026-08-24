from __future__ import annotations

import json
from pathlib import Path

from src.data.graph.builder import build_graph


EVENTS_PATH = Path(
    "data/processed/unified/aapl_unified_events.json"
)

OUTPUT_PATH = Path(
    "data/processed/unified/aapl_temporal_graph.json"
)


def main() -> None:
    with EVENTS_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        events = json.load(file)

    graph = build_graph(events)

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            graph.to_dict(),
            file,
            indent=2,
            ensure_ascii=False,
        )

    print("=" * 80)
    print("AAPL TEMPORAL GRAPH REBUILT")
    print("=" * 80)
    print("Events:", len(events))
    print("Nodes:", len(graph.nodes))
    print("Edges:", len(graph.edges))
    print("Saved:", OUTPUT_PATH)


if __name__ == "__main__":
    main()