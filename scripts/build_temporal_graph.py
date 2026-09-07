from __future__ import annotations

import json
import sys
from pathlib import Path


# ============================================================
# PROJECT ROOT
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ============================================================
# IMPORTS
# ============================================================

from src.data.graph.builder import build_graph


# ============================================================
# PATHS
# ============================================================

EVENTS_PATH = (
    ROOT
    / "data"
    / "processed"
    / "unified"
    / "aapl_unified_events.json"
)

OUTPUT_PATH = (
    ROOT
    / "data"
    / "processed"
    / "unified"
    / "aapl_temporal_graph.json"
)


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    # --------------------------------------------------------
    # Load unified events
    # --------------------------------------------------------

    with EVENTS_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        events = json.load(file)

    # --------------------------------------------------------
    # Build temporal graph
    # --------------------------------------------------------

    graph = build_graph(events)

    # --------------------------------------------------------
    # Ensure output directory exists
    # --------------------------------------------------------

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Save graph
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print("=" * 80)
    print("AAPL TEMPORAL GRAPH REBUILT")
    print("=" * 80)
    print("Events:", len(events))
    print("Nodes:", len(graph.nodes))
    print("Edges:", len(graph.edges))
    print("Saved:", OUTPUT_PATH)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()