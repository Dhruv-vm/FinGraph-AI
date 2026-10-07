from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.data.graph.schema import TemporalGraph


def save_graph(
    graph: TemporalGraph,
    path: str | Path,
    indent: int = 2,
) -> Path:
    """Save a TemporalGraph to a JSON file."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(
            graph.to_dict(),
            handle,
            indent=indent,
            ensure_ascii=False,
        )

    return output_path


def load_graph(path: str | Path) -> TemporalGraph:
    """Load a TemporalGraph from a JSON file."""
    input_path = Path(path)

    if not input_path.exists():
        raise FileNotFoundError(f"Knowledge graph file not found: {input_path}")

    with input_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    if not isinstance(data, dict):
        raise ValueError(
            f"Knowledge graph file must contain a JSON object: {input_path}"
        )

    return TemporalGraph.from_dict(data)
