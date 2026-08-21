from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def save_json(
    data: Any,
    path: str | Path,
) -> Path:
    """Save data as formatted UTF-8 JSON."""

    output_path = Path(path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            data,
            file,
            indent=2,
            ensure_ascii=False,
        )

    return output_path


def load_json(
    path: str | Path,
) -> Any:
    """Load JSON data from disk."""

    input_path = Path(path)

    with input_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)
