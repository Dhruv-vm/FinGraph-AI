from __future__ import annotations

import json
from pathlib import Path

from src.retrieval.vector import VectorStore


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHUNK_DIR = PROJECT_ROOT / "data" / "processed" / "chunks"
VECTOR_DIR = PROJECT_ROOT / "data" / "vector_store"

BATCH_SIZE = 64


def load_chunks() -> list[dict]:
    """Load all processed chunk JSON files."""
    chunks: list[dict] = []

    for path in sorted(CHUNK_DIR.glob("*.json")):
        with path.open("r", encoding="utf-8") as f:
            file_chunks = json.load(f)

        if not isinstance(file_chunks, list):
            raise ValueError(f"Expected list of chunks in {path}")

        chunks.extend(file_chunks)

    return chunks


def main() -> None:
    print("=== FinGraph AI Vector Index Builder ===")
    print(f"Chunk directory: {CHUNK_DIR}")
    print(f"Vector store:    {VECTOR_DIR}")

    chunks = load_chunks()

    print(f"\nLoaded chunks: {len(chunks)}")

    if not chunks:
        raise RuntimeError("No chunks found.")

    store = VectorStore(path=VECTOR_DIR)

    print(f"Embedding model: {store.model}")
    print(f"Vector dimension: {store.vector_size}")

    indexed = store.upsert_chunks(
        chunks,
        batch_size=BATCH_SIZE,
    )

    print("\n=== Index Complete ===")
    print(f"Chunks indexed: {indexed}")
    print(f"Qdrant count:   {store.count()}")


if __name__ == "__main__":
    main()
