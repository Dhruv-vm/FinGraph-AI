from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import uuid
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http import models
from sentence_transformers import SentenceTransformer


DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_COLLECTION = "fingraph_chunks"
DEFAULT_VECTOR_SIZE = 384


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: str
    score: float
    text: str
    payload: dict[str, Any]


class VectorStore:
    """Embedding and persistent Qdrant retrieval for FinGraph AI chunks."""

    def __init__(
        self,
        path: str | Path = "data/vector_store",
        model_name: str = DEFAULT_MODEL,
        collection_name: str = DEFAULT_COLLECTION,
    ) -> None:
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)

        self.model = SentenceTransformer(model_name)
        self.client = QdrantClient(path=str(self.path))
        self.collection_name = collection_name

    @property
    def vector_size(self) -> int:
        return self.model.get_embedding_dimension()

    def create_collection(self, recreate: bool = False) -> None:
        """Create the Qdrant collection if it does not exist."""
        exists = self.client.collection_exists(self.collection_name)

        if exists and recreate:
            self.client.delete_collection(self.collection_name)
            exists = False

        if not exists:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=self.vector_size,
                    distance=models.Distance.COSINE,
                ),
            )

    def upsert_chunks(
        self,
        chunks: list[dict[str, Any]],
        batch_size: int = 64,
    ) -> int:
        """Embed and persist document chunks."""
        if not chunks:
            return 0

        self.create_collection()

        total = 0

        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            texts = [chunk["text"] for chunk in batch]

            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                normalize_embeddings=True,
                show_progress_bar=False,
            )

            points = []

            for chunk, embedding in zip(batch, embeddings):
                payload = {
                    "chunk_id": chunk["chunk_id"],
                    "document_id": chunk["document_id"],
                    "text": chunk["text"],
                    "chunk_index": chunk.get("chunk_index"),
                    "section": chunk.get("section"),
                    "company": chunk.get("company"),
                    "source": chunk.get("source"),
                    "document_type": chunk.get("document_type"),
                    "publication_date": chunk.get("publication_date"),
                    "fiscal_period": chunk.get("fiscal_period"),
                    "event_time": chunk.get("event_time"),
                    "available_time": chunk.get("available_time"),
                    "ingested_at": chunk.get("ingested_at"),
                    "source_url": chunk.get("source_url"),
                    "source_reference": chunk.get("source_reference"),
                }

                points.append(
                    models.PointStruct(
                        id=str(uuid.uuid5(uuid.NAMESPACE_URL, chunk["chunk_id"])),
                        vector=embedding.tolist(),
                        payload=payload,
                    )
                )

            self.client.upsert(
                collection_name=self.collection_name,
                points=points,
            )

            total += len(points)

        return total

    def search(
        self,
        query: str,
        top_k: int = 5,
        reference_time: datetime | None = None,
    ) -> list[RetrievedChunk]:
        """Retrieve semantically similar chunks with optional temporal filtering."""
        if not self.client.collection_exists(self.collection_name):
            return []

        query_vector = self.model.encode(
            query,
            normalize_embeddings=True,
        ).tolist()

        query_filter = None

        if reference_time is not None:
            reference_iso = reference_time.isoformat()

            query_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="available_time",
                        range=models.DatetimeRange(
                            lte=reference_iso,
                        ),
                    )
                ]
            )

        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            query_filter=query_filter,
            limit=top_k,
            with_payload=True,
        ).points

        return [
            RetrievedChunk(
                chunk_id=str(result.payload.get("chunk_id", result.id)),
                score=float(result.score),
                text=result.payload.get("text", ""),
                payload=result.payload,
            )
            for result in results
        ]

    def count(self) -> int:
        """Return the number of indexed chunks."""
        if not self.client.collection_exists(self.collection_name):
            return 0

        return self.client.count(
            collection_name=self.collection_name,
            exact=True,
        ).count
