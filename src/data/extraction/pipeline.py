from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .llm import ExtractionRequest, LLMClient, extract_document


class ExtractionPipeline:
    """Run structured extraction over processed document chunks."""

    def __init__(
        self,
        client: LLMClient,
        output_dir: str | Path = "data/processed/extractions",
    ) -> None:
        self.client = client
        self.output_dir = Path(output_dir)

    def _build_request(
        self,
        chunk: dict[str, Any],
    ) -> ExtractionRequest:
        return ExtractionRequest(
            document_id=str(chunk["document_id"]),
            text=str(chunk["text"]),
            available_time=chunk.get("available_time"),
            publication_date=chunk.get("publication_date"),
            metadata={
                "chunk_id": chunk.get("chunk_id"),
                "company": chunk.get("company"),
                "source": chunk.get("source"),
                "document_type": chunk.get("document_type"),
                "section": chunk.get("section"),
                "fiscal_period": chunk.get("fiscal_period"),
                "source_url": chunk.get("source_url"),
                "source_reference": chunk.get("source_reference"),
            },
        )

    def extract_chunk(
        self,
        chunk: dict[str, Any],
    ) -> dict[str, Any]:
        """Extract and attach chunk-level provenance."""
        request = self._build_request(chunk)
        result = extract_document(self.client, request)

        output = result.to_dict()

        output["chunk_id"] = chunk.get("chunk_id")
        output["chunk_index"] = chunk.get("chunk_index")
        output["provenance"] = {
            "document_id": chunk.get("document_id"),
            "chunk_id": chunk.get("chunk_id"),
            "source": chunk.get("source"),
            "source_url": chunk.get("source_url"),
            "source_reference": chunk.get("source_reference"),
            "publication_date": chunk.get("publication_date"),
            "available_time": chunk.get("available_time"),
            "section": chunk.get("section"),
        }

        return output

    def extract_document(
        self,
        chunks: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Extract all chunks belonging to one document."""
        return [
            self.extract_chunk(chunk)
            for chunk in chunks
        ]

    def save_document_extraction(
        self,
        document_id: str,
        results: list[dict[str, Any]],
    ) -> Path:
        """Persist extraction results for one document."""
        self.output_dir.mkdir(parents=True, exist_ok=True)

        safe_id = str(document_id).replace("/", "_")
        output_path = self.output_dir / f"{safe_id}.json"

        output_path.write_text(
            json.dumps(results, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        return output_path

    def process_chunk_file(
        self,
        chunk_file: str | Path,
    ) -> Path:
        """Process one existing chunk JSON file."""
        chunk_path = Path(chunk_file)

        chunks = json.loads(
            chunk_path.read_text(encoding="utf-8")
        )

        if not isinstance(chunks, list):
            raise ValueError(
                f"Chunk file must contain a JSON list: {chunk_path}"
            )

        if not chunks:
            raise ValueError(
                f"Chunk file is empty: {chunk_path}"
            )

        document_id = chunks[0]["document_id"]

        for chunk in chunks:
            if chunk.get("document_id") != document_id:
                raise ValueError(
                    "All chunks in a chunk file must belong "
                    "to the same document."
                )

        results = self.extract_document(chunks)

        return self.save_document_extraction(
            document_id=document_id,
            results=results,
        )
