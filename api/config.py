"""
FinGraph AI — API Configuration & Environment Settings.
"""

from __future__ import annotations

import os
from pathlib import Path
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]


class Settings(BaseModel):
    api_title: str = "FinGraph AI Research API"
    api_version: str = "0.1.0"
    api_description: str = (
        "REST API exposing the FinGraph AI GraphRAG, temporal reasoning, "
        "and provenance-verified financial question-answering pipeline."
    )
    api_host: str = Field(default_factory=lambda: os.getenv("API_HOST", "127.0.0.1"))
    api_port: int = Field(default_factory=lambda: int(os.getenv("API_PORT", "8000")))

    # CORS configuration
    cors_origins: list[str] = Field(
        default_factory=lambda: [
            origin.strip()
            for origin in os.getenv(
                "CORS_ORIGINS",
                "http://localhost:3000,http://127.0.0.1:3000,http://localhost:8000,http://127.0.0.1:8000",
            ).split(",")
            if origin.strip()
        ]
    )

    # Core Pipeline paths and settings
    kg_path: Path = Field(
        default_factory=lambda: Path(
            os.getenv("KG_PATH", str(ROOT / "data" / "processed" / "unified" / "fingraph_semantic_kg.json"))
        )
    )
    vector_path: Path = Field(
        default_factory=lambda: Path(os.getenv("VECTOR_PATH", str(ROOT / "data" / "vector_store")))
    )
    ollama_host: str = Field(default_factory=lambda: os.getenv("OLLAMA_HOST", "http://localhost:11434"))
    llm_model: str = Field(default_factory=lambda: os.getenv("LLM_MODEL", "qwen3.5:4b"))

    # Phase 7 Reasoning & Fusion configuration
    hop_mode: str = Field(default_factory=lambda: os.getenv("HOP_MODE", "adaptive"))
    balanced_fusion: bool = Field(
        default_factory=lambda: os.getenv("BALANCED_FUSION", "true").lower() in ("true", "1", "yes")
    )
    min_per_entity: int = Field(default_factory=lambda: int(os.getenv("MIN_PER_ENTITY", "3")))
    regulatory_linking: bool = Field(
        default_factory=lambda: os.getenv("REGULATORY_LINKING", "true").lower() in ("true", "1", "yes")
    )


settings = Settings()
