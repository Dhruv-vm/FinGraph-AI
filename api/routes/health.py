"""
Health check endpoints for FinGraph AI API.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import logging
import urllib.request
import json
from fastapi import APIRouter, Depends, Request

from api.config import settings
from api.dependencies import get_pipeline
from api.schemas.health import ComponentHealth, HealthResponse
from scripts.run_fingraph import FinGraphPipeline

logger = logging.getLogger("fingraph_api.routes.health")

router = APIRouter(prefix="/health", tags=["Health"])


def _check_ollama(host: str, model: str) -> ComponentHealth:
    """Check Ollama service reachability and model availability."""
    url = f"{host.rstrip('/')}/api/tags"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "FinGraph-API/0.1.0"})
        with urllib.request.urlopen(req, timeout=2.5) as resp:
            if resp.status == 200:
                body = json.loads(resp.read().decode())
                models = [m.get("name", "") for m in body.get("models", [])]
                model_found = any(model in m or m.startswith(model) for m in models)
                return ComponentHealth(
                    status="healthy" if model_found else "degraded",
                    message="Ollama reachable" + ("" if model_found else f" (model {model} not found)"),
                    details={
                        "host": host,
                        "model": model,
                        "model_present": model_found,
                        "available_models": models,
                    },
                )
            return ComponentHealth(
                status="degraded",
                message=f"Ollama returned status {resp.status}",
                details={"host": host, "status_code": resp.status},
            )
    except Exception as exc:
        return ComponentHealth(
            status="unavailable",
            message=f"Cannot connect to Ollama host: {exc}",
            details={"host": host, "error": str(exc)},
        )


@router.get(
    "",
    response_model=HealthResponse,
    summary="Deep health check across all FinGraph components",
    description=(
        "Inspects runtime health of the FastAPI application, FinGraphPipeline, "
        "persistent Qdrant vector store, semantic Knowledge Graph index, and local Ollama LLM."
    ),
)
async def get_health(
    request: Request,
    pipeline: FinGraphPipeline = Depends(get_pipeline),
) -> HealthResponse:
    # 1. FinGraph Pipeline Check
    pipeline_health = ComponentHealth(
        status="healthy",
        message="FinGraphPipeline active and ready",
        details={
            "hop_mode": getattr(pipeline, "hop_mode", "adaptive"),
            "balanced_fusion": getattr(pipeline, "balanced_fusion", True),
            "min_per_entity": getattr(pipeline, "min_per_entity", 3),
            "regulatory_linking": getattr(pipeline, "regulatory_linking", True),
        },
    )

    # 2. Qdrant Vector Store Check
    qdrant_health: ComponentHealth
    try:
        if pipeline.vector_store and pipeline.vector_store.client:
            col_info = pipeline.vector_store.client.get_collection(
                pipeline.vector_store.collection_name
            )
            qdrant_health = ComponentHealth(
                status="healthy",
                message="Qdrant vector collection accessible",
                details={
                    "collection": pipeline.vector_store.collection_name,
                    "points_count": getattr(col_info, "points_count", 0),
                    "vector_size": pipeline.vector_store.vector_size,
                    "storage_path": str(pipeline.vector_path),
                },
            )
        else:
            qdrant_health = ComponentHealth(
                status="unavailable",
                message="Vector store is not initialized or path missing.",
                details={"storage_path": str(pipeline.vector_path)},
            )
    except Exception as exc:
        qdrant_health = ComponentHealth(
            status="degraded",
            message=f"Qdrant collection inspection error: {exc}",
            details={"error": str(exc)},
        )

    # 3. Knowledge Graph Check
    kg_health: ComponentHealth
    try:
        if pipeline.graph_retriever and pipeline.graph_retriever.graph:
            node_cnt = len(pipeline.graph_retriever.graph.nodes)
            edge_cnt = len(pipeline.graph_retriever.graph.edges)
            kg_health = ComponentHealth(
                status="healthy" if (node_cnt > 0 and edge_cnt > 0) else "degraded",
                message=f"Knowledge Graph loaded ({node_cnt} nodes, {edge_cnt} edges)",
                details={
                    "node_count": node_cnt,
                    "edge_count": edge_cnt,
                    "kg_path": str(pipeline.kg_path),
                    "regulatory_linking": getattr(pipeline.graph_retriever, "regulatory_linking", True),
                },
            )
        else:
            kg_health = ComponentHealth(
                status="unavailable",
                message="Knowledge Graph graph_retriever not initialized.",
                details={"kg_path": str(pipeline.kg_path)},
            )
    except Exception as exc:
        kg_health = ComponentHealth(
            status="degraded",
            message=f"Knowledge Graph error: {exc}",
            details={"error": str(exc)},
        )

    # 4. Ollama LLM Check (run in thread to not block)
    llm_health = await asyncio.to_thread(_check_ollama, settings.ollama_host, settings.llm_model)

    # Calculate overall status
    component_statuses = [
        pipeline_health.status,
        qdrant_health.status,
        kg_health.status,
        llm_health.status,
    ]
    if all(s == "healthy" for s in component_statuses):
        overall_status = "healthy"
    elif any(s == "unavailable" for s in component_statuses):
        overall_status = "unavailable"
    else:
        overall_status = "degraded"

    return HealthResponse(
        status=overall_status,
        version=settings.api_version,
        timestamp=datetime.now(timezone.utc).isoformat(),
        pipeline=pipeline_health,
        qdrant=qdrant_health,
        knowledge_graph=kg_health,
        llm=llm_health,
    )
