"""
FinGraph AI FastAPI Application Entry Point.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timezone
import logging
import time
import uuid
from typing import AsyncGenerator

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.config import settings
from api.routes import (
    benchmark_router,
    corpus_router,
    evidence_router,
    graph_router,
    health_router,
    query_router,
    live_router,
)
from scripts.run_fingraph import FinGraphPipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("fingraph_api.main")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Application lifespan manager.
    Initializes the singleton FinGraphPipeline instance and concurrency lock on startup,
    and safely closes persistent connections on shutdown.
    """
    logger.info("Initializing FinGraph AI FastAPI backend...")
    logger.info(
        "Config: KG=%s, VectorStore=%s, LLM=%s (%s), hop_mode=%s, balanced_fusion=%s",
        settings.kg_path,
        settings.vector_path,
        settings.llm_model,
        settings.ollama_host,
        settings.hop_mode,
        settings.balanced_fusion,
    )

    # Allow injection/mocking for tests: only instantiate if not already present
    if not hasattr(app.state, "pipeline") or app.state.pipeline is None:
        logger.info("Loading FinGraphPipeline (VectorStore, Knowledge Graph, LLM generator)...")
        pipeline = FinGraphPipeline(
            kg_path=settings.kg_path,
            vector_path=settings.vector_path,
            model=settings.llm_model,
            hop_mode=settings.hop_mode,
            balanced_fusion=settings.balanced_fusion,
            min_per_entity=settings.min_per_entity,
            regulatory_linking=settings.regulatory_linking,
        )
        app.state.pipeline = pipeline
    else:
        logger.info("Using pre-configured FinGraphPipeline on app.state.pipeline.")

    if not hasattr(app.state, "pipeline_lock") or app.state.pipeline_lock is None:
        app.state.pipeline_lock = asyncio.Lock()

    logger.info("FinGraph AI backend initialized successfully and ready for queries.")

    yield

    logger.info("Shutting down FinGraph AI backend...")
    pipeline = getattr(app.state, "pipeline", None)
    if pipeline and getattr(pipeline, "vector_store", None):
        client = getattr(pipeline.vector_store, "client", None)
        if client and hasattr(client, "close"):
            try:
                client.close()
                logger.info("Closed Qdrant client connection.")
            except Exception as exc:
                logger.warning("Error closing Qdrant client: %s", exc)


def create_app() -> FastAPI:
    """FastAPI Application Factory."""
    app = FastAPI(
        title=settings.api_title,
        version=settings.api_version,
        description=settings.api_description,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    # CORS configuration for frontend
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Request ID and Timing Middleware
    @app.middleware("http")
    async def request_timing_and_id_middleware(request: Request, call_next):
        req_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        start_time = time.perf_counter()

        logger.info("Incoming request: %s %s [req_id=%s]", request.method, request.url.path, req_id)
        response = await call_next(request)

        duration_ms = (time.perf_counter() - start_time) * 1000.0
        response.headers["X-Request-ID"] = req_id
        response.headers["X-Response-Time-Ms"] = f"{duration_ms:.2f}"

        logger.info(
            "Completed request: %s %s -> status %d (%.2f ms) [req_id=%s]",
            request.method,
            request.url.path,
            response.status_code,
            duration_ms,
            req_id,
        )
        return response

    # Global Validation Error Handler
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            content={
                "error": "ValidationError",
                "detail": exc.errors(),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )

    # Global Exception Handler
    @app.exception_handler(Exception)
    async def generic_exception_handler(request: Request, exc: Exception):
        logger.exception("Unhandled server exception on %s %s: %s", request.method, request.url.path, exc)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": "InternalServerError",
                "detail": str(exc),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )

    # Mount API v1 Routers
    api_prefix = "/api/v1"
    app.include_router(query_router, prefix=api_prefix)
    app.include_router(health_router, prefix=api_prefix)
    app.include_router(benchmark_router, prefix=api_prefix)
    app.include_router(graph_router, prefix=api_prefix)
    app.include_router(evidence_router, prefix=api_prefix)
    app.include_router(corpus_router, prefix=api_prefix)
    app.include_router(live_router, prefix=api_prefix)

    # Root diagnostic endpoint
    @app.get("/", tags=["Diagnostic"], summary="API Root metadata")
    async def root():
        return {
            "name": settings.api_title,
            "version": settings.api_version,
            "docs": "/docs",
            "openapi": "/openapi.json",
            "api_v1": {
                "query": "/api/v1/query",
                "health": "/api/v1/health",
                "benchmark_demo": "/api/v1/benchmark/demo-questions",
                "benchmark_summary": "/api/v1/benchmark/summary",
                "graph_companies": "/api/v1/graph/companies",
                "graph_subgraph": "/api/v1/graph/subgraph",
                "graph_search": "/api/v1/graph/search",
                "evidence_chunks": "/api/v1/evidence/chunks",
                "corpus_filings": "/api/v1/corpus/filings",
                "benchmark_per_question": "/api/v1/benchmark/per-question",
                "benchmark_parametric_leakage": "/api/v1/benchmark/parametric-leakage",
            },
            "status": "online",
        }

    return app


app = create_app()
