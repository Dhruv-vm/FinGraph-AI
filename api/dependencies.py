"""
FastAPI Dependencies for FinGraph AI API.
"""

from __future__ import annotations

import asyncio
from typing import cast
from fastapi import Request

from api.services.pipeline_service import PipelineService
from scripts.run_fingraph import FinGraphPipeline


def get_pipeline(request: Request) -> FinGraphPipeline:
    """Retrieve the singleton FinGraphPipeline instance from application state."""
    pipeline = getattr(request.app.state, "pipeline", None)
    if pipeline is None:
        raise RuntimeError("FinGraphPipeline has not been initialized.")
    return cast(FinGraphPipeline, pipeline)


def get_pipeline_lock(request: Request) -> asyncio.Lock:
    """Retrieve the application pipeline concurrency lock."""
    lock = getattr(request.app.state, "pipeline_lock", None)
    if lock is None:
        raise RuntimeError("Pipeline concurrency lock has not been initialized.")
    return cast(asyncio.Lock, lock)


def get_pipeline_service(request: Request) -> PipelineService:
    """Dependency provider for PipelineService."""
    pipeline = get_pipeline(request)
    lock = get_pipeline_lock(request)
    return PipelineService(pipeline=pipeline, lock=lock)
