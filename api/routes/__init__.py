"""
FinGraph AI API Routes.
"""

from api.routes.benchmark import router as benchmark_router
from api.routes.corpus import router as corpus_router
from api.routes.evidence import router as evidence_router
from api.routes.graph import router as graph_router
from api.routes.health import router as health_router
from api.routes.live import router as live_router
from api.routes.query import router as query_router

__all__ = [
    "query_router",
    "health_router",
    "benchmark_router",
    "graph_router",
    "evidence_router",
    "corpus_router",
    "live_router",
]
