# FinGraph AI — Backend API Documentation (Phase 8.1)

## Overview

The FinGraph AI Backend API exposes the verified Phase 7 GraphRAG and financial question-answering pipeline via production-grade FastAPI REST endpoints.

It provides complete programmatic access for web clients (such as the Next.js Research Terminal) to query SEC 10-K/10-Q disclosures and the semantic temporal knowledge graph while strictly preserving all fail-closed provenance checks, point-in-time temporal barriers, and refusal mechanisms.

---

## 1. Quick Start & Server Startup

### Prerequisites
- Active Python virtual environment (`.venv`) with dependencies installed (`fastapi`, `uvicorn`, `qdrant-client`, `sentence-transformers`, `ollama`).
- Local Ollama daemon running with the target model:
  ```bash
  ollama run qwen3.5:4b
  ```

### Start the FastAPI Server
From the repository root:
```bash
.venv/bin/uvicorn api.main:app --host 127.0.0.1 --port 8000
```

For development with hot reloading:
```bash
.venv/bin/uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload
```

Interactive API documentation:
- **Swagger UI**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **ReDoc**: [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)
- **OpenAPI JSON**: [http://127.0.0.1:8000/openapi.json](http://127.0.0.1:8000/openapi.json)

---

## 2. Configuration & Environment Variables

All settings are configured via environment variables or loaded from `.env`:

| Variable | Default Value | Description |
| :--- | :--- | :--- |
| `API_HOST` | `127.0.0.1` | API binding host |
| `API_PORT` | `8000` | API binding port |
| `CORS_ORIGINS` | `http://localhost:3000,http://127.0.0.1:3000` | Allowed origins for Next.js frontend |
| `KG_PATH` | `data/processed/unified/fingraph_semantic_kg.json` | Path to semantic Knowledge Graph JSON |
| `VECTOR_PATH` | `data/vector_store` | Path to local Qdrant vector storage |
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama daemon endpoint |
| `LLM_MODEL` | `qwen3.5:4b` | Default LLM model for answer generation |
| `HOP_MODE` | `adaptive` | Hop-depth mode (`adaptive` for Phase 7 baseline) |
| `BALANCED_FUSION` | `true` | Entity-balanced RRF fusion for comparative queries |
| `MIN_PER_ENTITY` | `3` | Candidate allocation per required entity in fusion |
| `REGULATORY_LINKING` | `true` | Structured regulatory/jurisdictional entity linking |

---

## 3. Architecture & Concurrency Safety

### Singleton Lifespan Management
Embedded Qdrant maintains an exclusive file lock on the SQLite storage folder (`data/vector_store`). Creating multiple `FinGraphPipeline` instances concurrently within the same or child processes causes file lock errors (`RuntimeError: Storage folder ... is already accessed`).

The API application manages a single instance of `FinGraphPipeline` through FastAPI's async lifespan context manager (`@asynccontextmanager`). The pipeline is initialized once upon server boot and gracefully closed when the server shuts down.

### Concurrency Locking
To prevent race conditions and GPU/CPU thrashing during local Ollama inference, the API serializes concurrent pipeline calls using an `asyncio.Lock` and executes the blocking retrieval/generation code in a worker thread (`asyncio.to_thread`).

---

## 4. REST Endpoints Reference

### 4.1 Root Diagnostic Endpoint
- **URL**: `GET /`
- **Description**: Returns API service metadata and route discovery links.

**Sample Response**:
```json
{
  "name": "FinGraph AI Research API",
  "version": "0.1.0",
  "docs": "/docs",
  "openapi": "/openapi.json",
  "api_v1": {
    "query": "/api/v1/query",
    "health": "/api/v1/health",
    "benchmark_demo": "/api/v1/benchmark/demo-questions"
  },
  "status": "online"
}
```

---

### 4.2 Health Check Endpoint
- **URL**: `GET /api/v1/health`
- **Description**: Deep component inspection across the FastAPI application, FinGraphPipeline, persistent Qdrant vector store, semantic Knowledge Graph index, and local Ollama LLM.

**Sample Response**:
```json
{
  "status": "healthy",
  "version": "0.1.0",
  "timestamp": "2026-10-31T08:25:00.123456Z",
  "pipeline": {
    "status": "healthy",
    "message": "FinGraphPipeline active and ready",
    "details": {
      "hop_mode": "adaptive",
      "balanced_fusion": true,
      "min_per_entity": 3,
      "regulatory_linking": true
    }
  },
  "qdrant": {
    "status": "healthy",
    "message": "Qdrant vector collection accessible",
    "details": {
      "collection": "fingraph_chunks",
      "points_count": 8181,
      "vector_size": 384,
      "storage_path": "/Users/dhruv/dhruv files/FinGraph-AI/data/vector_store"
    }
  },
  "knowledge_graph": {
    "status": "healthy",
    "message": "Knowledge Graph loaded (10054 nodes, 9822 edges)",
    "details": {
      "node_count": 10054,
      "edge_count": 9822,
      "kg_path": "/Users/dhruv/dhruv files/FinGraph-AI/data/processed/unified/fingraph_semantic_kg.json",
      "regulatory_linking": true
    }
  },
  "llm": {
    "status": "healthy",
    "message": "Ollama reachable",
    "details": {
      "host": "http://localhost:11434",
      "model": "qwen3.5:4b",
      "model_present": true,
      "available_models": ["qwen3.5:4b"]
    }
  }
}
```

---

### 4.3 Demonstration Benchmark Questions
- **URL**: `GET /api/v1/benchmark/demo-questions`
- **Description**: Returns 5 verified demonstration questions from the Phase 7 benchmark covering factual, comparative, multi-hop, temporal, and unanswerable refusal scenarios. Each question contains its audited Phase 7 metrics alongside the ground truth.

**Questions Included**:
1. `FSH-07`: "What major video gaming and entertainment company did Microsoft acquire?" (Factual single-hop)
2. `CDC-04`: "How do NVIDIA and Microsoft disclose their respective relationships with AI cloud computing infrastructure?" (Cross-document comparative)
3. `RMH-07`: "What gaming company acquired by Microsoft had its transaction challenged by competition regulators?" (Relational multi-hop)
4. `TPT-01`: "What financial or cloud service revenue did Microsoft disclose as of May 1, 2025?" (Temporal point-in-time)
5. `TPT-02`: "What risks affected NVIDIA as of June 1, 2025?" (Temporal point-in-time safe refusal)

---

### 4.4 Live FinGraph Query Endpoint
- **URL**: `POST /api/v1/query`
- **Description**: Executes end-to-end question answering through the active GraphRAG pipeline.

#### Request Schema (`QueryRequest`):
```json
{
  "query": "What major video gaming and entertainment company did Microsoft acquire?",
  "reference_time": null,
  "top_k": 10,
  "candidate_pool_size": 20
}
```

#### Response Schema (`QueryResponse`):
```json
{
  "query": "What major video gaming and entertainment company did Microsoft acquire?",
  "decision": "answer",
  "answer": "Microsoft acquired Activision Blizzard.",
  "insufficient_evidence": false,
  "confidence": 0.98,
  "reasoning_summary": "Direct acquisition relationship identified in knowledge graph and SEC 10-K filing disclosures.",
  "citations": [
    {
      "evidence_id": "hybrid:doc:sec:6a50d47d6a8dc43d:chunk:160",
      "document_id": "doc:sec:6a50d47d6a8dc43d",
      "chunk_id": "chunk:160",
      "source_type": "hybrid",
      "available_time": "2024-07-30T00:00:00+00:00",
      "text_snippet": "On October 13, 2023, Microsoft completed its acquisition of Activision Blizzard, Inc."
    }
  ],
  "evidence": [
    {
      "evidence_id": "hybrid:doc:sec:6a50d47d6a8dc43d:chunk:160",
      "source_type": "hybrid",
      "score": 0.9412,
      "rank": 1,
      "text": "...",
      "document_id": "doc:sec:6a50d47d6a8dc43d",
      "chunk_id": "chunk:160",
      "available_time": "2024-07-30T00:00:00+00:00",
      "provenance_verified": true,
      "metadata": {}
    }
  ],
  "graph_paths": [
    ["company:MSFT", "ACQUIRED", "Activision Blizzard"]
  ],
  "sufficiency": {
    "sufficient": true,
    "reason": "All query entities and relationship requirements satisfied.",
    "missing_entities": [],
    "missing_relationships": [],
    "required_hops": 1,
    "observed_hops": 1,
    "evidence_count": 8,
    "temporal_valid": true,
    "provenance_complete": true
  },
  "temporal_validation": {
    "temporal_valid": true,
    "reference_time": null,
    "context_items_count": 5,
    "future_items_count": 0,
    "feir": 0.0
  },
  "validation_status": {
    "is_valid": true,
    "errors": []
  },
  "sources": [
    "doc:sec:6a50d47d6a8dc43d",
    "doc:sec:076412151b515b96"
  ],
  "retrieval_rounds": 1,
  "latency_ms": 1824.50,
  "timestamp": "2026-10-31T08:26:12.345678Z"
}
```

---

### 4.4 Knowledge Graph Exploration Endpoints

#### 4.4.1 List Core Companies
- **URL**: `GET /api/v1/graph/companies?limit=50`
- **Description**: Returns corporate entities indexed in the knowledge graph with priority ordering for core S&P universe tickers (NVDA, MSFT, AAPL, AMZN, GOOGL, META, TSLA, JPM, XOM, JNJ) followed by degree-sorted corporate nodes.

#### 4.4.2 Search Entities
- **URL**: `GET /api/v1/graph/search?q={query}&limit=20`
- **Description**: Real-time substring and token matching across entity IDs, canonical names, and entity types with exact node degrees.

#### 4.4.3 Subgraph Neighborhood Extraction
- **URL**: `GET /api/v1/graph/subgraph?node_id={node_id}&max_hops=1&max_nodes=50`
- **Description**: Extracts localized ego-network for graph visualization and inspector analysis. Supports bare tickers (`MSFT`), prefixed IDs (`company:MSFT`), and canonical names. Returns nodes, directional edges with predicate semantics, and connected node degrees.

---

## 5. cURL Verification Examples

### Health Check:
```bash
curl -X GET http://127.0.0.1:8000/api/v1/health
```

### Fetch Demonstration Questions:
```bash
curl -X GET http://127.0.0.1:8000/api/v1/benchmark/demo-questions
```

### Live Query (Factual Single-Hop):
```bash
curl -X POST http://127.0.0.1:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What major video gaming and entertainment company did Microsoft acquire?", "top_k": 5}'
```

### Live Query (Temporal Point-in-Time Refusal):
```bash
curl -X POST http://127.0.0.1:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What risks affected NVIDIA as of June 1, 2025?", "reference_time": "2025-06-01T00:00:00+00:00"}'
```
*(Returns `decision: "refuse"` with refusal message and zero future evidence leakage).*
