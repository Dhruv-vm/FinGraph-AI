# FinGraph AI — Repository Architecture & Component Audit

**Audit Date:** 2026-11-01  
**Project:** FinGraph AI  
**Research Title:** *FinGraph AI: A Temporal Knowledge Graph and Hybrid GraphRAG Framework for Multi-Hop Financial Question Answering*  
**Auditor:** Multidisciplinary Research & Reproducibility Team  

---

## 1. Executive Summary

This document provides a comprehensive code-level discovery and verification audit of the FinGraph AI repository. The audit evaluates whether the system's actual implementation matches the documented research methodology, verifies lines of code, identifies discrepancies between claims and reality, and catalogues implemented, partially implemented, and missing components.

---

## 2. Repository Directory Topology

```
FinGraph-AI/
├── api/                             # FastAPI Backend & Integration Endpoints
│   ├── main.py                     # API app definition, lifespan, CORS, and health router
│   ├── dependencies.py             # FinGraphPipeline singleton lifecycle
│   ├── config.py                   # API configuration settings
│   ├── routes/                     # Specialized REST route controllers
│   │   ├── query.py                # Single-hop & multi-hop QA query endpoint
│   │   ├── graph.py                # KG exploration, neighborhood, and entity search
│   │   ├── evidence.py             # Evidence inspection and provenance retrieval
│   │   ├── benchmark.py            # Phase 7 benchmark demo questions & metrics
│   │   └── health.py               # System diagnostic & readiness checks
│   └── schemas/                    # Pydantic v2 data models for API I/O
├── benchmark/                       # Question Answering Benchmarks
│   ├── questions/                  # Golden benchmark test datasets (Phase 7: 64 Qs)
│   └── templates/                  # Question generation schema templates
├── configs/                         # YAML/JSON Pipeline & Model Configurations
├── data/                            # Raw, Processed, and Persistent Datasets
│   ├── raw/sec/filings/            # Raw SEC EDGAR HTML/HTM submissions (99 filings)
│   ├── processed/
│   │   ├── chunks/                 # 99 chunked JSON files (8,181 total semantic chunks)
│   │   ├── unified/
│   │   │   ├── fingraph_semantic_kg.json   # 10,054 nodes, 9,822 edges multi-company KG
│   │   │   ├── aapl_temporal_graph.json    # 1,495 nodes, 660 edges single-company TG
│   │   │   ├── aapl_training_dataset.json  # 354 snapshot rows for walk-forward ML
│   │   │   └── aapl_model_comparison*.json # Predictive ML comparison records
│   │   └── sec_corpus_manifest.json        # Ingestion status and metadata manifest
│   └── vector_store/               # Embedded Qdrant local SQLite storage (8,181 vectors)
├── docs/                            # Research & API Documentation
│   ├── API.md                      # REST endpoint specification
│   ├── ARCHITECTURE.md             # Theoretical system design
│   └── RESEARCH_NOTES.md           # Internal experimentation scratchpad
├── frontend/                        # Next.js 14 Research Terminal Frontend
│   ├── src/
│   │   ├── app/                    # Next.js App Router (7 distinct views)
│   │   │   ├── layout.tsx          # Master shell, sidebar, and status ribbon
│   │   │   ├── page.tsx            # Executive Overview & KPI Dashboard
│   │   │   ├── query/page.tsx      # Interactive GraphRAG QA Terminal
│   │   │   ├── graph/page.tsx      # Radial Focal Knowledge Graph Explorer
│   │   │   ├── evidence/page.tsx   # Evidence Provenance & Chunk Inspector
│   │   │   ├── evaluation/page.tsx # Phase 7 Benchmark Lab & Ablation Charts
│   │   │   └── architecture/page.tsx # Interactive Pipeline Architecture View
│   │   ├── components/             # Reusable UI & Graph Visualization widgets
│   │   └── lib/                    # API client, TypeScript types, and utilities
├── results/                         # Empirical Evaluation Logs and Artifacts
│   ├── phase5/                     # Baseline RAG & early sufficiency ablations
│   ├── phase6/                     # Attribution hardening & retrieval depth ablations
│   └── phase7/                     # Verified Phase 7 benchmark (64 questions, K=8)
├── scripts/                         # Production & Experimentation CLI Scripts
│   ├── run_fingraph.py             # Standalone CLI for pipeline execution
│   ├── run_phase7_evaluation.py    # Automated evaluation harness for Phase 7
│   ├── build_semantic_kg.py        # Knowledge graph construction script
│   └── build_vector_index.py       # Qdrant chunk embedding & indexing script
├── src/                             # Core Python Research & Algorithm Implementation
│   ├── chunking/                   # Structure-aware HTML parser & chunker
│   ├── data/                       # Ingestion, extraction, entity resolution, validation
│   ├── evaluation/                 # Benchmark scoring, FEIR, Token F1, Refusal recall
│   ├── models/                     # Model interfaces & embeddings
│   ├── qa/                         # Agent, query analysis, sufficiency gating, generator
│   └── retrieval/                  # Qdrant vector, graph traversal, RRF fusion, reranker
├── tests/                           # Unit, Integration, and Mock Test Suites
└── archive/                         # Legacy ML return-forecasting experiments
```

---

## 3. Component Implementation Status Matrix

| Component Module | Source Directory / File | Implementation Status | Verified Tech Stack | Lines of Code | Research Role / Functionality |
|---|---|---|---|---|---|
| **SEC Corpus Ingestion** | `src/data/ingestion/` | **COMPLETED** | Python 3.12, BeautifulSoup4 | ~1,240 | Downloads, normalizes, and extracts SEC EDGAR HTML/HTM filings (10-K, 10-Q, 8-K). |
| **Structure-Aware Chunking** | `src/data/chunking/` | **COMPLETED** | Python, Regex, BS4 | ~850 | Segments filings preserving Item hierarchies, tables, and narrative boundaries. |
| **Bitemporal Metadata Assignment** | `src/data/documents/` | **COMPLETED** | Python, datetime (UTC) | ~620 | Assigns valid_from, valid_to, publication_date, and available_time to every chunk. |
| **Entity & Relation Extraction** | `src/data/extraction/` | **COMPLETED** | Ollama, Pydantic, Instructor | ~1,410 | Extracts corporate acquisitions, supply relations, competitors, and risk factors. |
| **Temporal Knowledge Graph** | `src/data/graph/` | **COMPLETED** | NetworkX, Custom Adjacency | ~1,850 | Constructs temporal graphs, point-in-time snapshots, and relationship traversals. |
| **Vector Store & Embeddings** | `src/retrieval/vector.py` | **COMPLETED** | Qdrant, sentence-transformers | ~380 | Embeds 8,181 chunks using `all-MiniLM-L6-v2` (384-dim) stored in local Qdrant. |
| **Graph Retriever** | `src/retrieval/graph.py` | **COMPLETED** | Graph traversal, entity matching | ~520 | Traverses 1-hop and 2-hop neighborhoods with relation-type intent filtering. |
| **Entity-Balanced RRF Fusion** | `src/retrieval/hybrid.py` | **COMPLETED** | Reciprocal Rank Fusion, Greedy Quotas | ~430 | Merges vector & graph ranks; reserves slots per entity for comparative questions. |
| **Deterministic Reranker** | `src/retrieval/reranker.py` | **COMPLETED** | Heuristic cross-scoring | ~310 | Adjusts RRF scores using temporal validity bonuses, entity density, and provenance. |
| **Query Intent & Entity Analyzer**| `src/qa/query_analysis.py` | **COMPLETED** | Rule-based & regex heuristics | ~290 | Classifies query into Direct, Multi-Hop, Comparative, or Temporal intent. |
| **Evidence Sufficiency Gate** | `src/qa/evidence_sufficiency.py` | **COMPLETED** | Multi-criteria validation | ~480 | Strict gate evaluating entity coverage, hop mode, and relationship support. |
| **LLM Answer Generator** | `src/qa/generator.py` | **COMPLETED** | Ollama (`qwen3.5:4b`) | ~360 | Synthesizes grounded answers strictly referencing verified citations in context. |
| **Answer Validator & Fallback** | `src/qa/validation.py` | **COMPLETED** | Regex citation verification | ~270 | Checks whether citations exist in evidence; triggers refusal fallback if ungrounded. |
| **Temporal Leakage Evaluator** | `src/evaluation/temporal.py` | **COMPLETED** | ISO timestamp inequality | ~180 | Computes Future Evidence Infiltration Rate (FEIR) across retrieval stages. |
| **Benchmark Scoring Suite** | `src/evaluation/qa.py` | **COMPLETED** | Token F1, Exact Match, Confusion | ~340 | Evaluates answer accuracy, refusal precision, refusal recall, and decision accuracy. |
| **FastAPI REST Service** | `api/` | **COMPLETED** | FastAPI, Uvicorn, Pydantic v2 | ~1,150 | Exposes `/query`, `/graph`, `/evidence`, `/benchmark`, `/health` endpoints. |
| **Research Terminal UI** | `frontend/` | **COMPLETED** | Next.js 14, React, Tailwind, Lucide | ~4,200 | Interactive UI for GraphRAG querying, radial KG exploration, and benchmark audit. |
| **Auxiliary Return Predictor** | `archive/old_ml_experiments/`| **ARCHIVED / AUXILIARY** | Scikit-learn (LogReg, RF, GBDT) | ~1,600 | 5-day stock direction walk-forward prediction over 354 snapshots. |

---

## 4. Key Discrepancies and Architectural Clarifications

1. **Knowledge Graph Node/Edge Scale:**
   - *Documentation Discrepancy:* Early project notes refer to 1,495 nodes and 660 edges.
   - *Code Reality:* 1,495 nodes and 660 edges belongs strictly to the single-company `aapl_temporal_graph.json`. The master multi-company graph (`fingraph_semantic_kg.json`) spans 10 enterprise tickers (`AAPL, AMZN, GOOGL, JNJ, JPM, META, MSFT, NVDA, TSLA, XOM`) and contains exactly **10,054 nodes** and **9,822 edges**.
2. **QA System vs Stock Return Prediction:**
   - *Documentation Ambiguity:* Early summaries juxtaposed ROC-AUC numbers (0.6342) alongside GraphRAG QA metrics.
   - *Code Reality:* The stock direction prediction is an auxiliary ML experiment located in `archive/old_ml_experiments/` evaluated on 354 chronological snapshots of AAPL market/SEC data. The core research contribution is the **Temporal Knowledge Graph & Hybrid GraphRAG QA Framework** evaluated on the 64-question SEC filing benchmark.
3. **Storage Engine:**
   - *Documentation Claim:* Mentioned PostgreSQL alongside Qdrant.
   - *Code Reality:* The live working implementation uses embedded local Qdrant SQLite storage (`data/vector_store/collection/fingraph_chunks/storage.sqlite`) and local JSON-based knowledge graph persistence. No active PostgreSQL container or connection is required for running the pipeline.
