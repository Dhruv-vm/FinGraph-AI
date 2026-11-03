# FinGraph AI — Master Research Checklist

**Audit Date:** 2026-11-01  
**Status Key:**
- **COMPLETED:** Verified in actual executable code and datasets.
- **PARTIALLY COMPLETED:** Core logic exists but requires extended benchmarking or visual refinement.
- **NOT IMPLEMENTED:** Planned or required for conference camera-ready but not yet created.
- **NEEDS VALIDATION:** Implemented but requires cross-model or ablation rerun.

---

| Task Description | Category | Status | Priority | Exact File Path | Blocking Submission? | Effort |
|---|---|---|---|---|---|---|
| SEC 10-K/10-Q/8-K Ingestion | Data Pipeline | **COMPLETED** | High | `src/data/ingestion/` | No | Done |
| Structure-Aware HTML Chunker | Data Pipeline | **COMPLETED** | High | `src/data/chunking/structure_aware.py` | No | Done |
| 8,181 Semantic Chunks Dataset | Dataset | **COMPLETED** | Critical | `data/processed/chunks/` | No | Done |
| Multi-Company Semantic KG (10k nodes) | Knowledge Graph| **COMPLETED** | Critical | `data/processed/unified/fingraph_semantic_kg.json` | No | Done |
| Point-in-Time Temporal Snapshotting | Knowledge Graph| **COMPLETED** | Critical | `src/data/graph/snapshot.py` | No | Done |
| Embedded Qdrant Vector Retrieval | Retrieval | **COMPLETED** | High | `src/retrieval/vector.py` | No | Done |
| Entity-Balanced RRF Fusion | Retrieval | **COMPLETED** | Critical | `src/retrieval/hybrid.py` | No | Done |
| Adaptive Hop Sufficiency Gating | QA System | **COMPLETED** | Critical | `src/qa/evidence_sufficiency.py` | No | Done |
| Phase 7 Evaluation Benchmark (64 Qs) | Evaluation | **COMPLETED** | Critical | `results/phase7/phase7_summary.json` | No | Done |
| Zero Future Leakage (FEIR = 0.0%) | Evaluation | **COMPLETED** | Critical | `src/evaluation/temporal.py` | No | Done |
| Zero Hallucination on Unanswerable | Evaluation | **COMPLETED** | Critical | `src/evaluation/qa.py` | No | Done |
| FastAPI REST Daemon | Backend | **COMPLETED** | High | `api/main.py` | No | Done |
| Next.js Research Terminal UI (7 views) | Frontend | **COMPLETED** | High | `frontend/src/app/` | No | Done |
| Radial Concentric KG Explorer | Frontend | **COMPLETED** | High | `frontend/src/app/graph/` | No | Done |
| Pure Vector-Only Baseline on 64-Q | Missing Exp | **NOT IMPLEMENTED** | High | `scripts/run_rag_baseline.py` | **YES (for Tier 1/2)** | 1 day |
| Pure Graph-Only Baseline on 64-Q | Missing Exp | **NOT IMPLEMENTED** | High | `scripts/run_graph_baseline.py` | **YES (for Tier 1/2)** | 1 day |
| Comparative EB-RRF vs Standard RRF | Missing Exp | **NEEDS VALIDATION** | High | `results/phase7/ablation_table.json` | **YES (for Tier 1/2)** | 0.5 days |
| System Latency & Profiling Table | Missing Exp | **NOT IMPLEMENTED** | Medium | `src/evaluation/latency.py` | No | 0.5 days |
| Multi-Model Generator Evaluation (Llama) | Missing Exp | **NOT IMPLEMENTED** | Medium | `src/qa/generator.py` | No | 2 days |
| LaTeX Conference Paper Manuscript | Manuscript | **PARTIALLY COMPLETED**| Critical | `paper/main.tex` | **YES** | 2 weeks |
| Zenodo DOI Dataset Minting | Reproducibility| **NOT IMPLEMENTED** | Medium | Zenodo repository | No | 0.5 days |
