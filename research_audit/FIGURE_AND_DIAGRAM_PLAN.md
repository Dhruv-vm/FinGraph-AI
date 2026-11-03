# FinGraph AI — Academic Figure and Diagram Specification Plan

**Audit Date:** 2026-11-01  
**Lead Authors:** Senior Research Scientist, Full-Stack Architect  

---

This document outlines the 12 publication-ready figures required for submission to an international IEEE/ACM conference. Every diagram corresponds to an implemented mechanism.

| Figure No. | Proposed Title | Technical Components & Flow | Data / File Origin | Readiness |
|---|---|---|---|---|
| **Fig. 1** | **Overall System Architecture** | SEC EDGAR $\to$ Structure Chunker $\to$ Temporal Graph + Vector Store $\to$ Hybrid Orchestrator $\to$ Sufficiency Gate $\to$ LLM Generator $\to$ Citation Validator. | `src/` modular topology | **READY TO RENDER (TikZ / SVG)** |
| **Fig. 2** | **Structure-Aware Document Decomposition** | Raw 10-K HTML hierarchy $\to$ Item 1A/7/8 tagging $\to$ Table cell preservation $\to$ Bitemporal chunk serialization. | `src/data/chunking/structure_aware.py` | **READY TO RENDER** |
| **Fig. 3** | **Bitemporal Knowledge Graph Topology** | Multi-entity node network (Companies, Regulators, Products) connected by typed edges with `[valid_from, valid_to]` and `available_time`. | `data/processed/unified/fingraph_semantic_kg.json` | **READY TO EXPORT (NetworkX / Gephi)** |
| **Fig. 4** | **Point-in-Time Snapshot Slicing** | Full graph spanning 2024–2026 filtered by $t_{\text{as\_of}} = \text{2025-06-30}$; ghosted nodes/edges illustrating pruned future facts. | `src/data/graph/snapshot.py` | **READY TO GENERATE** |
| **Fig. 5** | **Hybrid Retrieval & Entity-Balanced RRF** | Dual-tower retrieval (Qdrant Cosine + BFS Graph Paths) $\to$ RRF scoring $\to$ Entity-balanced quota partitioning ($k_{\min}=3$). | `src/retrieval/hybrid.py` | **READY TO RENDER** |
| **Fig. 6** | **Multi-Hop Traversal Workflow** | Step-by-step query decomposition: Query $\to$ Path $(E_1 \xrightarrow{R_1} E_2 \xrightarrow{R_2} E_3)$ $\to$ Contextual evidence binding. | `src/qa/query_analysis.py`, `src/retrieval/graph.py` | **READY TO RENDER** |
| **Fig. 7** | **Adaptive Sufficiency Gating Logic** | Decision flowchart: Entity Coverage $\to$ Direct vs Compound intent branch $\to$ Temporal validity check $\to$ Pass / Refusal Fallback. | `src/qa/evidence_sufficiency.py` | **READY TO RENDER** |
| **Fig. 8** | **Evidence Grounding & Citation Audit** | Generated answer with token citations linked directly to SEC Accession, Item section, and filing date. | `src/data/documents/provenance.py` | **READY TO RENDER** |
| **Fig. 9** | **Benchmark Decision Confusion Matrix** | $2 \times 2$ matrix illustrating 45 Correct Answers, 5 False Refusals, 14 Correct Refusals, and 0 Hallucinations on Phase 7. | `results/phase7/phase7_summary.json` | **READY TO PLOT (Matplotlib / Seaborn)** |
| **Fig. 10** | **Temporal Leakage vs FEIR by Stage** | Bar chart comparing FEIR across: Pre-filter, Post-filter, Context, Cited tokens (all 0.0% in FinGraph vs unconstrained RAG). | `src/evaluation/temporal.py` | **READY TO PLOT** |
| **Fig. 11** | **Ablation Performance Across Iterations** | Grouped bar chart comparing Phase 5 (Baseline), Phase 6 (Attribution Hardened), and Phase 7 (Adaptive + Balanced RRF). | `results/phase7/ablation_table.json` | **READY TO PLOT** |
| **Fig. 12** | **Radial Corporate Ecosystem Subgraph** | Concentric radial layout centered on NVIDIA (NVDA) displaying 118 first-degree corporate partners, suppliers, and competitors. | `frontend/src/app/graph/` & semantic KG | **READY (Live in Next.js)** |
