# FinGraph AI — IEEE Conference Paper Blueprint & Section-by-Section Plan

**Target Venue:** IEEE BigData / ACM ICAIF / IEEE ICTAI  
**Format:** Double-Column, 10 pt font, 10 Pages + 2 Pages References  
**Working Title:** *FinGraph AI: A Temporal Knowledge Graph and Hybrid GraphRAG Framework for Multi-Hop Financial Question Answering*  

---

## Section Budget & File Evidence Map

| Section No. & Title | Page Budget | Repository Evidence Files | Required Equations | Figures & Tables | Status |
|---|---|---|---|---|---|
| **I. Introduction** | 1.25 pages | `docs/ARCHITECTURE.md`, `README.md` | None | Fig. 1 (System High-Level Teaser) | Ready to write |
| **II. Related Work** | 1.0 page | `research_audit/RELATED_WORK_RESEARCH_PLAN.md` | None | Table 1 (Taxonomy of RAG vs GraphRAG) | Ready to write |
| **III. Problem Formulation** | 0.75 pages | `src/data/graph/schema.py`, `src/qa/` | Eq. 1, 3, 7 | None | Ready to write |
| **IV. Bitemporal KG Construction** | 1.25 pages | `src/data/chunking/`, `src/data/graph/` | Eq. 7, 8 | Fig. 2 (Chunking), Fig. 3 (KG Topology) | Ready to write |
| **V. Hybrid Retrieval & EB-RRF** | 1.5 pages | `src/retrieval/vector.py`, `src/retrieval/hybrid.py` | Eq. 1, 2, 4, 5, 6 | Fig. 4 (Snapshot), Fig. 5 (EB-RRF Architecture) | Ready to write |
| **VI. Adaptive Sufficiency & Reasoning**| 1.0 page | `src/qa/evidence_sufficiency.py`, `src/qa/query_analysis.py` | Eq. 13, 14 | Fig. 6 (Multi-hop Path), Fig. 7 (Gate Flowchart) | Ready to write |
| **VII. Experimental Setup** | 1.0 page | `benchmark/`, `src/evaluation/` | Eq. 9, 10, 11, 12 | Table 2 (Corpus Statistics), Table 3 (Benchmark Breakdown) | Ready to write |
| **VIII. Results and Empirical Analysis**| 1.25 pages | `results/phase7/phase7_summary.json` | None | Table 4 (Main Performance Table), Fig. 9 (Confusion Matrix) | Ready to write |
| **IX. Ablation Studies** | 0.75 pages | `results/phase7/ablation_table.json` | None | Table 5 (Ablation Matrix), Fig. 11 (Phase Evolution) | Needs EXP 1–5 runs |
| **X. Downstream Applications (Predictive Case Study)**| 0.5 pages | `archive/old_ml_experiments/` | Eq. 8 (Auxiliary) | Table 6 (Return Forecasting Walk-Forward) | Ready to summarize |
| **XI. Threats to Validity & Limitations**| 0.5 pages | `results/phase7/error_analysis.json` | None | None | Ready to write |
| **XII. Conclusion & Future Work** | 0.25 pages | Summary of contributions | None | None | Ready to write |
| **References** | 1.5–2.0 pages | `research_audit/REFERENCE_VERIFICATION.md` | None | 30+ peer-reviewed citations | Ready to compile |
