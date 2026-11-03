# FinGraph AI — Missing Experiments & Evaluation Roadmap

**Audit Date:** 2026-11-01  
**Auditor:** Senior AI/ML Research Scientist & Reviewer  
**Purpose:** Identify empirical gaps required to satisfy Tier-1/Tier-2 conference peer reviewers.

---

## 1. Critical Experimental Gaps Identified

While Phase 7 established strong results on the 64-question benchmark (Decision Accuracy = 92.19%, Refusal Recall = 100%, FEIR = 0.0%), academic reviewers at top venues (e.g., CIKM, IEEE BigData, ICAIF) will expect formal baseline comparisons and ablation tables across modalities.

### Summary of Missing Experiments:

| ID | Experiment Description | Current Status | Reviewer Impact | Effort to Complete |
|---|---|---|---|---|
| **EXP-1** | **Pure Vector-Only Baseline (Dense RAG)** | Missing on full 64-Q benchmark | **CRITICAL (Blocker)** | 1 day (Run existing pipeline with `mode='vector_only'`) |
| **EXP-2** | **Pure Graph-Only Baseline (Graph Traverse)** | Missing on full 64-Q benchmark | **CRITICAL (Blocker)** | 1 day (Run existing pipeline with `mode='graph_only'`) |
| **EXP-3** | **Standard RRF vs Entity-Balanced RRF** | Partially observed on CDC-05 | **HIGH (Essential for novelty claim)** | 0.5 days (Run ablation over 10 comparative questions) |
| **EXP-4** | **Temporal Gating Ablation (FEIR Impact)** | Tested in Phase 7; unconstrained baseline missing | **HIGH (Demonstrates temporal value)** | 0.5 days (Run benchmark with temporal filtering disabled) |
| **EXP-5** | **Sufficiency Gate Ablation (Strict vs Adaptive vs None)**| Phase 5/6 data exists; needs unified table | **HIGH** | 0.5 days (Aggregate existing runs into unified table) |
| **EXP-6** | **LLM Generator Robustness (Cross-Model)**| Evaluated only on `qwen3.5:4b` | **MEDIUM (Reviewer favorite)** | 2 days (Evaluate on Llama 3.1 8B or Mistral 7B) |
| **EXP-7** | **System Latency & Compute Profiling** | Ad-hoc measurements | **MEDIUM** | 0.5 days (Record breakdown of retrieval, rerank, LLM time) |

---

## 2. Experimental Protocols for Missing Baselines

### EXP-1: Pure Vector Retrieval Baseline (Dense RAG)
- **Research Question:** What is the performance deficit when using standard dense retrieval without Knowledge Graph assistance?
- **Protocol:** Submit all 64 benchmark questions to `VectorStore.search(query, top_k=8)` without graph expansion. Feed retrieved chunks to the exact same generator and sufficiency gate.
- **Metrics:** Decision Accuracy, Correct Answers, False Refusals, Token F1, FEIR.
- **Expected Outcome:** Lower Recall@8 on multi-hop questions (e.g. RMH series), resulting in higher false refusal rate or lower Token F1.

### EXP-2: Pure Graph Retrieval Baseline
- **Research Question:** How well does structured graph traversal perform in isolation without dense semantic chunks?
- **Protocol:** Run BFS graph neighborhood expansion from recognized entities; synthesize answer purely from edge metadata without text chunks.
- **Metrics:** Decision Accuracy, Token F1, hallucination rate.
- **Expected Outcome:** High precision on direct relational queries, but complete failure on descriptive/narrative questions where text nuance is required.

### EXP-3: Entity-Balanced RRF vs Standard RRF
- **Research Question:** Does EB-RRF statistically outperform standard RRF on comparative queries?
- **Protocol:** Evaluate the 10 comparative questions (`CDC-01` to `CDC-10`) under:
  1. Standard RRF ($k=60$)
  2. Entity-Balanced RRF ($k=60, 	ext{min\_alloc}=3$)
- **Metrics:** Entity Recall Ratio $\min(|E_1|, |E_2|) / \max(|E_1|, |E_2|)$, Question Answer Rate, Token F1.

### EXP-4: Temporal Gating Ablation (FEIR Measurement)
- **Research Question:** What is the quantitative rate of future evidence leakage when temporal constraints are removed?
- **Protocol:** Submit point-in-time questions (e.g., "As of June 2025, did X acquire Y?") to:
  1. FinGraph Temporal Pipeline ($t_{	ext{available}} \le 2025	ext{-}06	ext{-}30$)
  2. Unconstrained Pipeline (all 8,181 chunks and 9,822 edges available)
- **Metrics:** Pre-filter FEIR, Post-filter FEIR, Cited FEIR, Anachronism Hallucination Rate.
- **Expected Outcome:** Unconstrained RAG exhibits FEIR $> 20\%$, producing factually false answers based on 2026 events. FinGraph maintains FEIR $= 0.0\%$.

---

## 3. Standard Information Retrieval Metric Additions

In addition to QA-level metrics (Token F1, Exact Match), paper reviewers expect standard IR metrics on retrieved evidence:
- **Recall@K ($K \in \{3, 5, 8\}$):** Fraction of golden evidence chunks retrieved.
- **MRR (Mean Reciprocal Rank):** Position of the first relevant evidence chunk.
- **NDCG@8 (Normalized Discounted Cumulative Gain):** Quality of ranking order.
- **Hit Rate@K:** Percentage of queries where at least one ground-truth document is retrieved.
