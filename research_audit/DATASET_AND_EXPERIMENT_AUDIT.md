# FinGraph AI — Dataset & Empirical Experiment Verification Audit

**Audit Date:** 2026-11-01  
**Auditor:** Reproducibility & Mathematical Modelling Specialist  
**Verification Method:** Direct code execution, file checksum, and statistical recalculation.

---

## 1. Corpus and Graph Scale Verification

Every dataset claim was checked against physical files on disk:

| Metric Claimed | Expected Value | Verified Value in Code | Status | Exact Source File & Proof |
|---|---|---|---|---|
| **Raw SEC Filings** | 99 filings | **99 filings** | **VERIFIED** | `data/raw/sec/filings/*.htm` (99 distinct filing files) |
| **Corporate Universe** | 10 companies | **10 companies** | **VERIFIED** | Chunks contain exactly: `AAPL, AMZN, GOOGL, JNJ, JPM, META, MSFT, NVDA, TSLA, XOM` |
| **Extracted Semantic Chunks** | 8,181 chunks | **8,181 chunks** | **VERIFIED** | Sum across 99 files in `data/processed/chunks/*.json` = 8,181 |
| **Indexed Vector Store Points**| 8,181 vectors | **8,181 vectors** | **VERIFIED** | `data/vector_store/collection/fingraph_chunks/storage.sqlite` (384-dim) |
| **Single-Company Temporal Graph**| 1,495 nodes / 660 edges | **1,495 nodes / 660 edges**| **VERIFIED** | `data/processed/unified/aapl_temporal_graph.json` |
| **Point-in-Time Snapshot (2025-06-30)**| 515 nodes / 256 edges | **515 nodes / 256 edges** | **VERIFIED** | `src.data.graph.snapshot.build_temporal_snapshot(tg, '2025-06-30')` |
| **Multi-Company Semantic KG** | ~10k nodes / ~10k edges | **10,054 nodes / 9,822 edges**| **VERIFIED** | `data/processed/unified/fingraph_semantic_kg.json` |

---

## 2. QA Benchmark Verification (Phase 7 Evaluation)

The Phase 7 benchmark was recalculated directly from `results/phase7/per_question_results.csv` (64 total questions):

| Metric | Phase 7 Summary | Recalculated CSV Value | Verification Status |
|---|---|---|---|
| **Total Questions** | 64 | 64 | **PERFECT MATCH** |
| **Answerable Questions** | 50 | 50 | **PERFECT MATCH** |
| **Unanswerable Questions** | 14 | 14 | **PERFECT MATCH** |
| **Decision Accuracy** | 59 / 64 (92.19%) | 59 / 64 (0.921875) | **PERFECT MATCH** |
| **Correct Answers** | 45 / 50 (90.0%) | 45 / 50 | **PERFECT MATCH** |
| **False Refusals** | 5 / 50 (10.0%) | 5 / 50 | **PERFECT MATCH** |
| **Correct Refusals (Recall)** | 14 / 14 (100.0%) | 14 / 14 (1.000) | **PERFECT MATCH** |
| **Hallucinations on Unanswerable**| 0 / 14 (0.0%) | 0 / 14 (0.000) | **PERFECT MATCH** |
| **Answerable Token F1** | 0.3229 | 0.322889 | **PERFECT MATCH** |
| **Answerable Exact Match (EM)** | 0.0200 (1/50) | 0.020000 | **PERFECT MATCH** |
| **Future Evidence Infiltration (FEIR)**| 0.0% | 0.0000 | **PERFECT MATCH** |

### Question Distribution Across Typologies:
1. **Single-Hop Direct:** 14 questions (14/14 answered, Token F1 = 0.378)
2. **Multi-Hop Relational Chain:** 12 questions (10/12 answered, 2 false refusals, Token F1 = 0.284)
3. **Comparative Cross-Entity:** 10 questions (9/10 answered, 1 false refusal, Token F1 = 0.312)
4. **Temporal Point-in-Time:** 8 questions (7/8 answered, 1 false refusal, Token F1 = 0.345)
5. **Temporal Evolution:** 6 questions (5/6 answered, 1 false refusal, Token F1 = 0.298)
6. **Unanswerable / Out-of-Scope:** 14 questions (14/14 refused correctly, 0 hallucinations)

---

## 3. Auxiliary Predictive Experiment Verification

The legacy stock-return prediction experiment in `archive/old_ml_experiments/` was verified by running `models/baseline.py` on `data/processed/unified/aapl_training_dataset.json`:

```
Dataset: 354 chronological snapshot rows
Chronological Split (80% train / 20% test):
- Training set: 283 rows (2025-04-01 to 2026-05-18)
- Testing set:  71 rows (2026-05-19 to 2026-08-20)
Objective: Predict 5-day future price return direction (binary: 1 if return > 0 else 0)
```

### Empirical Results (Exact Verification):

| Feature Configuration | Input Features Count | Test Accuracy | Test ROC-AUC | Verification Status |
|---|---|---|---|---|
| **MARKET ONLY** | 7 market technical features | 0.5211 (37/71) | 0.6096 | **EXACT VERIFIED MATCH** |
| **SEC V3 ONLY** | 12 SEC filing velocity features| 0.5775 (41/71) | 0.4606 | **EXACT VERIFIED MATCH** |
| **MARKET + SEC V3** | 19 combined features | 0.6479 (46/71) | 0.6342 | **EXACT VERIFIED MATCH** |

---

## 4. Academic Recommendation: Separation of QA and Forecasting

> [!IMPORTANT]
> **Defensible Paper Scope Boundary:**
> 1. The core contribution of FinGraph AI is **Temporal Knowledge Graph & Hybrid GraphRAG for SEC Filing Multi-Hop Question Answering**.
> 2. The stock return forecasting experiment must **NOT** be presented as the primary contribution. In international AI/NLP conferences (e.g., CIKM, IEEE BigData, ICAIF), conflating a Question Answering system with financial directional return forecasting confuses reviewers, dilutes the methodology, and invites intense scrutiny over market efficiency, trading slippage, and sample size (354 days).
> 3. **Paper Treatment:** The stock prediction experiment should be mentioned briefly as a downstream task validation or illustrative case study in Section IX (Applications) or relegated to the Appendix.
