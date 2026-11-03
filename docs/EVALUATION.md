# FinGraph AI Empirical Evaluation & Benchmark Report

This document reports the canonical empirical evaluation results for **FinGraph AI: A Temporal Knowledge Graph and Hybrid GraphRAG Framework for Multi-Hop Financial Question Answering**.

All reported figures are derived from the canonical Phase 5 benchmark run across the 99 SEC filing corpus and audited in `results/phase5/benchmark_evaluation_summary.json` and `results/phase5/per_question_results.csv`.

---

## 1. Benchmark Dataset Design

The evaluation benchmark (`benchmark/questions.json`) comprises **64 hand-verified questions** designed to probe multi-hop reasoning, temporal point-in-time constraints, and parametric leakage prevention.

### Category Distribution & Composition
- **Total Questions:** 64
- **Answerable Questions:** 50
- **Unanswerable (Refusal) Questions:** 14

| Benchmark Category | Question Count | Primary Evaluation Focus |
| :--- | :---: | :--- |
| `factual_single_hop` | 13 | Direct corporate disclosures and financial metric retrieval |
| `relational_single_hop` | 13 | Single-hop corporate relations (competitors, customers, subsidiaries) |
| `relational_multi_hop` | 13 | Multi-hop supplier risk chains, connected dependencies |
| `temporal_point_in_time` | 13 | State-dependent risk and business queries with explicit historical cutoff ($t_{\text{ref}}$) |
| `cross_document_comparative` | 12 | Comparative queries spanning multiple company filings (`AAPL`, `MSFT`, `GOOGL`, etc.) |

---

## 2. Canonical Experimental Results: B1 Baseline vs. FinGraph AI

The framework was evaluated against a pure vector-only RAG baseline (**B1**), which employs dense `all-MiniLM-L6-v2` cosine similarity retrieval without graph augmentation, hybrid fusion, RRF, or reranking.

### Comprehensive Metric Comparison

| Evaluation Metric Dimension | B1 Vector-Only RAG | FinGraph AI Full Framework | Absolute Delta | Relative Change |
| :--- | :---: | :---: | :---: | :---: |
| **Retrieval Hit@1** | 0.0625 | **0.0781** | +0.0156 | **+25.0%** |
| **Retrieval Hit@5** | 0.1406 | **0.2031** | +0.0625 | **+44.4%** |
| **Retrieval MRR** | 0.0847 | **0.1305** | +0.0458 | **+54.1%** |
| **Retrieval NDCG@5** | 0.0375 | **0.0536** | +0.0161 | **+42.9%** |
| **Document Hit@5** | 0.7344 | **0.7812** | +0.0468 | **+6.4%** |
| **Cross-Document Fact Coverage** | 0.3929 | **0.5231** | +0.1302 | **+33.1%** |
| **Exact Match (EM)** | 0.1406 | **0.2188** | +0.0782 | **+55.6%** |
| **Token F1** | 0.3045 | **0.3309** | +0.0264 | **+8.7%** |
| **Refusal Recall (Safety on Unanswerable)** | 0.6429 (9/14) | **0.9286 (13/14)** | +0.2857 | **+44.4%** |
| **Refusal Precision** | **0.3913** (9/23) | 0.3171 (13/41) | -0.0742 | -19.0% |
| **Decision Accuracy** | **0.7031** | 0.5469 | -0.1562 | -22.2% |
| **Hallucinations on Unanswerable** | 5 | **1** | -4 | **-80.0%** |
| **Future Evidence Infiltration Rate (FEIR)** | 0.00% | **0.00%** | 0.00% | Zero Violation |
| **Average Retrieval Rounds** | 1.00 | 1.58 | +0.58 | Adaptive ReAct |

---

## 3. Confusion Matrix for Refusal Decisions

Evaluating how each system handles the 64 questions (50 answerable, 14 unanswerable):

### B1 Vector-Only RAG
- **True Positive (Refused Unanswerable):** 9
- **False Positive (Refused Answerable):** 14
- **False Negative (Answered Unanswerable / Hallucination):** 5
- **True Negative (Answered Answerable):** 36
- **Refusal Recall:** $9 / 14 = 64.29\%$
- **Refusal Precision:** $9 / 23 = 39.13\%$
- **False Refusal Rate on Answerable:** $14 / 50 = 28.00\%$

### FinGraph AI Full Framework (CFG0 Strict Mode)
- **True Positive (Refused Unanswerable):** 13
- **False Positive (Refused Answerable):** 28
- **False Negative (Answered Unanswerable / Hallucination):** 1
- **True Negative (Answered Answerable):** 22
- **Refusal Recall:** $13 / 14 = 92.86\%$
- **Refusal Precision:** $13 / 41 = 31.71\%$
- **False Refusal Rate on Answerable:** $28 / 50 = 56.00\%$

> **Analysis:** FinGraph AI's strict evidence sufficiency gate prioritizes financial safety. It successfully catches 13 of the 14 unanswerable questions (reducing hallucinations by 80%), at the cost of a higher false refusal rate on answerable queries where partial entity mentions fail strict matching.

---

## 4. Evidence Sufficiency Ablation Study (CFG0–CFG5)

To systematically analyze the trade-off between safety (refusal recall) and answering coverage (false refusal rate), we conducted a 6-configuration ablation across the 64 benchmark questions:

| Config | Name | Entity Mode | Relationship Mode | Hop Mode | Gate Enabled | Refusal Recall | Refusal Prec | False Refusal Rate | Hallucinations | Accuracy |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **CFG0** | Phase 5 Strict Gate | Strict (100%) | Strict + Compound | Strict | Yes | **92.86%** | 31.71% | 56.00% | **1** | 54.69% |
| **CFG1** | Soft Hop Gate | Strict (100%) | Strict + Compound | Soft | Yes | 78.57% | 29.73% | 52.00% | 3 | 54.69% |
| **CFG2** | Partial Entity + Soft Hop | Partial ($\ge 50\%$) | Strict + Compound | Soft | Yes | 78.57% | **50.00%** | **22.00%** | 3 | **78.12%** |
| **CFG3** | Textual Relationship | Strict (100%) | Textual Rel | Soft | Yes | 71.43% | 27.78% | 52.00% | 4 | 53.12% |
| **CFG4** | Balanced Textual Gate | Partial ($\ge 50\%$) | Textual Rel | Soft | Yes | 50.00% | 38.89% | **22.00%** | 7 | 71.88% |
| **CFG5** | Disabled Gate | *Disabled* | *Disabled* | *Disabled* | No | 50.00% | 46.67% | 16.00% | 7 | 76.56% |

### Key Ablation Insights
1. **Gate Value (CFG0 vs CFG5):** Disabling the gate (CFG5) results in 7 hallucinations on unanswerable questions (a 50% failure rate). CFG0 suppresses this to just 1 hallucination, demonstrating the critical protective value of the sufficiency gate.
2. **The Operational Frontier (CFG2):** While CFG0 is optimal for high-stakes regulatory auditing, **CFG2 (Partial Entity Coverage $\ge 50\%$ with Soft Hop)** achieves the highest overall decision accuracy ($78.12\%$), cutting false refusals on answerable queries from $56.00\%$ to $22.00\%$ while maintaining $78.57\%$ refusal recall and only 3 hallucinations.

---

## 5. Controlled Compound Evidence Chain Rerun

Following the implementation of strict compound relationship validation (`SUPPLIES` + `HAS_RISK` connected chains), we conducted a controlled rerun on compound multi-hop queries (`RMH-05`, `RMH-11`, `RMH-12`):

```text
Query RMH-11: "What internal manufacturing risks affect the suppliers that supply Apple?"
Reference Time: 2026-08-21T00:00:00+00:00
```

- **Without Compound Chain Validation:** Retrieval accepted passages where Foxconn supplied Apple alongside unrelated chunks describing general manufacturing risks at Intel, falsely declaring evidence sufficient.
- **With Compound Chain Validation:** The sufficiency gate requires that the entity experiencing the risk ($S$) is identical to the entity supplying Apple ($S \xrightarrow{\text{SUPPLIES}} \text{AAPL} \land S \xrightarrow{\text{HAS\_RISK}} R$). If an unbroken path is not present in the retrieved evidence, the gate flags the defect and prompts targeted ReAct expansion or issues a clean refusal.

---

## 6. Temporal Safety & Parametric Leakage Evaluation

### Future Evidence Infiltration Rate (FEIR)
In an unconstrained RAG pipeline without temporal filtering:
- **Without-Filtering FEIR:** **78.46%** of retrieved evidence items for point-in-time queries contain future information published after the reference date.
- **Gate Rejections:** When FinGraph AI's temporal filter is applied, it intercepts and rejects **102 future candidate items**.
- **With-Filtering FEIR:** **0.00%** (0 future items infiltrate the context).

### Parametric Leakage Rate (PLR)
To measure the intrinsic tendency of LLMs to leak post-cutoff knowledge from training weights:
- When retrieval is disabled and `qwen3.5:4b` is asked historical point-in-time questions about post-cutoff events (e.g., NVIDIA's Blackwell architecture prior to its public disclosure), the raw model leaks future facts with a **$\text{PLR} = 80.00\%$**.
- Grounding answers strictly within FinGraph AI's temporally verified evidence package eliminates parametric leakage entirely.

### Temporal Evaluation Instrumentation & Edge Cases
The evaluation harness handles nullable stage metrics cleanly:
- When a batch contains zero temporal queries or when upstream retrieval pre-filters all future candidates during index loading, the stage FEIR is serialized as `null` rather than masquerading as `0.00%`, preserving statistical integrity and preventing false empirical claims.

---

## 7. Latency and Agent Efficiency

| System Configuration | Avg Retrieval Rounds | Avg Latency / Query | Generation Backend |
| :--- | :---: | :---: | :--- |
| **B1 Vector-Only RAG** | 1.00 | 1.82s | Local `qwen3.5:4b` (Ollama) |
| **FinGraph AI (Refusal)** | 1.58 | 0.00s (Generation) / 0.45s (Retrieval) | Zero LLM token execution |
| **FinGraph AI (Answered)** | 1.58 | 3.42s | Local `qwen3.5:4b` (Ollama, $T=0$) |

The adaptive ReAct loop converges in an average of **1.58 rounds** (out of a maximum 2 refinement rounds), successfully resolving missing entities and relationships without entering unbounded search loops.
