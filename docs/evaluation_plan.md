# FinGraph AI Evaluation Plan

## Research Evaluation Framework

This document outlines the rigorous evaluation methodology for the research paper:
**“FinGraph AI: A Temporal Knowledge Graph and Hybrid GraphRAG Framework for Multi-Hop Financial Question Answering”**

The goal is to quantitatively validate the contributions across retrieval quality, temporal validity, multi-hop reasoning, and parametric leakage prevention against a pure vector-based RAG baseline.

---

## 1. Core Evaluation Dimensions

### Dimension A: Retrieval & Multi-Hop Accuracy
Evaluates the retrieval precision, recall, and ability to reconstruct multi-hop reasoning chains.
- **Hit@K ($K \in \{1, 3, 5, 10\}$):** Proportion of queries where at least one ground-truth fact is retrieved in the top $K$.
- **Mean Reciprocal Rank (MRR):** Reciprocal rank of the first relevant evidence item.
- **Normalized Discounted Cumulative Gain (NDCG@K):** Evaluates ranking quality with relevance grades.
- **Multi-Hop Path Accuracy:** Proportion of multi-hop queries where the complete relational chain (e.g., $E_1 \xrightarrow{R_1} E_2 \xrightarrow{R_2} E_3$) is present in the evidence package.
- **Cross-Document Fact Coverage:** Proportion of required facts retrieved when evidence spans across multiple SEC filings.

### Dimension B: Strict Temporal Validity (Zero Leakage)
Validates that future information never infiltrates point-in-time financial decisions.
- **Future Evidence Infiltration Rate (FEIR):**
  $$\text{FEIR} = \frac{\sum_{i=1}^N \sum_{e \in E_i} \mathbb{I}(t_e > t_{\text{ref}, i})}{\sum_{i=1}^N |E_i|}$$
  **Target:** Exactly $0.00\%$ ($0$ violations across all benchmarks).
- **Point-in-Time Temporal Gate Rejection Count:** Number of future edges/chunks successfully blocked during point-in-time traversal.

### Dimension C: Parametric Leakage Evaluation
Evaluates the tendency of LLMs to answer historical questions using training data weights (parametric memory) rather than retrieved evidence.
- **Retriever-Disabled Baseline:** Querying LLM directly on historical point-in-time questions where subsequent events occurred after the cutoff date (e.g., asking about NVIDIA's 2025 chip roadmap as of January 2024).
- **Parametric Leakage Rate (PLR):** Percentage of answers mentioning future facts (events occurring after $t_{\text{ref}}$) despite the prompt specifying $t_{\text{ref}}$.
- **Retriever-Grounded Verification:** Measuring whether providing FinGraph AI's temporally validated evidence package eliminates parametric hallucination.

### Dimension D: Evidence Sufficiency & Agent Efficiency
Evaluates the performance and restraint of the lightweight ReAct retrieval loop:
- **Average Retrieval Rounds:** Expected average number of rounds ($1 \le \bar{R} \le 3$).
- **Sufficiency Convergence Rate:** Proportion of initially insufficient queries resolved by targeted follow-up iterations.
- **Action Adherence Rate:** Proportion of agent actions strictly matching the whitelisted action set ($100\%$).

---

## 2. Experimental Baselines & Comparisons

The paper evaluates four distinct architectural configurations:

| Model / Configuration | Knowledge Graph | Vector Retrieval | Temporal Filter | Reranker | Multi-Hop ReAct |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **B1: Vector-Only RAG** | ❌ | Dense MiniLM | Chunk Date | Cosine Sim | ❌ (Single shot) |
| **B2: Graph-Only RAG** | TKG | ❌ | Edge $t_{\text{avail}}$ | Graph Degree | ❌ (Single shot) |
| **B3: Hybrid GraphRAG (No ReAct)** | TKG | Dense MiniLM | Both | Deterministic Reranker | ❌ (Single shot) |
| **FinGraph AI (Full Framework)** | TKG | Dense MiniLM | Point-in-Time BFS | Deterministic Reranker | ✅ Controlled ReAct (Max 2) |

---

## 3. Financial Benchmark Dataset Construction

The benchmark comprises 50–80 hand-verified questions over the 99 SEC 10-K/10-Q filings. LLM-assisted question generation may be used only as a generation aid, with manual verification required:
1. **Factual Single-Hop (15–20 questions):** Direct financial facts (e.g., "What was Apple's total net sales in Q3 2026?").
2. **Relational Single-Hop (10–15 questions):** Corporate relations (e.g., "Which companies compete with Microsoft?").
3. **Temporal Point-in-Time (10–15 questions):** State-dependent queries with explicit reference cutoffs (e.g., "What risks affected NVIDIA as of June 1, 2025?").
4. **Relational Multi-Hop (10–15 questions):** Multi-hop supplier/customer chains (e.g., "What products manufactured by NVIDIA depend on TSMC?").
5. **Cross-Document Comparative (5–15 questions):** Questions requiring facts across different companies and filings (e.g., "Compare the primary cloud suppliers mentioned by Apple and Alphabet").
