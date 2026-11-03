# FinGraph AI Research Methodology

This document outlines the theoretical foundations, research hypotheses, formal methodology, and experimental protocols governing **FinGraph AI: A Temporal Knowledge Graph and Hybrid GraphRAG Framework for Multi-Hop Financial Question Answering**.

---

## 1. Research Overview & Theoretical Foundations

Financial intelligence over corporate regulatory filings (SEC 10-K, 10-Q) represents a demanding domain for AI systems. Financial analysts require answers that:
1. Reconstruct multi-hop corporate connections across distinct disclosures (e.g., supply-chain vendor dependencies, joint ventures, competitive exposures).
2. Explicitly adhere to point-in-time informational constraints, eliminating lookahead bias and temporal information leakage.
3. Refuse to fabricate answers when the historical public record lacks verified evidentiary support.

### Primary Research Question
*Can a temporal knowledge graph-augmented hybrid retrieval framework improve multi-hop financial question answering compared with conventional vector-only RAG, while maintaining strict point-in-time evidence constraints and eliminating temporal leakage?*

### Formal Hypotheses
- **Hypothesis 1 (H1 — Hybrid Retrieval Superiority):** Combining dense semantic vector retrieval with structured graph traversal via Reciprocal Rank Fusion (RRF) achieves higher multi-hop retrieval recall, MRR, and cross-document coverage than vector-only RAG.
- **Hypothesis 2 (H2 — Point-in-Time Temporal Safety):** Enforcing explicit point-in-time availability boundaries ($t_{\text{avail}} \le t_{\text{ref}}$) at the graph edge and chunk retrieval levels achieves a Future Evidence Infiltration Rate of $\text{FEIR} = 0.00\%$, preventing lookahead contamination observed in unconstrained retrieval pipelines.
- **Hypothesis 3 (H3 — Evidence Sufficiency & Hallucination Prevention):** Pre-generation evidence sufficiency gating—requiring entity coverage, relationship presence, hop depth, and connected compound chains—significantly reduces hallucination on unanswerable financial questions compared to unconstrained generative QA.

---

## 2. Data Strategy & Structure-Aware Chunking

### SEC Ingestion Universe
The empirical corpus consists of **99 SEC regulatory filings** spanning 10 core public corporations across major economic sectors:
- **Universe Tickers:** `AAPL` (Apple), `AMZN` (Amazon), `GOOGL` (Alphabet), `JNJ` (Johnson & Johnson), `JPM` (JPMorgan Chase), `META` (Meta), `MSFT` (Microsoft), `NVDA` (NVIDIA), `TSLA` (Tesla), `XOM` (ExxonMobil).
- **Filing Types:** Annual Reports (Form 10-K) and Quarterly Reports (Form 10-Q).

### Structure-Aware Chunking Pipeline
Conventional token-window chunkers fracture financial disclosures arbitrarily, splitting item boundaries and disassociating footnotes from tabular data. FinGraph AI implements a structure-aware chunking pipeline (`src/ingestion/chunker.py`):
1. **Item & Section Detection:** Detects SEC regulatory sections (Item 1: Business, Item 1A: Risk Factors, Item 7: Management's Discussion and Analysis) using regex and HTML heading markers.
2. **Context Preservation:** Preserves section hierarchies and item headers within chunk payloads.
3. **Corpus Scale:** Produces **8,181 structured chunks** across the 99 filings.
4. **Temporal Metadata Attachment:** Every chunk is stamped with an immutable metadata envelope:
   ```python
   {
       "chunk_id": "chunk_aapl_10k_2026_item1a_042",
       "document_id": "sec_aapl_10k_2026",
       "ticker": "AAPL",
       "publication_date": "2026-10-24",
       "fiscal_period": "FY2026",
       "available_time": "2026-10-24T18:00:00+00:00",
       "item_header": "Item 1A. Risk Factors",
       "section": "Supply Chain Dependencies"
   }
   ```

---

## 3. Temporal Knowledge Graph (TKG) Construction

FinGraph AI constructs a directed, labeled, temporally indexed Knowledge Graph from the structured chunks.

### Knowledge Graph Scale
- **Unified Multi-Company Semantic KG (`data/processed/unified/fingraph_semantic_kg.json`):**
  - **10,054 Nodes**
  - **9,822 Directed Edges**
- **Single-Stock In-Depth AAPL Graph (`data/processed/unified/aapl_temporal_graph.json`):**
  - **1,495 Nodes**
  - **660 Directed Edges**

### Entity Resolution & Node Identification
Entities are normalized to canonical IDs to ensure deterministic graph linking:
- Universe companies are resolved to canonical ticker IDs: `company:AAPL`, `company:NVDA`, `company:MSFT`.
- Other entities (suppliers, products, risks, competitors) are resolved using typed hashing: `{node_type}:{normalized_name_hash}`.

### Edge Schema & Predicates
Directed edges represent extracted assertions between entities:
$$e = (u, v, r, t_{\text{event}}, t_{\text{avail}}, P)$$
where:
- $u, v \in V$: Source and target entity nodes.
- $r \in \mathcal{R}$: Canonical relationship predicate (`SUPPLIES`, `AFFECTED_BY`, `HAS_RISK`, `COMPETES_WITH`, `DEPENDS_ON`, `MANUFACTURES`, `PARTNERS_WITH`).
- $t_{\text{avail}}$: Official SEC public availability timestamp.
- $P$: Provenance properties dictionary containing `document_id`, `chunk_id`, and extraction confidence score $\delta \in [0.0, 1.0]$.

---

## 4. Hybrid GraphRAG Retrieval

FinGraph AI executes hybrid retrieval across dense vector space and structured graph space.

```
       Query (q) + Reference Time (t_ref)
                     |
       +-------------+-------------+
       |                           |
       v                           v
  Dense Vector Search        Graph BFS Traversal
  - all-MiniLM-L6-v2         - Canonical Entity Linking
  - Cosine Distance          - Point-in-Time Edge Gate:
  - Temporal Gate:             t_avail <= t_ref
    t_chunk <= t_ref         - Cycle Prevention (1-3 hops)
       |                           |
       +-------------+-------------+
                     |
                     v
       Reciprocal Rank Fusion (RRF)
            Score = Σ 1/(60 + rank)
                     |
                     v
       Deterministic Multi-Factor Reranker
```

### Reciprocal Rank Fusion (RRF)
Vector candidates ($R_v$) and graph-derived paths ($R_g$) are unified via RRF:
$$\text{RRF}(e) = \sum_{m \in \{v, g\}} \frac{\mathbb{I}(e \in R_m)}{k + \text{rank}_m(e)}$$
with smoothing parameter $k = 60$.

### Deterministic Multi-Factor Reranking
Candidates are reranked according to a multi-factor scoring function:
$$\text{Score}(e) = \alpha \cdot \text{RRF}_{\text{norm}}(e) + \beta \cdot \text{Overlap}(e) + \gamma \cdot \text{RelMatch}(e) + \delta \cdot \text{Conf}(e) - \epsilon \cdot \text{hop\_penalty}(e)$$
where:
- $\alpha = 0.20$ (Normalized RRF hybrid retrieval score)
- $\beta = 0.25$ (Entity lexical overlap between evidence text and query)
- $\gamma = 0.35$ (Relationship predicate alignment)
- $\delta = 0.15$ (Extraction confidence)
- $\epsilon = 0.05$ (Hop distance penalty applied to $\max(0, \text{hops}(e) - 1)$)

**Hard Temporal Drop Gate:**
$$\text{If } t_{\text{avail}}(e) > t_{\text{ref}} \implies \text{Score}(e) = -\infty$$
Evidence with timestamps exceeding the query cutoff date is eliminated immediately.

---

## 5. Evidence Sufficiency Gating & Compound Chain Validation

The pre-generation sufficiency gate (`src/qa/evidence_sufficiency.py`) verifies five critical criteria before invoking the language model:

### 1. Entity Coverage
Asserts that all detected query entities ($E_{\text{query}}$) have matching representations in the candidate evidence pool:
$$\text{Coverage}_{\text{entity}} = \frac{|E_{\text{query}} \cap E_{\text{evidence}}|}{|E_{\text{query}}|}$$
- In **Strict Mode (CFG0)**: Requires $\text{Coverage}_{\text{entity}} = 1.0$.
- In **Partial Mode (CFG2)**: Requires $\text{Coverage}_{\text{entity}} \ge 0.50$.

### 2. Relationship Coverage
Verifies that query relationship intents (e.g., `SUPPLIES`, `HAS_RISK`) match observed graph predicates or verified textual mentions.

### 3. Compound Evidence Chain Validation
In multi-relational queries, verifying individual relationships in isolation introduces spurious reasoning. For example, for the query:
*"What manufacturing risks affect the suppliers that supply Apple?"*
the evidence must demonstrate an unbroken directional chain:
$$\text{Supplier}_i \xrightarrow{\text{SUPPLIES}} \text{company:AAPL} \quad \land \quad \text{Supplier}_i \xrightarrow{\text{HAS\_RISK} \lor \text{AFFECTED\_BY}} \text{Risk}_j$$
If retrieved evidence contains $S_1 \xrightarrow{\text{SUPPLIES}} \text{Apple}$ and $S_2 \xrightarrow{\text{HAS\_RISK}} R$ where $S_1 \ne S_2$, the evidence chain is rejected as disconnected.

### 4. Reasoning Hop Depth
Verifies that the observed traversal depth in evidence paths meets or exceeds the query's required reasoning depth:
$$\text{hops}_{\text{observed}} \ge \text{hops}_{\text{expected}}$$

### 5. Point-in-Time & Provenance Completeness
Every candidate evidence item must have:
- Valid non-empty `document_id` and `chunk_id`.
- Valid `available_time` satisfying $t_{\text{avail}} \le t_{\text{ref}}$.

---

## 6. Controlled ReAct Retrieval Loop

When initial retrieval yields insufficient evidence, `RetrievalAgent` (`src/qa/agent.py`) executes a bounded ReAct-style refinement loop (maximum 2 refinement rounds):

| Failure Diagnosis | Agent Refinement Action | Execution Behavior |
| :--- | :--- | :--- |
| Missing Relationship | `retrieve_missing_relationship` | Directs graph traversal specifically along the unobserved predicate. |
| Insufficient Hop Depth | `increase_hop_depth` | Expands BFS traversal radius ($k \to k+1$, max 3). |
| Missing Entity | `retrieve_missing_entity` | Executes targeted dense vector lookup for the unrepresented entity. |
| General Defect | `broaden_retrieval` | Increases candidate retrieval pool from top-5 to top-10 items. |

If after 2 refinement iterations evidence remains insufficient, the system emits an immediate structured refusal.

---

## 7. Evidence-Grounded Answer Generation & Refusal Policy

### Refusal Mechanism (Zero-Token Execution)
When `package.sufficient == False`, generation bypasses the LLM entirely:
```json
{
  "answer": "Insufficient evidence in the retrieved corpus.",
  "confidence": 0.0,
  "evidence_ids": [],
  "reasoning_summary": "Retrieved evidence does not satisfy query entity, relationship, or reasoning depth constraints."
}
```
This guarantees zero hallucination on ungrounded queries and produces instantaneous response latency ($0.00$s).

### Grounded Prompting & Post-Validation
When sufficient, `AnswerGenerator` prompts the local `qwen3.5:4b` backend at temperature $0.0$, strictly constraining claims to provided evidence IDs. `AnswerValidator` validates the generated JSON:
1. `answer` is non-empty.
2. Every cited ID in `evidence_ids` exists in the retrieved evidence package (`unknown_evidence_ids == 0`).
3. Zero cited evidence items have `available_time > reference_time`.
4. `confidence` $\in [0.0, 1.0]$.

---

## 8. Evaluation Framework & Mathematical Metrics

### 8.1 Retrieval Performance Metrics
- **Hit@K ($K \in \{1, 5\}$):** Proportion of queries where at least one ground-truth chunk is retrieved within the top $K$ ranks.
- **Mean Reciprocal Rank (MRR):**
  $$\text{MRR} = \frac{1}{|Q|} \sum_{i=1}^{|Q|} \frac{1}{\text{rank}_i}$$
- **NDCG@5:** Evaluates ranking quality with relevance discounting.
- **Document Hit@5:** Proportion of queries where at least one ground-truth filing document is retrieved in top 5.
- **Cross-Document Fact Coverage:** Proportion of cross-document queries where required facts from multiple filings are simultaneously retrieved.

### 8.2 Generative QA & Refusal Metrics
- **Exact Match (EM):** String equivalence between normalized prediction and ground truth.
- **Token F1:** Harmonic mean of precision and recall over answer tokens.
- **Refusal Recall:** Ability to correctly refuse unanswerable questions:
  $$\text{Refusal Recall} = \frac{\text{True Refusals (TP)}}{\text{Total Unanswerable Questions (TP + FN)}}$$
- **Refusal Precision:** Proportion of refusals that were legitimately unanswerable:
  $$\text{Refusal Precision} = \frac{\text{True Refusals (TP)}}{\text{Total Issued Refusals (TP + FP)}}$$
- **Refusal Accuracy:** Overall decision accuracy across answerable and unanswerable questions:
  $$\text{Refusal Accuracy} = \frac{\text{TP} + \text{TN}}{\text{Total Questions}}$$

### 8.3 Temporal Safety & Leakage Metrics
- **Future Evidence Infiltration Rate (FEIR):**
  $$\text{FEIR} = \frac{\sum_{i=1}^{N} \sum_{e \in E_i} \mathbb{I}(t_e > t_{\text{ref}, i})}{\sum_{i=1}^{N} |E_i|}$$
  FinGraph AI enforces $\text{FEIR} = 0.00\%$ through hard point-in-time graph and vector gates.
- **Parametric Leakage Rate (PLR):** Evaluates intrinsic training-weight leakage when retrieval is disabled:
  $$\text{PLR} = \frac{1}{|Q_{\text{future}}|} \sum_{q \in Q_{\text{future}}} \mathbb{I}(\text{future\_facts} \in \text{LLM\_answer}_q)$$
  On post-cutoff benchmark queries, local LLMs demonstrate $\text{PLR} = 80.00\%$ without retrieval grounding, validating the imperative for strict evidence-sufficiency gating.
