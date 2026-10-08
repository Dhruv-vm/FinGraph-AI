# FinGraph AI System Architecture

## Overview

FinGraph AI is a Temporal Knowledge Graph and Hybrid GraphRAG framework designed for multi-hop financial question answering over SEC filings. The system addresses temporal leakage, semantic entity ambiguity, and relational multi-hop reasoning by combining structure-aware chunking, schema-constrained extraction, a unified Temporal Knowledge Graph (TKG), point-in-time temporal traversal, Reciprocal Rank Fusion (RRF) hybrid retrieval, deterministic reranking, evidence sufficiency checking, and a controlled ReAct-style retrieval loop.

```
+---------------------------------------------------------------------------------------+
|                                    Financial Sources                                  |
|                             (SEC 10-K, 10-Q, 8-K Filings)                             |
+---------------------------------------------------------------------------------------+
                                           |
                                           v
+---------------------------------------------------------------------------------------+
|                         Structure-Aware Processing & Chunking                         |
|                       (Item Headers, Sections, Ingestion Dates)                       |
+---------------------------------------------------------------------------------------+
                                           |
                                           v
+---------------------------------------------------------------------------------------+
|                             Temporal Metadata & Provenance                            |
|             (available_time, publication_date, fiscal_period, chunk_id)               |
+---------------------------------------------------------------------------------------+
                                           |
                                           v
+---------------------------------------------------------------------------------------+
|                      Schema-Constrained LLM Extraction Pipeline                       |
|                   (Companies, Products, Risks, Financial Relations)                   |
+---------------------------------------------------------------------------------------+
                                           |
                    +----------------------+----------------------+
                    |                                             |
                    v                                             v
+---------------------------------------+     +-----------------------------------------+
|     Temporal Knowledge Graph (TKG)    |     |          Vector Store (Qdrant)          |
|  - Canonical entity IDs (company:SYM) |     |  - Dense embeddings (MiniLM-L6-v2)      |
|  - Multi-hop relational graph edges   |     |  - Exact chunk payload preservation     |
|  - Strict available_time on edges     |     |  - Point-in-time datetime filter        |
+---------------------------------------+     +-----------------------------------------+
                    |                                             |
                    +----------------------+----------------------+
                                           |
                                           v
+---------------------------------------------------------------------------------------+
|                                  Hybrid Retrieval Layer                               |
|                                                                                       |
|  1. Query Entity Linking:                                                             |
|     - Matches tickers, company universe, corporate aliases, and KG nodes              |
|  2. Graph Retrieval (GraphRetriever):                                                 |
|     - BFS multi-hop traversal (1 to 3 hops) with cycle prevention                     |
|     - Strict point-in-time edge validation: available_time <= reference_time          |
|     - Candidate-bounded BFS search prevents combinatorial explosion                   |
|  3. Vector Retrieval (VectorStore):                                                   |
|     - Dense semantic similarity with point-in-time chunk filtering                    |
|  4. Reciprocal Rank Fusion (RRF):                                                     |
|     - Combines vector and graph ranks using k=60                                      |
|     - Chunk-level matching identifies multi-modal evidence                            |
+---------------------------------------------------------------------------------------+
                                           |
                                           v
+---------------------------------------------------------------------------------------+
|                                  Query Analysis Layer                                 |
|  - Lexical temporal constraint extraction (ISO, YYYY-MM-DD, QX YYYY, FY YYYY)         |
|  - Entity extraction & resolution to canonical KG nodes                               |
|  - Relationship-intent mapping from financial ontology                                |
|  - Reasoning hop depth estimation (1, 2, or 3 hops) & multi-hop detection             |
|  - Question classification (factual, temporal, relationship, comparison, multi-hop)   |
|  - Cross-document requirement determination                                           |
+---------------------------------------------------------------------------------------+
                                           |
                                           v
+---------------------------------------------------------------------------------------+
|                              Deterministic Reranking Layer                            |
|  - Hard temporal gate: rejects available_time > reference_time (Zero Leakage)         |
|  - Research scoring function:                                                         |
|    Score = 0.20 * NormRRF + 0.25 * EntOverlap + 0.35 * RelMatch + 0.15 * Conf          |
|            - 0.05 * HopPenalty                                                        |
|  - Full provenance preservation across all candidate evidence                         |
+---------------------------------------------------------------------------------------+
                                           |
                                           v
+---------------------------------------------------------------------------------------+
|                           Evidence Sufficiency Checking                               |
|  - Entity coverage evaluation (with corporate alias and ticker resolution)            |
|  - Relationship-intent coverage verification                                          |
|  - Reasoning depth verification (observed_hops >= required_hops)                      |
|  - Provenance completeness (document_id, chunk_id, available_time)                    |
|  - Point-in-time temporal compliance                                                  |
+---------------------------------------------------------------------------------------+
                                           |
                                           v
+---------------------------------------------------------------------------------------+
|                      Controlled ReAct Retrieval Orchestration                         |
|  - Bounded iteration loop: maximum 2 additional retrieval rounds                      |
|  - Whitelisted targeted retrieval actions:                                            |
|    * retrieve_missing_relationship                                                    |
|    * increase_hop_depth                                                               |
|    * retrieve_missing_entity                                                          |
|    * broaden_retrieval                                                                |
|  - Transparent execution trace recording                                              |
|  - Consolidated QAEvidencePackage with primary/supporting evidence & graph paths       |
+---------------------------------------------------------------------------------------+
```

## Retrieval Architecture Details

### 1. Query Entity Linking
The entity linking component resolves company names, tickers, corporate aliases, and named entities in natural language queries to canonical KG nodes:
- **Ticker Identification:** High-precision regex identifying exchange symbols against the canonical universe (`configs/company_universe.csv`).
- **Alias Resolution:** Dictionary resolution mapping common corporate names (e.g., "Google" $\to$ `GOOGL`, "TSMC" $\to$ `TSM`, "Meta Platforms" $\to$ `META`) to canonical node IDs (`company:TICKER`).
- **Fuzzy Suffix Normalization:** Normalizes legal corporate designations (`Inc`, `Corp`, `LLC`, `Holdings`) via `strip_company_legal_suffix`.
- **Node-Text Indexing:** Multi-word sliding n-grams (1 to 5 words) matched against an in-memory index of 29,000+ normalized names and aliases.

### 2. Temporal Multi-Hop Graph Traversal
The `GraphRetriever` implements deterministic breadth-first search (BFS) over incident graph edges:
- **Multi-Hop Traversal:** Supports path exploration from 1 to 3 hops.
- **Cycle Prevention:** Maintains traversal history in the current path to prevent cyclical loops.
- **Point-in-Time Temporal Enforcement:** Every candidate edge in a path must satisfy `is_available(edge.available_time, reference_time)`. If an intermediate hop edge was published after `reference_time`, the path is immediately pruned, completely preventing temporal leakage.
- **Candidate Cap:** Caps BFS queue at 1,000 candidate paths to prevent combinatorial state explosions on high-degree nodes while guaranteeing millisecond latency.
- **Relational Verbalization:** Formats graph paths into human- and LLM-interpretable relational assertions while retaining full document provenance (`document_id`, `chunk_id`, `fiscal_period`, `publication_date`, `confidence`).

### 3. Reciprocal Rank Fusion (RRF)
The `HybridRetriever` merges vector search and graph retrieval without requiring arbitrary score calibration:
$$RRF(d) = \sum_{m \in \{\text{vector}, \text{graph}\}} \frac{w_m}{k + \text{rank}_m(d)}$$
where $k = 60$ is the standard RRF constant, and $w_m$ represents modality weights.
- When an SEC chunk from vector search matches a relation extracted from that same chunk in the KG, the item is labeled as `source_type="hybrid"` and receives an additive fusion boost.
- Single-modality results are labeled `source_type="vector"` or `source_type="graph"`.
- Results are deterministically ranked and returned as `RetrievalEvidence` objects.

## Retrieval and Reasoning Layer (Phase 3)

### 4. Query Analysis (`QueryAnalyzer`)
Performs deterministic, rule-based extraction from financial queries:
- **Explicit Temporal Parsing:** Extracts ISO timestamps, exact dates (`Month DD, YYYY`), quarters (`Q1 2024` $\to$ `2024-03-31T23:59:59Z`), and fiscal years (`FY 2023`).
- **Entity Linking:** Maps query spans to canonical KG nodes and detects unresolved capitalized spans.
- **Relationship Intent:** Matches query tokens against ontology mappings (`DEPENDS_ON`, `SUPPLIES`, `COMPETES_WITH`, `HAS_RISK`, `AFFECTED_BY`, `PARTNERS_WITH`, `ACQUIRED`).
- **Reasoning Hop Depth:** Deterministically classifies queries as 1-hop, 2-hop, or 3-hop based on relational chains, connectors ("connected to", "chain of suppliers"), and multiple entities.
- **Question Classification:** Classifies questions into `factual`, `temporal`, `relationship`, `comparison`, or `multi-hop`.

### 5. Deterministic Evidence Reranking (`DeterministicReranker`)
Reranks fused evidence using an explainable multi-factor scoring function (Option A):
$$\text{hop\_penalty}(e) = \max(0, \text{hops}(e) - 1)$$
$$\text{Score}(e) = \alpha \cdot \text{RRF}_{\text{norm}}(e) + \beta \cdot \text{Overlap}(e) + \gamma \cdot \text{RelMatch}(e) + \delta \cdot \text{Conf}(e) - \epsilon \cdot \text{hop\_penalty}(e)$$
with weights:
- $\alpha = 0.20$ (Normalized RRF hybrid retrieval score)
- $\beta = 0.25$ (Entity lexical overlap between evidence and query)
- $\gamma = 0.35$ (Relationship intent match)
- $\delta = 0.15$ (Edge extraction confidence)
- $\epsilon = 0.05$ (Hop distance penalty applied to $\text{hop\_penalty}(e)$)

**Zero Temporal Leakage Gate:** If $t_e > t_{\text{ref}}$, the candidate evidence is immediately dropped ($\text{Score} = -\infty$).

### 6. Evidence Sufficiency Evaluation (`EvidenceSufficiencyChecker`)
Evaluates whether candidate evidence satisfies the information needs of the query:
1. **Entity Coverage:** Ensures all query entities are represented in the evidence pool (using canonical IDs, ticker aliases, and graph node names).
2. **Relationship Coverage:** Verifies that detected relationship intents have corresponding graph relations or textual mentions.
3. **Reasoning Hop Depth:** Verifies that observed evidence hop depth meets or exceeds required hop depth.
4. **Provenance Completeness:** Asserts that every evidence item contains non-empty `document_id`, `chunk_id`, and `available_time`.
5. **Point-in-Time Validity:** Asserts that zero items violate the reference time cutoff.

### 7. Controlled ReAct Retrieval Loop (`RetrievalAgent`)
Orchestrates an adaptive, bounded retrieval loop without unrestricted LLM tool use:
- **Initial Round:** Query analysis $\to$ Hybrid retrieval $\to$ Reranking $\to$ Sufficiency check.
- **Conditional Additional Rounds (Max 2):** If evidence is insufficient, formulates targeted follow-up queries:
  - If relationships are missing $\to$ `retrieve_missing_relationship`
  - If hop depth is insufficient $\to$ `increase_hop_depth`
  - If entities are missing $\to$ `retrieve_missing_entity`
  - Otherwise $\to$ `broaden_retrieval`
- **Output:** Encapsulates findings into `QAEvidencePackage` with primary evidence (top 5), supporting evidence, graph traversal paths, complete document provenance, execution trace, and temporal validity flags.

## Generative Financial QA & Verification Layer (Phase 4)

### 8. Evidence-Grounded Answer Generation (`AnswerGenerator`)
The generative layer transforms the temporally validated `QAEvidencePackage` into a structured, provenance-backed `QAAnswer`:
- **Local Ollama Backend:** Powered by `qwen3.5:4b` running locally at deterministic sampling ($T = 0$, `think=False`).
- **Hard Pre-Generation Temporal Gate:** Asserts `available_time <= reference_time` across all package evidence prior to prompt construction. If any future item exists, generation is aborted immediately.
- **Insufficient Evidence Refusal Policy:** If `package.sufficient == False` or evidence is empty, the generator immediately emits a structured refusal:
  ```json
  {
    "answer": "Insufficient evidence in the retrieved corpus.",
    "confidence": 0.0,
    "evidence_ids": [],
    "reasoning_summary": "Retrieved evidence does not satisfy query entity, relationship, or reasoning depth constraints."
  }
  ```
  This is completed in $0.00$s without invoking LLM tokens, preventing fabrication on out-of-scope or unverified queries.
- **Strict Evidence Injection:** Prompt strictly constrains the LLM to use only the provided evidence IDs and text snippets, forbidding external parametric facts.
- **Structured JSON Output:** Requires strictly valid JSON output with schema keys: `answer`, `confidence`, `evidence_ids`, `reasoning_summary`.

### 9. Post-Generation Answer Validation (`AnswerValidator`)
Every LLM response is submitted to rigorous post-generation schema, citation, and temporal verification:
- **JSON Schema Conformance:** Validates JSON syntax and required keys.
- **Citation Grounding:** Asserts that every factual claim cites one or more evidence IDs, and all cited IDs belong to the retrieved package (`unknown_evidence_ids == 0`).
- **Post-Generation Temporal Compliance:** Asserts that zero cited evidence items have `available_time > reference_time`.
- **Confidence Range Verification:** Confirms confidence scores lie strictly in $[0.0, 1.0]$.
- **Provenance Retention:** Retains complete document provenance (`document_id`, `chunk_id`, `available_time`, `source_type`) on all cited evidence objects for auditability and compliance.
