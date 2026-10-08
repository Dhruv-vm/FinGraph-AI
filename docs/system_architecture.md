# FinGraph AI System Architecture

## Overview

FinGraph AI is a Temporal Knowledge Graph and Hybrid GraphRAG framework designed for multi-hop financial question answering over SEC filings. The system addresses temporal leakage, semantic entity ambiguity, and relational multi-hop reasoning by combining structure-aware chunking, schema-constrained extraction, a unified Temporal Knowledge Graph (TKG), point-in-time temporal traversal, and Reciprocal Rank Fusion (RRF) hybrid retrieval.

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
|     - Future-edge pruning prevents temporal leakage across path hops                  |
|  3. Vector Retrieval (VectorStore):                                                   |
|     - Dense semantic similarity with point-in-time chunk filtering                    |
|  4. Reciprocal Rank Fusion (RRF):                                                     |
|     - Combines vector and graph ranks using k=60                                      |
|     - Chunk-level matching identifies multi-modal evidence                            |
+---------------------------------------------------------------------------------------+
                                           |
                                           v
+---------------------------------------------------------------------------------------+
|                                Unified Retrieval Evidence                             |
|      (RetrievalEvidence objects with complete provenance, ranks, and source types)    |
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
- **Relational Verbalization:** Formats graph paths into human- and LLM-interpretable relational assertions while retaining full document provenance (`document_id`, `chunk_id`, `fiscal_period`, `publication_date`, `confidence`).

### 3. Reciprocal Rank Fusion (RRF)
The `HybridRetriever` merges vector search and graph retrieval without requiring arbitrary score calibration:
$$RRF(d) = \sum_{m \in \{\text{vector}, \text{graph}\}} \frac{w_m}{k + \text{rank}_m(d)}$$
where $k = 60$ is the standard RRF constant, and $w_m$ represents modality weights.
- When an SEC chunk from vector search matches a relation extracted from that same chunk in the KG, the item is labeled as `source_type="hybrid"` and receives an additive fusion boost.
- Single-modality results are labeled `source_type="vector"` or `source_type="graph"`.
- Results are deterministically ranked and returned as `RetrievalEvidence` objects.
