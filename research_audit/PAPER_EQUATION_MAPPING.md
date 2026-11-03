# FinGraph AI — Paper Equation to Code Implementation Mapping

**Audit Date:** 2026-11-01  
**Auditor:** Mathematical Modelling Specialist  

---

This document establishes the precise correspondence between formal mathematical equations designated for the research paper and the actual executable source code in the repository.

| Equation No. in Paper | Formal Description | Mathematical Symbolism | Source File | Class / Method | Code Line / Context |
|---|---|---|---|---|---|
| **Eq. 1** | Vector Cosine Distance | $	ext{sim}(\mathbf{q}, \mathbf{d}) = rac{\mathbf{q}\cdot\mathbf{d}}{\|\mathbf{q}\|\|\mathbf{d}\|}$ | `src/retrieval/vector.py` | `VectorStore.create_collection` | Configures `models.Distance.COSINE` |
| **Eq. 2** | Dense Query Embedding | $\mathbf{q} = \phi(q)$ | `src/retrieval/vector.py` | `VectorStore.search` | `self.model.encode(query)` |
| **Eq. 3** | Graph Path Admissibility | $P = (v_0, e_1, v_1, \dots, e_L, v_L)$ | `src/retrieval/graph.py` | `GraphRetriever.find_relation_path` | Traverses adjacency list with edge filtering |
| **Eq. 4** | Reciprocal Rank Fusion | $	ext{RRF}(d) = \sum_{m} rac{1}{k + r_m(d)}$ | `src/retrieval/hybrid.py` | `reciprocal_rank_fusion` | Fuses dense vector ranks & KG neighborhood |
| **Eq. 5** | Entity-Balanced Quota | $\mathcal{S}_{	ext{alloc}} = igcup_{E_i} 	ext{Top}(\mathcal{A}(E_i), k_{\min})$ | `src/retrieval/hybrid.py` | `apply_entity_balanced_allocation` | Lines 20–85: Partitions candidates by target entities |
| **Eq. 6** | Deterministic Rerank Score| $S_{	ext{rerank}}(d) = S_{	ext{RRF}}(d) + lpha \Delta_{	ext{temp}} + eta \Omega_{	ext{prov}}$ | `src/retrieval/reranker.py` | `Reranker.rerank` | Composite scoring over text match, time, & entity |
| **Eq. 7** | Point-in-Time Availability| $t_{	ext{available}}(c) \le t_{	ext{as\_of}}$ | `src/data/graph/snapshot.py` | `is_available` | Lines 37–52: Strict timestamp comparison |
| **Eq. 8** | Bitemporal Snapshot Slicing| $G(t) = (V_t, E_t)$ | `src/data/graph/snapshot.py` | `build_temporal_snapshot` | Lines 55–125: Node & edge temporal pruning |
| **Eq. 9** | Future Evidence Infiltration| $	ext{FEIR} = rac{|\{e \mid t_{	ext{avail}}(e) > t_{	ext{ref}}\}|}{|\mathcal{E}|}$ | `src/evaluation/temporal.py` | `calculate_feir` | Lines 14–50: Quantifies post-inquiry violations |
| **Eq. 10** | Token Precision & Recall | $	ext{Prec} = rac{|\mathcal{P} \cap \mathcal{G}|}{|\mathcal{P}|}, 	ext{Rec} = rac{|\mathcal{P} \cap \mathcal{G}|}{|\mathcal{G}|}$ | `src/evaluation/qa.py` | `token_f1_score` | Lines 30–48: Bag-of-words token overlap |
| **Eq. 11** | Token F1 Harmonic Mean | $	ext{F1} = rac{2 \cdot 	ext{Prec} \cdot 	ext{Rec}}{	ext{Prec} + 	ext{Rec}}$ | `src/evaluation/qa.py` | `token_f1_score` | Line 47: Standard harmonic F1 mean |
| **Eq. 12** | Exact Match Score | $	ext{EM} = \mathbb{I}(	ext{norm}(\hat{y}) == 	ext{norm}(y^*))$ | `src/evaluation/qa.py` | `exact_match_score` | Lines 25–28: Binary string equality check |
| **Eq. 13** | Multi-Hop Sufficiency Gate | $G_{	ext{suff}}(Q, \mathcal{E}) \in \{0, 1\}$ | `src/qa/evidence_sufficiency.py`| `verify_evidence_sufficiency` | Evaluates entity coverage & relational support |
| **Eq. 14** | Citation Provenance Check | $	ext{ValidCitation}(\hat{y}, \mathcal{E})$ | `src/qa/validation.py` | `validate_generated_answer` | Verifies chunk IDs cited in generated tokens |
