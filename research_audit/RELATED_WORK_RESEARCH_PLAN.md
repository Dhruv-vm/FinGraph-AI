# FinGraph AI — Related Work Research Plan & Literature Taxonomy

**Audit Date:** 2026-11-01  
**Specialist:** Financial NLP Researcher & Reviewer  

---

## 1. Literature Taxonomy

The research paper situates FinGraph AI at the intersection of four mature and emerging research tracks:

```
                      ┌─────────────────────────────────────────┐
                      │    Financial Question Answering (NLP)    │
                      │       FinQA, ConvFinQA, TAT-QA          │
                      └────────────────────┬────────────────────┘
                                           │
                      ┌────────────────────┴────────────────────┐
                      │  Temporal Knowledge Graphs in Finance   │
                      │    FinKG, EDGAR-KG, Bitemporal KGs      │
                      └────────────────────┬────────────────────┘
                                           │
                      ┌────────────────────┴────────────────────┐
                      │  Graph Retrieval-Augmented Generation   │
                      │      GraphRAG, HippoRAG, Hybrid RRF     │
                      └────────────────────┬────────────────────┘
                                           │
                      ┌────────────────────┴────────────────────┐
                      │ Hallucination & Temporal Leakage Gating │
                      │  Lookahead Bias, Refusal Recall, FEIR   │
                      └─────────────────────────────────────────┘
```

---

## 2. Exhaustive Analysis of Prior Work

### Track 1: Financial NLP & Multi-Hop Question Answering
- **Key Works:** FinQA (Chen et al., EMNLP 2021), ConvFinQA (Chen et al., ACL 2022), TAT-QA (Zhu et al., ACL 2021).
- **Core Gap:** These benchmarks focus heavily on numerical arithmetic over single tables within a single page of an earnings release. They fail to test cross-filing, multi-company relational reasoning (e.g. supply dependencies and litigation risks spanning multiple years).

### Track 2: Knowledge Graphs in the Financial Domain
- **Key Works:** FinKG, EDGAR-KG, Financial Event Graph Mining.
- **Core Gap:** Prior financial KGs treat nodes and edges as static factual instances. They lack bitemporal valid-time versus transaction-time semantics, leading to lookahead bias when querying historical states.

### Track 3: Graph-Augmented Retrieval (GraphRAG)
- **Key Works:** Microsoft GraphRAG (Edge et al., 2024), HippoRAG (Gutiérrez et al., NeurIPS 2024), Hybrid Search (Cormack et al., 2009).
- **Core Gap:** Standard GraphRAG systems perform unconstrained community summarization. They do not implement entity-balanced rank fusion for comparative questions, nor do they support deterministic refusal gating when relational paths are incomplete.

### Track 4: Temporal Integrity and Hallucination Mitigation
- **Key Works:** Lookahead Bias in Financial ML (de Prado, 2018), Factual Grounding in Financial LLMs (Zhang et al., 2023).
- **FinGraph AI Contribution:** Formalizes the Future Evidence Infiltration Rate (FEIR) metric and proves zero leakage across all retrieval stages.
