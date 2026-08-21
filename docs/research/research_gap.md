# Research Gap

Existing financial AI approaches commonly focus on individual capabilities such as:

- financial sentiment analysis
- financial text classification
- semantic retrieval
- large language model reasoning
- knowledge graph construction
- financial prediction

However, several challenges remain.

## Gap 1 — Temporal Context

Financial relationships and events change over time.

A static knowledge graph can represent a relationship but may not adequately represent when that relationship was valid or how it changed.

## Gap 2 — Structured + Unstructured Information

Financial intelligence requires combining:

- structured market information
- company relationships
- financial events
- unstructured news and documents

Using only vector retrieval or only graph retrieval can result in incomplete context.

## Gap 3 — Evidence-Grounded Reasoning

Financial AI systems should connect generated conclusions to supporting evidence rather than relying solely on language-model knowledge.

## Gap 4 — Retrieval and Prediction

Retrieval systems and prediction systems are often evaluated separately.

FinGraph AI investigates whether information retrieved from temporal financial knowledge structures can contribute useful features and context for downstream prediction.

## Research Opportunity

FinGraph AI therefore investigates a unified pipeline combining:

Financial NLP
+
Temporal Knowledge Graph
+
Vector Retrieval
+
Graph Retrieval
+
RAG
+
Financial Prediction

with explicit evaluation of each component and the complete system.