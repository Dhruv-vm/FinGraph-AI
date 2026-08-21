# FinGraph AI — Research Problem

## Problem Statement

Financial information is distributed across heterogeneous and continuously changing sources such as financial news, company disclosures, market data, corporate events, and historical records.

Conventional financial NLP systems can extract sentiment, entities, and events from individual documents, while conventional retrieval systems primarily retrieve semantically similar information. However, these approaches often have limited ability to represent relationships between entities, reason over changing relationships across time, and connect retrieved evidence to financial predictions.

FinGraph AI addresses this problem by developing an evidence-driven financial intelligence framework that combines financial NLP, temporal knowledge graphs, graph-aware retrieval, retrieval-augmented generation, and machine learning.

The system aims to represent financial entities, relationships, events, and their temporal validity while combining structured graph information with unstructured financial documents.

The resulting system will support:

- temporal financial knowledge representation
- entity and event understanding
- graph and semantic retrieval
- evidence-grounded financial question answering
- explainable financial analysis
- machine-learning-based financial prediction

## Core Research Idea

Instead of treating financial documents, relationships, and market information as independent sources, FinGraph AI models them as interconnected and time-dependent information.

```text
Financial Data
      ↓
Data Processing
      ↓
Financial NLP
      ↓
Entities + Events + Relations
      ↓
Temporal Knowledge Graph
      ↓
Graph + Vector Retrieval
      ↓
Evidence-Grounded RAG
      ↓
Financial Prediction