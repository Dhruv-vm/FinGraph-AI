# FinGraph AI — Data Strategy

## 1. Data Architecture

FinGraph AI uses multiple complementary financial data sources rather than relying on a single dataset.

The data layers are:

1. Financial news and text
2. Regulatory financial documents
3. Market time series
4. Macroeconomic indicators
5. Financial reasoning benchmarks
6. Temporal news/event data

## 2. Primary Data Sources

### 2.1 SEC EDGAR

SEC EDGAR will provide primary financial documents and structured XBRL information.

Primary document types:

- 10-K
- 10-Q
- 8-K
- Other relevant corporate filings

Uses:

- Financial document retrieval
- Entity extraction
- Event extraction
- Financial relationship extraction
- Temporal knowledge graph construction
- RAG evidence

### 2.2 Financial PhraseBank

Financial PhraseBank will be used as a controlled benchmark for financial sentiment classification.

Labels:

- Positive
- Neutral
- Negative

The dataset will primarily be used for NLP benchmarking rather than as the primary temporal prediction corpus.

### 2.3 Market Data

Historical market data will provide:

- Open
- High
- Low
- Close
- Adjusted close where available
- Volume

Derived features may include:

- Returns
- Rolling volatility
- Momentum
- Volume changes
- Moving averages

### 2.4 FRED

FRED will provide macroeconomic variables where relevant.

Potential variables include:

- Interest rates
- Inflation
- Unemployment
- GDP-related indicators
- Other relevant economic series

### 2.5 FinQA

FinQA will be used as a financial numerical reasoning benchmark.

It will support evaluation of:

- Financial retrieval
- Evidence selection
- Numerical reasoning
- RAG-based financial question answering

### 2.6 GDELT

GDELT may be used as a temporal news/event source.

Its primary role will be:

- News event discovery
- Temporal event extraction
- Entity-event relationships
- Longitudinal analysis

The initial implementation will use a controlled subset rather than attempting to ingest the complete GDELT corpus.

## 3. Entity Universe

The initial research implementation will use a controlled set of publicly traded companies.

The company universe will be selected before data collection and documented explicitly.

Each company should have stable identifiers such as:

- Company name
- Ticker
- SEC CIK where applicable

## 4. Temporal Alignment

Every information item must have an explicit timestamp or date.

Important timestamps include:

- Publication timestamp
- Filing timestamp
- Filing period
- Market timestamp
- Event timestamp
- Retrieval timestamp

The system will distinguish between:

- Event time
- Publication time
- Observation time
- Prediction time

## 5. Prediction Data Leakage Policy

Prediction inputs must only contain information available at or before the prediction timestamp.

Future information must never be included in model features.

For a prediction made at time T:

```text
Allowed:
information.timestamp <= T

Not allowed:
information.timestamp > T