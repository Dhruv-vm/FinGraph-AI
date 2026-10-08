# FinGraph AI Knowledge Graph Schema

## Node Schema

Each node in the FinGraph Temporal Knowledge Graph represents an entity identified in SEC filings:

| Field | Type | Description |
|---|---|---|
| `node_id` | `str` | Canonical entity identifier (`company:{TICKER}` for universe companies, or `{node_type}:{hash}` for other entities) |
| `node_type` | `str` | Canonical title-cased entity category (e.g., `Company`, `Product`, `Supplier`, `Risk`) |
| `properties` | `dict` | Entity attributes (`canonical_name`, `name`, `ticker`, `cik`, `sector`, `document_ids`, `chunk_ids`) |
| `event_time` | `str \| None` | Point-in-time timestamp of the historical business event (ISO format) |
| `available_time` | `str \| None` | Filing ingestion timestamp when the entity was officially available to the public |

### Supported Node Types
- `Company`: Publicly traded corporations and commercial entities
- `Product`: Commercial goods, platforms, software, and hardware
- `Supplier`: Supply chain vendors and component providers
- `Competitor`: Market rivals and industry peers
- `Risk`: Operational, geopolitical, regulatory, or macroeconomic risk factors
- `Person`: Corporate executives, directors, and key figures
- `FinancialMetric`: Quantified financial figures, revenue segments, and margins
- `Event`: Mergers, acquisitions, earnings releases, and product launches
- `Country`: Geographic jurisdictions and markets

---

## Edge Schema

Each directed edge represents an asserted semantic relationship extracted from an SEC filing chunk:

| Field | Type | Description |
|---|---|---|
| `edge_id` | `str` | Deterministic edge identifier (`{source}->{rel}->{target}:{hash}`) |
| `source` | `str` | Source `node_id` |
| `target` | `str` | Target `node_id` |
| `relationship` | `str` | Canonical uppercase relationship predicate |
| `event_time` | `str \| None` | Historical date when the relation occurred |
| `available_time` | `str` | SEC acceptance timestamp when the relation became publicly available |
| `properties` | `dict` | Relational metadata (`document_id`, `chunk_id`, `publication_date`, `fiscal_period`, `confidence`) |

### Supported Relationship Types
- `SUPPLIES`: Source supplies products or components to target
- `MANUFACTURES`: Source manufactures target product
- `DEPENDS_ON`: Source has an operational or supply chain dependency on target
- `COMPETES_WITH`: Source competes in the marketplace with target
- `PARTNERS_WITH`: Commercial alliance or strategic agreement
- `INVESTED_IN`: Equity or financial investment
- `ACQUIRED`: Corporate acquisition or merger
- `LOCATED_IN`: Geographic facility or operational headquarters
- `AFFECTED_BY`: Entity impacted by a risk factor or event
- `CAUSED`: Causal relationship between events
- `ANNOUNCED`: Corporate statement or public disclosure
- `HAS_RISK`: Operational, financial, or regulatory risk exposure
- `MENTIONED_IN`: Document citation

---

## Retrieval Evidence Schema

The retrieval layer exposes unified evidence objects across vector, graph, and hybrid sources:

```python
@dataclass
class RetrievalEvidence:
    evidence_id: str             # e.g., "hybrid:chunk_123", "vector:chunk_456", "graph:edge_789"
    source_type: str             # "vector", "graph", or "hybrid"
    score: float                 # Fused RRF score
    rank: int                    # 1-indexed overall retrieval rank
    text: str                    # Chunk passage or verbalized graph path
    document_id: str | None      # SEC document provenance
    chunk_id: str | None         # Source chunk provenance
    available_time: str | None   # Ingestion timestamp for point-in-time validation
    event_time: str | None       # Historical event timestamp
    publication_date: str | None # Official filing date
    source_url: str | None       # Source filing URL
    source_reference: str | None # Reference section / filing ID
    retrieval_sources: list[str] # ["vector"], ["graph"], or ["vector", "graph"]
    original_scores: dict[str, float]  # Individual scores per modality
    original_ranks: dict[str, int]     # Individual ranks per modality
    metadata: dict[str, Any]     # Extended metadata (e.g., paths, payload)
```
