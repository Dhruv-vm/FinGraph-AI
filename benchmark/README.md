# FinGraph AI Benchmark Dataset

## 1. Overview & Verification Summary

The FinGraph AI benchmark evaluates multi-hop question answering, temporal point-in-time reasoning, and parametric leakage prevention across 99 SEC 10-K and 10-Q filings.

In strict compliance with the Phase 5 methodological protocol, every benchmark question has been verified against source chunks in `data/processed/chunks` and Knowledge Graph edges in `data/processed/unified/fingraph_semantic_kg.json`.

### Audit & Verification Statistics
- **Drafted Candidate Questions:** 65
- **Reviewed Questions:** 65
- **Verified Questions:** 64
- **Rejected Questions:** 1
- **Answerable Questions:** 50
- **Unanswerable Refusal Cases:** 14

### Category Distribution
- **cross_document_comparative:** 12 questions
- **factual_single_hop:** 13 questions
- **relational_multi_hop:** 13 questions
- **relational_single_hop:** 13 questions
- **temporal_point_in_time:** 13 questions

---

## 2. Schema Specification

Every question adheres strictly to the following verified schema:

```json
{
  "question_id": "string (unique identifier)",
  "question": "string (natural language financial question)",
  "category": "factual_single_hop | relational_single_hop | temporal_point_in_time | relational_multi_hop | cross_document_comparative",
  "hop_level": 1 | 2 | 3,
  "entities": ["list of canonical entity IDs"],
  "reference_time": "ISO-8601 string or null",
  "expected_reasoning_hops": 1 | 2 | 3,
  "relevant_chunk_ids": ["list of verified chunk IDs for chunk-level Recall@K/Precision@K/MRR"],
  "relevant_document_ids": ["list of verified document IDs for doc-level retrieval metrics"],
  "expected_graph_path": [["source", "RELATION", "target"]],
  "ground_truth_answer": "string",
  "required_evidence": ["list of factual terms required in evidence"],
  "answerable": true | false,
  "unanswerability_rationale": "string or null",
  "verification_status": "verified"
}
```

---

## 3. Unanswerability & Refusal Ground Truth Protocol

To rigorously evaluate **Refusal Accuracy** and eliminate silent hallucinations:
1. **Pre-Corpus Temporal Cutoffs:** When a question specifies a historical reference date prior to a company's earliest filing in the corpus (e.g. `TPT-02` asking about NVIDIA as of June 1, 2025, where the earliest NVDA filing is August 27, 2025), the ground truth requires explicit refusal: `"Insufficient evidence in the retrieved corpus."`
2. **Missing Multi-Hop Relational Links:** When a question queries risks affecting third-party suppliers (e.g. `RMH-05`), customer filings disclose supplier identities and company-level risks, but third-party suppliers' internal operational risk disclosures are absent. Ground truth requires refusal.
3. **Fictitious / Counterfactual Entities:** Questions inquiring about non-existent corporate actions (e.g., Apple manufacturing passenger aircraft `RSH-12`) require refusal.

---

## 4. Benchmark Verification Audit Log

| Question ID | Category | Hops | Answerable | Verified Chunks | Status |
| :--- | :--- | :---: | :---: | :---: | :---: |
| `FSH-01` | `factual_single_hop` | 1 | True | 5 | `verified` |
| `FSH-02` | `factual_single_hop` | 1 | True | 5 | `verified` |
| `FSH-03` | `factual_single_hop` | 1 | True | 5 | `verified` |
| `FSH-04` | `factual_single_hop` | 1 | True | 5 | `verified` |
| `FSH-05` | `factual_single_hop` | 1 | True | 5 | `verified` |
| `FSH-06` | `factual_single_hop` | 1 | True | 5 | `verified` |
| `FSH-07` | `factual_single_hop` | 1 | True | 5 | `verified` |
| `FSH-08` | `factual_single_hop` | 1 | True | 5 | `verified` |
| `FSH-09` | `factual_single_hop` | 1 | True | 5 | `verified` |
| `FSH-10` | `factual_single_hop` | 1 | True | 5 | `verified` |
| `FSH-11` | `factual_single_hop` | 1 | True | 5 | `verified` |
| `FSH-12` | `factual_single_hop` | 1 | False | 0 | `verified` |
| `FSH-13` | `factual_single_hop` | 1 | False | 0 | `verified` |
| `RSH-01` | `relational_single_hop` | 1 | True | 5 | `verified` |
| `RSH-02` | `relational_single_hop` | 1 | True | 5 | `verified` |
| `RSH-03` | `relational_single_hop` | 1 | True | 5 | `verified` |
| `RSH-04` | `relational_single_hop` | 1 | True | 5 | `verified` |
| `RSH-05` | `relational_single_hop` | 1 | True | 5 | `verified` |
| `RSH-06` | `relational_single_hop` | 1 | True | 5 | `verified` |
| `RSH-07` | `relational_single_hop` | 1 | True | 5 | `verified` |
| `RSH-08` | `relational_single_hop` | 1 | True | 5 | `verified` |
| `RSH-09` | `relational_single_hop` | 1 | True | 4 | `verified` |
| `RSH-10` | `relational_single_hop` | 1 | True | 5 | `verified` |
| `RSH-11` | `relational_single_hop` | 1 | True | 5 | `verified` |
| `RSH-12` | `relational_single_hop` | 1 | False | 0 | `verified` |
| `RSH-13` | `relational_single_hop` | 1 | False | 0 | `verified` |
| `TPT-01` | `temporal_point_in_time` | 1 | True | 5 | `verified` |
| `TPT-02` | `temporal_point_in_time` | 1 | False | 0 | `verified` |
| `TPT-03` | `temporal_point_in_time` | 1 | True | 5 | `verified` |
| `TPT-04` | `temporal_point_in_time` | 1 | True | 5 | `verified` |
| `TPT-05` | `temporal_point_in_time` | 1 | True | 5 | `verified` |
| `TPT-06` | `temporal_point_in_time` | 1 | True | 5 | `verified` |
| `TPT-07` | `temporal_point_in_time` | 1 | True | 5 | `verified` |
| `TPT-08` | `temporal_point_in_time` | 1 | True | 5 | `verified` |
| `TPT-09` | `temporal_point_in_time` | 1 | True | 5 | `verified` |
| `TPT-10` | `temporal_point_in_time` | 1 | True | 5 | `verified` |
| `TPT-11` | `temporal_point_in_time` | 1 | True | 5 | `verified` |
| `TPT-12` | `temporal_point_in_time` | 1 | False | 0 | `verified` |
| `TPT-13` | `temporal_point_in_time` | 1 | False | 0 | `verified` |
| `RMH-01` | `relational_multi_hop` | 2 | True | 3 | `verified` |
| `RMH-02` | `relational_multi_hop` | 2 | True | 5 | `verified` |
| `RMH-03` | `relational_multi_hop` | 2 | True | 5 | `verified` |
| `RMH-04` | `relational_multi_hop` | 2 | True | 5 | `verified` |
| `RMH-05` | `relational_multi_hop` | 2 | False | 0 | `verified` |
| `RMH-06` | `relational_multi_hop` | 2 | True | 5 | `verified` |
| `RMH-07` | `relational_multi_hop` | 2 | True | 5 | `verified` |
| `RMH-08` | `relational_multi_hop` | 2 | True | 5 | `verified` |
| `RMH-09` | `relational_multi_hop` | 2 | True | 5 | `verified` |
| `RMH-10` | `relational_multi_hop` | 2 | True | 5 | `verified` |
| `RMH-11` | `relational_multi_hop` | 3 | False | 0 | `verified` |
| `RMH-12` | `relational_multi_hop` | 3 | False | 0 | `verified` |
| `RMH-13` | `relational_multi_hop` | 3 | False | 0 | `verified` |
| `CDC-01` | `cross_document_comparative` | 2 | True | 4 | `verified` |
| `CDC-02` | `cross_document_comparative` | 2 | True | 4 | `verified` |
| `CDC-03` | `cross_document_comparative` | 2 | True | 4 | `verified` |
| `CDC-04` | `cross_document_comparative` | 2 | True | 3 | `verified` |
| `CDC-05` | `cross_document_comparative` | 2 | True | 4 | `verified` |
| `CDC-06` | `cross_document_comparative` | 2 | True | 3 | `verified` |
| `CDC-07` | `cross_document_comparative` | 2 | True | 4 | `verified` |
| `CDC-08` | `cross_document_comparative` | 2 | True | 4 | `verified` |
| `CDC-10` | `cross_document_comparative` | 2 | True | 2 | `verified` |
| `CDC-11` | `cross_document_comparative` | 2 | False | 0 | `verified` |
| `CDC-12` | `cross_document_comparative` | 2 | False | 0 | `verified` |
| `CDC-13` | `cross_document_comparative` | 2 | False | 0 | `verified` |

### Rejected Candidate Log

| Question ID | Category | Rejection Reason |
| :--- | :--- | :--- |
| `CDC-09` | `cross_document_comparative` | Missing chunks for comparative term: ['Instagram', 'advertising products and services'] |
