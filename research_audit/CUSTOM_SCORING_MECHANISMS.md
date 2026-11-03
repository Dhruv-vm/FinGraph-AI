# FinGraph AI — Custom Scoring Mechanisms & Algorithmic Novelty

**Audit Date:** 2026-11-01  
**Specialist:** GraphRAG & Information Retrieval Specialist  

---

## 1. Entity-Balanced Reciprocal Rank Fusion (EB-RRF)

### Problem Solved: Comparative Query Starvation
In standard RRF, if a query asks: *"Compare Microsoft's capital expenditures against Apple's in fiscal 2025"*, dense retrieval often retrieves 8 high-scoring Microsoft chunks and zero Apple chunks due to slight lexical alignment asymmetries. The generator then fails to answer or hallucinates Apple's data.

### Algorithmic Specification:
```python
def apply_entity_balanced_allocation(
    candidates: list[RetrievalEvidence],
    target_entities: list[str],
    top_k: int = 8,
    min_per_entity: int = 3
) -> list[RetrievalEvidence]:
    # 1. Partition verified candidates by entity affiliation
    # 2. Reserve up to min_per_entity for each target
    # 3. Fill remaining slots from globally ranked candidates
    # 4. Return balanced set preserving provenance
```

### Empirical Impact:
Directly recovered question **CDC-05** from false refusal, enabling dual-entity synthesis while maintaining 100% provenance verification.

---

## 2. Adaptive Query-Intent Hop Sufficiency Gating

### Problem Solved: Over-Refusal of Direct Relationships
In early phases (Phase 5/6), a rigid hop gate required multi-hop path proofs for all questions flagged as relational. Consequently, queries that could be satisfied by a single verified direct relationship (e.g. *"Does NVIDIA supply GPUs to Microsoft?"*) were falsely refused because a 2-hop chain was not retrieved.

### Logic Matrix:
$$	ext{GateDecision}(Q, \mathcal{E}) = egin{cases} 
	ext{PASS} & 	ext{if } 	ext{EntitiesCovered}(Q, \mathcal{E}) \land 	ext{DirectRelationMatches}(Q, \mathcal{E}) \
	ext{PASS} & 	ext{if } 	ext{EntitiesCovered}(Q, \mathcal{E}) \land 	ext{CompoundChainMatches}(Q, \mathcal{E}) \
	ext{REFUSE} & 	ext{if } 
eg 	ext{EntitiesCovered}(Q, \mathcal{E}) \lor 	ext{EvidenceEmpty}(\mathcal{E}) \
	ext{REFUSE} & 	ext{if } 	ext{TemporalViolation}(\mathcal{E}, t_{	ext{ref}})
\end{cases}$$

### Empirical Verification:
- Recovered **RMH-09, RMH-07, RMH-06** from false refusals.
- Refusal Recall maintained at **14/14 (100.0%)** on truly unanswerable questions.
