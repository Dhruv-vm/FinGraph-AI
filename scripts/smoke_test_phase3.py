from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import sys
import time

# Ensure project root is on sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.graph.snapshot import is_available, parse_datetime
from src.retrieval.graph import GraphRetriever
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.vector import VectorStore
from src.retrieval.reranker import DeterministicReranker
from src.retrieval.orchestrator import RetrievalOrchestrator
from src.qa.evidence_sufficiency import EvidenceSufficiencyChecker
from src.qa.query_analysis import QueryAnalyzer
from src.qa.agent import RetrievalAgent, QAEvidencePackage


def run_query_test(
    index: int,
    query: str,
    agent: RetrievalAgent,
    expected_ref_time: str | None = None,
) -> QAEvidencePackage:
    print(f"\n{'=' * 75}")
    print(f"[Query {index}] '{query}'")
    if expected_ref_time:
        print(f"Explicit Reference Cutoff: {expected_ref_time}")
    print(f"{'=' * 75}")

    t0 = time.perf_counter()
    pkg = agent.run(query=query, reference_time=expected_ref_time)
    t1 = time.perf_counter()

    # 1. Query Analysis
    qa = pkg.analysis
    print(f"1. Query Analysis:")
    print(f"   Entities detected:       {qa.entities}")
    print(f"   Relationship intents:    {qa.relationship_intent}")
    print(f"   Estimated hop depth:     {qa.expected_hop_depth} (multi_hop={qa.is_multi_hop})")
    print(f"   Question type:           {qa.question_type} (secondary={qa.secondary_question_types})")
    print(f"   Extracted reference time:{qa.reference_time}")
    print(f"   Cross-doc required:      {qa.cross_document_required}")

    # 2. Retrieval trace & iterations
    print(f"\n2. Agent Execution Trace (Rounds: {len(pkg.retrieval_trace)}):")
    for step in pkg.retrieval_trace:
        round_num = step.get("iteration", step.get("round", "?"))
        action = step.get("action", "unknown")
        ev_count = step.get("evidence_count", 0)
        suff = step.get("is_sufficient", None)
        print(f"   Round {round_num}: Action='{action}', Evidence count={ev_count}, Sufficient={suff}")

    # Assert bounded iterations
    assert len(pkg.retrieval_trace) <= 3, f"Agent exceeded maximum iteration bounds: {len(pkg.retrieval_trace)} rounds!"

    # 3. Sufficiency Result
    suff = pkg.sufficiency_result or SufficiencyResult(sufficient=pkg.sufficient, reason="N/A")
    print(f"\n3. Evidence Sufficiency Evaluation:")
    print(f"   Is Sufficient:           {suff.sufficient} (reason: {suff.reason})")
    print(f"   Missing entities:        {suff.missing_entities}")
    print(f"   Missing relationships:   {suff.missing_relationships}")
    print(f"   Hop depth:               observed={suff.observed_hops}, required={suff.required_hops}")
    print(f"   Provenance complete:     {suff.provenance_complete}")
    print(f"   Temporal valid:          {suff.temporal_valid}")

    # 4. Evidence Package Summary
    print(f"\n4. QA Evidence Package (Total: {pkg.evidence_count} items in {t1 - t0:.2f}s):")
    print(f"   Primary Evidence items:   {len(pkg.primary_evidence)}")
    print(f"   Supporting Evidence items:{len(pkg.supporting_evidence)}")

    all_ev = pkg.all_evidence
    for i, ev in enumerate(all_ev[:5], 1):
        rel = ev.metadata.get("relationship", "N/A")
        snippet = ev.text[:100].replace("\n", " ") + "..." if len(ev.text) > 100 else ev.text.replace("\n", " ")
        print(f"   [{i}] (score={ev.score:.4f}, hops={ev.metadata.get('hop_count', 1)}, rel={rel})")
        print(f"       Text:  {snippet}")
        print(f"       Doc:   {ev.document_id} | Chunk: {ev.chunk_id} | Avail: {ev.available_time}")

    # 5. Assertions: Provenance completeness
    assert len(all_ev) > 0, f"Query '{query}' returned zero evidence!"
    for ev in all_ev:
        assert ev.document_id, f"Missing document_id in evidence {ev.evidence_id}!"
        assert ev.chunk_id, f"Missing chunk_id in evidence {ev.evidence_id}!"
        assert ev.available_time, f"Missing available_time in evidence {ev.evidence_id}!"

    # 6. Strict Temporal Validity Assertion
    effective_ref = pkg.reference_time or qa.reference_time
    if effective_ref:
        ref_dt = parse_datetime(effective_ref)
        if ref_dt is not None:
            violations = 0
            for ev in all_ev:
                if not is_available(ev.available_time, ref_dt):
                    print(f"   [TEMPORAL VIOLATION] {ev.evidence_id} available {ev.available_time} > ref {effective_ref}")
                    violations += 1
            assert violations == 0, f"Detected {violations} temporal violations for query '{query}'!"
            print(f"   Strict Temporal Check:   0 violations verified against {effective_ref}")

    return pkg


def main() -> None:
    kg_path = ROOT / "data" / "processed" / "unified" / "fingraph_semantic_kg.json"
    vector_path = ROOT / "data" / "vector_store"

    print("=" * 75)
    print("FinGraph AI — Phase 3 Retrieval-and-Reasoning Real-Data Verification")
    print("=" * 75)

    if not kg_path.exists():
        print(f"Error: Unified KG file not found at {kg_path}")
        sys.exit(1)

    print(f"\n[1/3] Initializing Knowledge Graph and Hybrid Retrieval Layer...")
    t0 = time.perf_counter()
    graph_retriever = GraphRetriever(graph=kg_path)
    t1 = time.perf_counter()
    print(f"   KG Loaded in {t1 - t0:.2f}s: {len(graph_retriever.nodes):,} nodes")

    vector_store = None
    if vector_path.exists():
        try:
            vector_store = VectorStore(path=vector_path)
            print(f"   Vector Store: Loaded ({vector_store.count():,} chunks)")
        except Exception as e:
            print(f"   Vector Store Warning: {e}")

    hybrid_retriever = HybridRetriever(
        vector_store=vector_store,
        graph_retriever=graph_retriever,
        rrf_k=60,
    )

    print(f"\n[2/3] Initializing Phase 3 Retrieval Orchestrator & Agent...")
    query_analyzer = QueryAnalyzer(graph_retriever=graph_retriever)
    reranker = DeterministicReranker()
    sufficiency_checker = EvidenceSufficiencyChecker(graph_retriever=graph_retriever)

    orchestrator = RetrievalOrchestrator(
        hybrid_retriever=hybrid_retriever,
        query_analyzer=query_analyzer,
        reranker=reranker,
        sufficiency_checker=sufficiency_checker,
    )

    agent = RetrievalAgent(
        orchestrator=orchestrator,
        max_additional_iterations=2,
    )

    print(f"\n[3/3] Executing 6 Research Verification Queries on Real Graph...")

    # Query 1: Dependencies
    run_query_test(
        index=1,
        query="What companies does NVIDIA depend on?",
        agent=agent,
    )

    # Query 2: Risks
    run_query_test(
        index=2,
        query="What risks affect NVIDIA?",
        agent=agent,
    )

    # Query 3: Competitors
    run_query_test(
        index=3,
        query="Which companies compete with Apple?",
        agent=agent,
    )

    # Query 4: Broad connection
    run_query_test(
        index=4,
        query="How is NVIDIA connected to TSMC?",
        agent=agent,
    )

    # Query 5: Explicit temporal constraint
    run_query_test(
        index=5,
        query="What risks affected NVIDIA as of June 1, 2025?",
        agent=agent,
        expected_ref_time="2025-06-01T00:00:00+00:00",
    )

    # Query 6: Multi-hop reasoning
    pkg_q6 = run_query_test(
        index=6,
        query="What risks affect companies that supply NVIDIA?",
        agent=agent,
    )
    assert pkg_q6.analysis.expected_hop_depth >= 2, f"Q6 expected hop depth was {pkg_q6.analysis.expected_hop_depth}, expected >= 2!"
    assert pkg_q6.analysis.is_multi_hop is True, "Q6 was not classified as multi-hop!"

    print("\n" + "=" * 75)
    print("Smoke Test PASSED successfully!")
    print("=" * 75)


if __name__ == "__main__":
    main()
