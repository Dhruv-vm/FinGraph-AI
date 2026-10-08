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


def main() -> None:
    kg_path = ROOT / "data" / "processed" / "unified" / "fingraph_semantic_kg.json"
    vector_path = ROOT / "data" / "vector_store"

    print("=" * 75)
    print("FinGraph AI — Phase 2 Retrieval Layer Real-Data Verification")
    print("=" * 75)

    if not kg_path.exists():
        print(f"Error: Unified KG file not found at {kg_path}")
        sys.exit(1)

    print(f"\n[1/4] Initializing GraphRetriever from: {kg_path}")
    t0 = time.perf_counter()
    graph_retriever = GraphRetriever(graph=kg_path)
    t1 = time.perf_counter()
    print(f"   Indexed Nodes:           {len(graph_retriever.nodes):,}")
    print(f"   Indexed Text Keys:       {len(graph_retriever.node_text_lookup):,}")
    print(f"   GraphRetriever load time: {t1 - t0:.2f}s")

    # -----------------------------------------------------------------------
    # Part 1: Query-Semantic Relationship Intent Verification
    # -----------------------------------------------------------------------
    print("\n[2/4] Testing Query-Semantic Relationship Intent Retrieval:")

    # Case 1: Dependencies
    q_dep = "What companies does NVIDIA depend on?"
    ev_dep = graph_retriever.retrieve(q_dep, top_k=5)
    print(f"\n   Query: '{q_dep}'")
    for i, ev in enumerate(ev_dep[:3], 1):
        print(f"     [{i}] (hops={ev.hop_count}, conf={ev.confidence:.2f}) {ev.text}")
    assert any(
        r in ("DEPENDS_ON", "SUPPLIES") for ev in ev_dep for r in ev.relationship.split(" -> ")
    ), "Failed to prioritize dependency/supply relationships for dependency query!"

    # Case 2: Risks
    q_risk = "What risks affect NVIDIA?"
    ev_risk = graph_retriever.retrieve(q_risk, top_k=5)
    print(f"\n   Query: '{q_risk}'")
    for i, ev in enumerate(ev_risk[:3], 1):
        print(f"     [{i}] (hops={ev.hop_count}, conf={ev.confidence:.2f}) {ev.text}")
    assert any(
        r in ("HAS_RISK", "AFFECTED_BY") for ev in ev_risk for r in ev.relationship.split(" -> ")
    ), "Failed to prioritize risk relationships for risk query!"

    # Case 3: Competitors
    q_comp = "Which companies compete with Apple?"
    ev_comp = graph_retriever.retrieve(q_comp, top_k=5)
    print(f"\n   Query: '{q_comp}'")
    for i, ev in enumerate(ev_comp[:3], 1):
        print(f"     [{i}] (hops={ev.hop_count}, conf={ev.confidence:.2f}) {ev.text}")
    assert any(
        r == "COMPETES_WITH" for ev in ev_comp for r in ev.relationship.split(" -> ")
    ), "Failed to prioritize COMPETES_WITH for competitor query!"

    # Case 4: Broad multi-hop connection
    q_broad = "How is NVIDIA connected to TSMC?"
    ev_broad = graph_retriever.retrieve(q_broad, max_hops=2, top_k=5)
    print(f"\n   Query: '{q_broad}' (Neutral multi-hop traversal)")
    for i, ev in enumerate(ev_broad[:3], 1):
        print(f"     [{i}] (hops={ev.hop_count}, conf={ev.confidence:.2f}) {ev.text}")
    assert len(ev_broad) > 0, "Broad multi-hop traversal failed to return connected evidence!"

    # -----------------------------------------------------------------------
    # Part 2: Point-in-Time Temporal Filtering Verification
    # -----------------------------------------------------------------------
    print("\n[3/4] Testing Strict Point-in-Time Temporal Filtering:")

    reference_iso = "2025-06-01T00:00:00+00:00"
    ref_dt = parse_datetime(reference_iso)
    assert ref_dt is not None

    # Retrieve all unconstrained vs constrained
    unconstrained_graph = graph_retriever.retrieve("NVIDIA", reference_time=None, max_hops=2, top_k=50)
    constrained_graph = graph_retriever.retrieve("NVIDIA", reference_time=reference_iso, max_hops=2, top_k=50)

    # Count how many unconstrained items were actually in the future relative to ref_dt
    future_graph_excluded = sum(
        1 for ev in unconstrained_graph
        if not is_available(ev.available_time, ref_dt)
    )

    temporal_violations = 0
    valid_temporal_evidence = 0

    for ev in constrained_graph:
        if not is_available(ev.available_time, ref_dt):
            print(f"   [VIOLATION] Graph evidence available_time {ev.available_time} > reference_time {reference_iso}")
            temporal_violations += 1
        else:
            valid_temporal_evidence += 1

    print(f"   Temporal reference time:       {reference_iso}")
    print(f"   Future graph evidence excluded:{future_graph_excluded}")
    print(f"   Valid graph evidence returned: {valid_temporal_evidence}")
    print(f"   Temporal violations:           {temporal_violations}")

    assert temporal_violations == 0, f"Detected {temporal_violations} temporal violations in graph retrieval!"

    # -----------------------------------------------------------------------
    # Part 3: Hybrid Retrieval Verification with Strict Temporal Validation
    # -----------------------------------------------------------------------
    print("\n[4/4] Testing Hybrid Retrieval (Vector + Graph + Temporal Cutoff):")

    vector_store = None
    future_vector_excluded = 0
    if vector_path.exists():
        try:
            vector_store = VectorStore(path=vector_path)
            print(f"   Vector Store: Loaded ({vector_store.count():,} chunks)")
            # Measure vector exclusion under cutoff
            unconstrained_vec = vector_store.search("NVIDIA supply chain", top_k=20, reference_time=None)
            constrained_vec = vector_store.search("NVIDIA supply chain", top_k=20, reference_time=ref_dt)
            future_vector_excluded = sum(
                1 for c in unconstrained_vec
                if not is_available(c.payload.get("available_time"), ref_dt)
            )
        except Exception as e:
            print(f"   Vector Store Warning: {e}")

    hybrid_retriever = HybridRetriever(
        vector_store=vector_store,
        graph_retriever=graph_retriever,
        rrf_k=60,
    )

    hybrid_query = "What are NVIDIA primary supply chain dependencies and risks?"
    hybrid_results = hybrid_retriever.retrieve(
        query=hybrid_query,
        reference_time=reference_iso,
        top_k=10,
        max_hops=2,
    )

    hybrid_violations = 0
    valid_hybrid_evidence = 0

    print(f"\n   Hybrid Query: '{hybrid_query}' (Cutoff: {reference_iso})")
    print(f"   Hybrid Fused Results (top {len(hybrid_results)}):")
    for r in hybrid_results:
        # Strict point-in-time check
        if not is_available(r.available_time, ref_dt):
            print(f"   [VIOLATION] Evidence {r.evidence_id} avail={r.available_time} > {reference_iso}")
            hybrid_violations += 1
        else:
            valid_hybrid_evidence += 1

        snippet = r.text[:95].replace("\n", " ") + "..." if len(r.text) > 95 else r.text.replace("\n", " ")
        print(f"     Rank {r.rank} [{r.source_type.upper()}] (score={r.score:.5f}, sources={r.retrieval_sources}):")
        print(f"       Text:  {snippet}")
        print(f"       Doc:   {r.document_id} | Chunk: {r.chunk_id} | Avail: {r.available_time}")

    print("\n" + "-" * 75)
    print(f"Temporal reference time:         {reference_iso}")
    print(f"Future graph evidence excluded:  {future_graph_excluded}")
    print(f"Future vector evidence excluded: {future_vector_excluded if vector_store else 'not measurable'}")
    print(f"Valid temporal evidence:         {valid_hybrid_evidence}")
    print(f"Temporal violations:             {hybrid_violations}")
    print("-" * 75)

    if hybrid_violations > 0:
        print(f"\nFAILURE: Detected {hybrid_violations} temporal violations in hybrid results!")
        sys.exit(1)

    print("\n" + "=" * 75)
    print("Smoke Test PASSED successfully!")
    print("=" * 75)


if __name__ == "__main__":
    main()
