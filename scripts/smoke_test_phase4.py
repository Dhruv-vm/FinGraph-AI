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
from src.qa.generator import AnswerGenerator, QAAnswer
from src.qa.validation import AnswerValidator


def run_qa_pipeline_test(
    index: int,
    query: str,
    agent: RetrievalAgent,
    generator: AnswerGenerator,
    expected_ref_time: str | None = None,
) -> tuple[QAEvidencePackage, QAAnswer]:
    print(f"\n{'=' * 78}")
    print(f"[Query {index}] '{query}'")
    if expected_ref_time:
        print(f"Explicit Reference Cutoff: {expected_ref_time}")
    print(f"{'=' * 78}")

    t0 = time.perf_counter()
    pkg = agent.run(query=query, reference_time=expected_ref_time)
    t1 = time.perf_counter()

    # 1. Query Analysis
    qa = pkg.analysis
    print(f"1. Query Analysis:")
    print(f"   Entities detected:       {qa.entities}")
    print(f"   Relationship intents:    {qa.relationship_intent}")
    print(f"   Estimated hop depth:     {qa.expected_hop_depth} (multi_hop={qa.is_multi_hop})")
    print(f"   Question type:           {qa.question_type}")
    print(f"   Extracted reference time:{qa.reference_time}")

    # 2. Retrieval trace
    print(f"\n2. Agent Execution:")
    print(f"   Rounds executed:         {len(pkg.retrieval_trace)}")
    print(f"   Evidence pool count:     {pkg.evidence_count}")
    print(f"   Sufficient evidence:     {pkg.sufficient}")

    # Temporal pre-generation check assertion
    ref_dt = parse_datetime(pkg.reference_time) if pkg.reference_time else None
    future_sent_to_llm = 0
    if ref_dt is not None:
        for ev in pkg.all_evidence:
            if not is_available(ev.available_time, ref_dt):
                future_sent_to_llm += 1

    assert future_sent_to_llm == 0, (
        f"CRITICAL: {future_sent_to_llm} future evidence items in package "
        f"about to be passed to LLM for query '{query}'!"
    )

    # 3. Answer Generation
    t2 = time.perf_counter()
    answer = generator.generate(pkg)
    t3 = time.perf_counter()

    print(f"\n3. Structured Answer Generation (Time: {t3 - t2:.2f}s):")
    print(f"   Answer Text:             {answer.answer}")
    print(f"   Confidence:              {answer.confidence:.4f}")
    print(f"   Evidence IDs cited:      {answer.evidence_ids}")
    print(f"   Insufficient flag:       {answer.insufficient_evidence}")
    if answer.reasoning_summary:
        print(f"   Reasoning summary:       {answer.reasoning_summary}")

    # 4. Provenance & Citation Verification
    print(f"\n4. Citations & Provenance ({len(answer.citations)} cited records):")
    for i, c in enumerate(answer.citations, 1):
        print(f"   [{i}] ID: {c['evidence_id']} | Doc: {c['document_id']} | "
              f"Chunk: {c['chunk_id']} | Avail: {c['available_time']} | Src: {c['source_type']}")

    # 5. Strict Assertions
    val_status = answer.validation_status or {}
    print(f"\n5. Validation Status:")
    print(f"   Is Valid:                {val_status.get('is_valid')}")
    if val_status.get("errors"):
        print(f"   Errors:                  {val_status.get('errors')}")

    # Assertion 1: Provenance completeness (100%)
    for ev in pkg.all_evidence:
        assert ev.document_id, f"Missing document_id in evidence {ev.evidence_id}"
        assert ev.chunk_id, f"Missing chunk_id in evidence {ev.evidence_id}"
        assert ev.available_time, f"Missing available_time in evidence {ev.evidence_id}"

    for c in answer.citations:
        assert c.get("document_id"), f"Missing document_id in citation {c.get('evidence_id')}"
        assert c.get("chunk_id"), f"Missing chunk_id in citation {c.get('evidence_id')}"
        assert c.get("available_time"), f"Missing available_time in citation {c.get('evidence_id')}"

    # Assertion 2: Unknown evidence IDs check
    pkg_ids = {ev.evidence_id for ev in pkg.all_evidence}
    unknown_ids = [eid for eid in answer.evidence_ids if eid not in pkg_ids]
    assert len(unknown_ids) == 0, f"Model cited unknown evidence IDs: {unknown_ids}"

    # Assertion 3: Temporal safety on citations
    future_cited = 0
    if ref_dt is not None:
        for c in answer.citations:
            if not is_available(c.get("available_time"), ref_dt):
                future_cited += 1
    assert future_cited == 0, f"Model cited future evidence: {future_cited} items"

    # Assertion 4: Insufficient evidence behavior check
    if not pkg.sufficient:
        assert answer.insufficient_evidence is True, "Model fabricated answer despite insufficient evidence!"
        assert "Insufficient evidence" in answer.answer
        assert len(answer.evidence_ids) == 0

    return pkg, answer


def main() -> None:
    kg_path = ROOT / "data" / "processed" / "unified" / "fingraph_semantic_kg.json"
    vector_path = ROOT / "data" / "vector_store"

    print("=" * 78)
    print("FinGraph AI — Phase 4 Generative Financial QA Real-Data Verification")
    print("=" * 78)

    if not kg_path.exists():
        print(f"Error: Unified KG file not found at {kg_path}")
        sys.exit(1)

    print("\n[1/4] Loading Semantic Knowledge Graph and Hybrid Retriever...")
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

    print("\n[2/4] Initializing Query Analysis, Orchestrator, and ReAct Agent...")
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

    print("\n[3/4] Initializing Evidence-Grounded AnswerGenerator (Qwen3.5:4B)...")
    validator = AnswerValidator()
    generator = AnswerGenerator(
        model="qwen3.5:4b",
        temperature=0.0,
        validator=validator,
    )

    print("\n[4/4] Executing 6 Benchmark Queries on Local Ollama Model...")

    # Q1: Dependencies
    # Q1: Dependencies
    pkg_q1, ans_q1 = run_qa_pipeline_test(
        index=1,
        query="What companies does NVIDIA depend on?",
        agent=agent,
        generator=generator,
    )
    # Verify Issue 2: Q1 preserves relationship direction (does NOT treat NVDA supplying others as NVDA depending on them)
    for c in ans_q1.citations:
        c_text = (c.get("text_snippet") or "").lower()
        c_id = (c.get("evidence_id") or "").lower()
        if "supplies" in c_id and "nvda" in c_id:
            # Must NOT cite NVDA supplying other companies as dependency evidence
            assert not c_id.startswith("graph:edge:company:nvda->supplies->"), (
                f"Direction violation: Model cited NVDA supplying a customer as dependency: {c_id}"
            )

    # Q2: Risks
    run_qa_pipeline_test(
        index=2,
        query="What risks affect NVIDIA?",
        agent=agent,
        generator=generator,
    )

    # Q3: Competitors
    run_qa_pipeline_test(
        index=3,
        query="Which companies compete with Apple?",
        agent=agent,
        generator=generator,
    )

    # Q4: Connection
    run_qa_pipeline_test(
        index=4,
        query="How is NVIDIA connected to TSMC?",
        agent=agent,
        generator=generator,
    )

    # Q5: Historical Point-in-Time Cutoff
    run_qa_pipeline_test(
        index=5,
        query="What risks affected NVIDIA as of June 1, 2025?",
        agent=agent,
        generator=generator,
        expected_ref_time="2025-06-01T00:00:00+00:00",
    )

    # Q6: Multi-Hop Supplier Risks
    pkg_q6, ans_q6 = run_qa_pipeline_test(
        index=6,
        query="What risks affect companies that supply NVIDIA?",
        agent=agent,
        generator=generator,
    )
    assert pkg_q6.analysis.expected_hop_depth >= 2, "Q6 hop depth must be >= 2!"
    assert pkg_q6.analysis.is_multi_hop is True, "Q6 must be classified as multi-hop!"

    # Verify Issue 1 & Issue 8: Q6 must NOT answer supplier-only question ("Who supplies NVIDIA?").
    # It MUST be either genuinely risk-supported OR an insufficient-evidence refusal.
    if ans_q6.insufficient_evidence:
        assert "Insufficient evidence" in ans_q6.answer, "Refusal answer text expected"
        assert len(ans_q6.evidence_ids) == 0, "No evidence should be cited on refusal"
    else:
        # If answered, cited evidence MUST include genuine risk relations affecting suppliers
        has_risk_cited = any(
            "HAS_RISK" in str(c) or "AFFECTED_BY" in str(c) or "risk" in str(c).lower()
            for c in ans_q6.citations
        )
        assert has_risk_cited, "Q6 answered without citing risk evidence affecting suppliers!"

    print("\n" + "=" * 78)
    print("Phase 4 Smoke Test PASSED successfully!")
    print("Temporal violations: 0")
    print("Future evidence sent to LLM: 0")
    print("Future evidence cited: 0")
    print("Unknown evidence IDs: 0")
    print("Provenance completeness: 100%")
    print("=" * 78)


if __name__ == "__main__":
    main()
