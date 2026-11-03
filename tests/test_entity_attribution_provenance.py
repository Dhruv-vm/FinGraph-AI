from __future__ import annotations

import json
from pathlib import Path
import pytest

from src.data.graph.schema import GraphEdge, GraphNode, TemporalGraph
from src.qa.agent import RetrievalAgent
from src.qa.evidence_sufficiency import EvidenceSufficiencyChecker
from src.qa.generator import AnswerGenerator
from src.qa.query_analysis import QueryAnalyzer
from src.retrieval.evidence import GraphEvidence, RetrievalEvidence
from src.retrieval.graph import GraphRetriever
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.orchestrator import RetrievalOrchestrator
from src.retrieval.vector import RetrievedChunk, VectorStore


def test_1_original_cross_company_misattribution_fails_closed():
    """
    Test 1: Original cross-company misattribution.
    A Microsoft filing chunk mentions 'GPUs and datacenter servers', but does NOT mention NVIDIA.
    An erroneous extraction edge 'company:NVDA -> HAS_RISK -> Revenue' must be marked unverified
    and rejected by the evidence sufficiency checker for NVIDIA queries.
    """
    graph = TemporalGraph(
        nodes=[
            GraphNode(node_id="company:NVDA", node_type="Company", properties={"canonical_name": "NVIDIA Corporation", "ticker": "NVDA"}),
            GraphNode(node_id="company:MSFT", node_type="Company", properties={"canonical_name": "Microsoft Corporation", "ticker": "MSFT"}),
            GraphNode(node_id="financialmetric:revenue", node_type="FinancialMetric", properties={"canonical_name": "Revenue"}),
        ],
        edges=[
            GraphEdge(
                edge_id="erroneous_nvda_edge",
                source="company:NVDA",
                target="financialmetric:revenue",
                relationship="HAS_RISK",
                available_time="2025-04-30T20:08:52+00:00",
                properties={
                    "document_id": "doc:sec:076412151b515b96",
                    "chunk_id": "doc:sec:076412151b515b96:chunk:51",
                    "confidence": 0.90,
                },
            )
        ],
    )
    retriever = GraphRetriever(graph=graph)
    edge = graph.edges[0]
    ver, stat, issuer = retriever.verify_edge_provenance(edge)

    assert ver is False
    assert stat in ("unsupported_cross_company", "unsupported_cross_company_risk")
    assert issuer == "MSFT"

    # Verify that an unverified GraphEvidence item cannot satisfy sufficiency for NVIDIA query
    ev = RetrievalEvidence(
        evidence_id="graph:erroneous_nvda_edge",
        source_type="graph",
        score=0.85,
        rank=1,
        text="Our datacenters depend on predictable energy, networking supplies, and servers, including graphics processing units ('GPUs').",
        document_id="doc:sec:076412151b515b96",
        chunk_id="doc:sec:076412151b515b96:chunk:51",
        available_time="2025-04-30T20:08:52+00:00",
        source_company="MSFT",
        provenance_verified=False,
        attribution_status="unsupported_cross_company",
        metadata={
            "path": ["company:NVDA", "financialmetric:revenue"],
            "relationship": "HAS_RISK",
            "provenance_verified": False,
            "attribution_status": "unsupported_cross_company",
        },
    )

    analyzer = QueryAnalyzer(graph_retriever=retriever)
    analysis = analyzer.analyze("What risks affected NVIDIA as of June 1, 2025?", reference_time_override="2025-06-01T00:00:00+00:00")
    checker = EvidenceSufficiencyChecker(graph_retriever=retriever)

    res = checker.check(analysis, [ev])
    assert res.sufficient is False
    assert "company:NVDA" in res.missing_entities


def test_2_legitimate_cross_company_relationship_supported_by_text():
    """
    Test 2: Legitimate cross-company relationship explicitly supported by text.
    Apple 10-K filing mentions TSMC explicitly as its chip manufacturing supplier.
    This edge is verified and satisfies entity coverage for both AAPL and TSMC.
    """
    graph = TemporalGraph(
        nodes=[
            GraphNode(node_id="company:AAPL", node_type="Company", properties={"canonical_name": "Apple Inc.", "ticker": "AAPL"}),
            GraphNode(node_id="company:TSM", node_type="Company", properties={"canonical_name": "Taiwan Semiconductor Manufacturing Co.", "ticker": "TSM"}),
        ],
        edges=[
            GraphEdge(
                edge_id="valid_tsm_supplies_aapl",
                source="company:TSM",
                target="company:AAPL",
                relationship="SUPPLIES",
                available_time="2024-11-01T12:00:00+00:00",
                properties={
                    "document_id": "doc_aapl_10k_mock",
                    "chunk_id": "chunk_aapl_tsm_mock",
                    "source_company": "AAPL",
                    "provenance_verified": True,
                    "attribution_status": "verified_cross_company",
                    "confidence": 0.95,
                },
            )
        ],
    )
    retriever = GraphRetriever(graph=graph)
    checker = EvidenceSufficiencyChecker(graph_retriever=retriever)

    ev = RetrievalEvidence(
        evidence_id="hybrid:chunk_aapl_tsm_mock",
        source_type="hybrid",
        score=0.92,
        rank=1,
        text="We depend on single-source suppliers including TSMC (Taiwan Semiconductor) to manufacture custom silicon for our devices.",
        document_id="doc_aapl_10k_mock",
        chunk_id="chunk_aapl_tsm_mock",
        available_time="2024-11-01T12:00:00+00:00",
        source_company="AAPL",
        provenance_verified=True,
        attribution_status="verified_cross_company",
        metadata={
            "path": ["company:TSM", "company:AAPL"],
            "relationship": "SUPPLIES",
            "provenance_verified": True,
            "attribution_status": "verified_cross_company",
        },
    )

    analyzer = QueryAnalyzer(graph_retriever=retriever)
    analysis = analyzer.analyze("Does TSMC supply Apple?")
    res = checker.check(analysis, [ev])

    assert res.sufficient is True
    assert len(res.missing_entities) == 0


def test_3_conflicting_source_company_and_evidence_attribution_rejected():
    """
    Test 3: Graph edge whose source company and evidence attribution conflict.
    When a vector chunk from Microsoft is erroneously fused with an NVIDIA edge,
    the conflicting attribution must mark the item as unverified and fail sufficiency.
    """
    checker = EvidenceSufficiencyChecker()

    ev_conflicting = RetrievalEvidence(
        evidence_id="hybrid:conflict:1",
        source_type="hybrid",
        score=0.80,
        rank=1,
        text="Our cloud operations and server infrastructure expand rapidly across North America.",
        document_id="doc:sec:076412151b515b96",
        chunk_id="doc:sec:076412151b515b96:chunk:51",
        available_time="2025-04-30T20:08:52+00:00",
        source_company="MSFT",
        provenance_verified=False,
        attribution_status="conflicting_attribution",
        metadata={
            "path": ["company:NVDA", "financialmetric:revenue"],
            "relationship": "HAS_RISK",
            "provenance_verified": False,
            "attribution_status": "conflicting_attribution",
        },
    )

    from src.qa.query_analysis import QueryAnalysis
    analysis = QueryAnalysis(
        original_query="What risks affect NVIDIA?",
        entities=["company:NVDA"],
        relationship_intent=["HAS_RISK"],
        expected_hop_depth=1,
    )
    res = checker.check(analysis, [ev_conflicting])

    assert res.sufficient is False
    assert "company:NVDA" in res.missing_entities


def test_4_point_in_time_query_all_filings_after_cutoff_refuses():
    """
    Test 4: Point-in-time query where target company filings are all after cutoff.
    For NVIDIA as of June 1, 2025 (earliest filing is August 27, 2025):
    Evidence sufficiency gate MUST refuse with sufficient=False, and AnswerGenerator
    MUST produce the canonical refusal: 'Insufficient evidence in the retrieved corpus.'
    """
    from src.qa.query_analysis import QueryAnalysis
    analysis = QueryAnalysis(
        original_query="What risks affected NVIDIA as of June 1, 2025?",
        entities=["company:NVDA"],
        relationship_intent=["HAS_RISK", "AFFECTED_BY"],
        reference_time="2025-06-01T00:00:00+00:00",
        expected_hop_depth=1,
    )

    # Suppose retriever returned only post-cutoff or third-party MSFT chunks
    msft_chunk = RetrievalEvidence(
        evidence_id="vector:msft:chunk:128",
        source_type="vector",
        score=0.35,
        rank=1,
        text="Geopolitical conditions also increase the security risks described elsewhere in these risk factors.",
        document_id="doc:sec:076412151b515b96",
        chunk_id="doc:sec:076412151b515b96:chunk:128",
        available_time="2025-04-30T20:08:52+00:00",
        source_company="MSFT",
        provenance_verified=True,
        attribution_status="verified_issuer",
        metadata={"company": "MSFT"},
    )

    checker = EvidenceSufficiencyChecker()
    sufficiency = checker.check(analysis, [msft_chunk])

    assert sufficiency.sufficient is False
    assert "company:NVDA" in sufficiency.missing_entities

    generator = AnswerGenerator(llm_client=lambda p, s: '{"answer": "hallucinated", "confidence": 0.9}')
    from src.qa.agent import QAEvidencePackage
    pkg = QAEvidencePackage(
        query=analysis.original_query,
        reference_time=analysis.reference_time,
        question_type=analysis.question_type,
        expected_hops=analysis.expected_hop_depth,
        evidence=[msft_chunk],
        graph_paths=[],
        sources=[msft_chunk.document_id],
        retrieval_rounds=1,
        temporal_valid=True,
        sufficient=sufficiency.sufficient,
        trace=[],
        primary_evidence=[msft_chunk],
        supporting_evidence=[],
        sufficiency_result=sufficiency,
        retrieval_trace=[],
        analysis=analysis,
    )
    ans = generator.generate(pkg)
    assert ans.insufficient_evidence is True
    assert ans.answer == "Insufficient evidence in the retrieved corpus."
    assert ans.confidence == 0.0


def test_5_valid_historical_target_company_filing_passes_sufficiency():
    """
    Test 5: Valid historical target-company filing that should still support an answer.
    When querying Apple's risks as of January 1, 2025, Apple's 10-K from October 2024
    is eligible, verified, and correctly satisfies evidence sufficiency.
    """
    from src.qa.query_analysis import QueryAnalysis
    analysis = QueryAnalysis(
        original_query="What risks affected Apple as of January 1, 2025?",
        entities=["company:AAPL"],
        relationship_intent=["HAS_RISK"],
        reference_time="2025-01-01T00:00:00+00:00",
        expected_hop_depth=1,
    )

    aapl_chunk = RetrievalEvidence(
        evidence_id="vector:doc:sec:e0ba72210ca7deb2:chunk:45",
        source_type="vector",
        score=0.88,
        rank=1,
        text="The Company faces risks related to global supply chain disruptions and component shortages for its consumer electronic products.",
        document_id="doc:sec:e0ba72210ca7deb2",
        chunk_id="doc:sec:e0ba72210ca7deb2:chunk:45",
        available_time="2024-10-31T20:30:00+00:00",
        source_company="AAPL",
        provenance_verified=True,
        attribution_status="verified_issuer",
        metadata={"company": "AAPL", "relationship": "HAS_RISK", "graph_relations": ["HAS_RISK"]},
    )

    checker = EvidenceSufficiencyChecker(relationship_mode="strict")
    sufficiency = checker.check(analysis, [aapl_chunk])

    assert sufficiency.sufficient is True
    assert len(sufficiency.missing_entities) == 0
    assert len(sufficiency.missing_relationships) == 0


def test_6_missing_or_invalid_provenance_fails_closed():
    """
    Test 6: Missing or invalid provenance failing closed.
    Evidence items missing document ID or availability timestamp fail closed.
    """
    from src.qa.query_analysis import QueryAnalysis
    analysis = QueryAnalysis(
        original_query="What risks affected Apple?",
        entities=["company:AAPL"],
        relationship_intent=["HAS_RISK"],
        expected_hop_depth=1,
    )

    ev_no_doc = RetrievalEvidence(
        evidence_id="vector:unknown:1",
        source_type="vector",
        score=0.90,
        rank=1,
        text="Apple faces intense competition in global smartphone markets.",
        document_id=None,
        chunk_id=None,
        available_time=None,
        source_company="AAPL",
        provenance_verified=False,
        attribution_status="missing_provenance",
    )

    checker = EvidenceSufficiencyChecker()
    sufficiency = checker.check(analysis, [ev_no_doc])

    assert sufficiency.sufficient is False
    assert sufficiency.provenance_complete is False
    assert "provenance" in sufficiency.reason.lower()


def test_7_existing_multihop_and_supplier_risk_remains_functional():
    """
    Test 7: Existing multi-hop and supplier-risk cases remaining functional.
    A complete, connected 2-hop chain linking Target -> Supplier -> Risk with verified
    provenance passes the compound relationship sufficiency gate.
    """
    from src.qa.query_analysis import QueryAnalysis
    analysis = QueryAnalysis(
        original_query="What geopolitical or manufacturing risks affect the suppliers that provide components to Apple?",
        entities=["company:AAPL"],
        relationship_intent=["HAS_RISK", "SUPPLIES"],
        is_multi_hop=True,
        expected_hop_depth=2,
        reference_time="2025-01-01T00:00:00+00:00",
    )

    ev1 = RetrievalEvidence(
        evidence_id="graph:apple_supplier",
        source_type="graph",
        score=0.88,
        rank=1,
        text="Taiwan Semiconductor Manufacturing Company supplies advanced processors to Apple.",
        document_id="doc_mock_aapl",
        chunk_id="chunk_mock_1",
        available_time="2024-10-01T00:00:00+00:00",
        source_company="AAPL",
        provenance_verified=True,
        attribution_status="verified_cross_company",
        metadata={
            "hop_count": 2,
            "graph_edges": [
                {"source": "company:TSM", "relationship": "SUPPLIES", "target": "company:AAPL"}
            ]
        },
    )

    ev2 = RetrievalEvidence(
        evidence_id="graph:tsm_risk",
        source_type="graph",
        score=0.85,
        rank=2,
        text="Taiwan Semiconductor Manufacturing Company faces geopolitical tensions and natural disaster risks in Taiwan.",
        document_id="doc_mock_tsm",
        chunk_id="chunk_mock_2",
        available_time="2024-10-01T00:00:00+00:00",
        source_company="TSM",
        provenance_verified=True,
        attribution_status="verified_issuer",
        metadata={
            "graph_edges": [
                {"source": "company:TSM", "relationship": "HAS_RISK", "target": "risk:geopolitical"}
            ]
        },
    )

    checker = EvidenceSufficiencyChecker()
    res = checker.check(analysis, [ev1, ev2])

    assert res.sufficient is True
    assert res.provenance_complete is True
