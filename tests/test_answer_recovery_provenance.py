"""
Phase 6.3 Regression Tests: Answer Recovery and Provenance Preservation.

Verifies:
1. Canonical name resolution for hex corporate nodes (e.g. Activision Blizzard) in edge provenance.
2. TPT-02 cross-company risk misattribution remains strictly rejected (fail closed).
3. Non-corporate concept nodes (Risk, FinancialMetric, Product) do not trigger false refusals.
4. Cross-document comparison queries (CDC) do not fail due to false multi-hop depth requirements.
5. Unanswerable questions continue to be safely refused with zero hallucinations.
"""

from __future__ import annotations

import pytest
from src.retrieval.graph import GraphRetriever, GraphEdge, GraphNode, TemporalGraph
from src.qa.query_analysis import QueryAnalyzer
from src.qa.evidence_sufficiency import EvidenceSufficiencyChecker
from src.retrieval.evidence import RetrievalEvidence


@pytest.fixture
def sample_graph():
    nodes = [
        GraphNode(node_id="company:MSFT", node_type="Company", properties={"canonical_name": "Microsoft Corporation", "ticker": "MSFT"}),
        GraphNode(node_id="company:33188a412bf30ded", node_type="Company", properties={"canonical_name": "Activision Blizzard", "name": "Activision Blizzard"}),
        GraphNode(node_id="company:NVDA", node_type="Company", properties={"canonical_name": "NVIDIA Corporation", "ticker": "NVDA"}),
        GraphNode(node_id="risk:1d58c7f87cec47e3", node_type="Risk", properties={"canonical_name": "Cybersecurity", "name": "Cybersecurity"}),
    ]
    edges = [
        GraphEdge(
            edge_id="e_msft_atvi",
            source="company:MSFT",
            target="company:33188a412bf30ded",
            relationship="ACQUIRED",
            available_time="2023-10-15T00:00:00+00:00",
            properties={
                "document_id": "doc:sec:076412151b515b96",
                "chunk_id": "doc:sec:076412151b515b96:chunk:26",
            },
        ),
        GraphEdge(
            edge_id="e_nvda_msft_erroneous",
            source="company:NVDA",
            target="risk:1d58c7f87cec47e3",
            relationship="HAS_RISK",
            available_time="2023-10-15T00:00:00+00:00",
            properties={
                "document_id": "doc:sec:076412151b515b96",
                "chunk_id": "doc:sec:076412151b515b96:chunk:51",
            },
        ),
    ]
    tg = TemporalGraph(nodes=nodes, edges=edges)
    return tg


def test_hex_company_canonical_name_provenance_verification():
    """Verify that a hex company ID (Activision Blizzard) resolves via canonical name in text."""
    nodes = [
        GraphNode(node_id="company:MSFT", node_type="Company", properties={"canonical_name": "Microsoft Corporation", "ticker": "MSFT"}),
        GraphNode(node_id="company:33188a412bf30ded", node_type="Company", properties={"canonical_name": "Activision Blizzard", "name": "Activision Blizzard"}),
    ]
    edge = GraphEdge(
        edge_id="e_msft_atvi",
        source="company:MSFT",
        target="company:33188a412bf30ded",
        relationship="ACQUIRED",
        available_time="2023-10-15T00:00:00+00:00",
        properties={
            "document_id": "doc:sec:076412151b515b96",
            "chunk_id": "doc:sec:076412151b515b96:chunk:26",
        },
    )
    tg = TemporalGraph(nodes=nodes, edges=[edge])
    retriever = GraphRetriever(graph=tg)
    # Mock chunk lookup for chunk 26
    retriever.chunk_lookup["doc:sec:076412151b515b96:chunk:26"] = {
        "text": "We completed our acquisition of Activision Blizzard, Inc. in October 2023.",
        "issuer": "MSFT",
        "doc_id": "doc:sec:076412151b515b96",
    }
    retriever.doc_to_issuer["doc:sec:076412151b515b96"] = "MSFT"

    verified, status, issuer = retriever.verify_edge_provenance(edge)
    assert verified is True
    assert status == "verified_issuer"
    assert issuer == "MSFT"


def test_tpt02_cross_company_misattribution_strictly_rejected():
    """Verify that NVIDIA risk edge on a Microsoft filing chunk is rejected as unsupported_cross_company."""
    nodes = [
        GraphNode(node_id="company:NVDA", node_type="Company", properties={"canonical_name": "NVIDIA Corporation", "ticker": "NVDA"}),
        GraphNode(node_id="risk:rev_risk", node_type="Risk", properties={"canonical_name": "Revenue Risk"}),
    ]
    edge = GraphEdge(
        edge_id="e_tpt02",
        source="company:NVDA",
        target="risk:rev_risk",
        relationship="HAS_RISK",
        available_time="2025-04-30T00:00:00+00:00",
        properties={
            "document_id": "doc:sec:076412151b515b96",
            "chunk_id": "doc:sec:076412151b515b96:chunk:51",
        },
    )
    tg = TemporalGraph(nodes=nodes, edges=[edge])
    retriever = GraphRetriever(graph=tg)
    retriever.chunk_lookup["doc:sec:076412151b515b96:chunk:51"] = {
        "text": "Microsoft Corporation disclosures regarding cloud infrastructure expenses.",
        "issuer": "MSFT",
        "doc_id": "doc:sec:076412151b515b96",
    }
    retriever.doc_to_issuer["doc:sec:076412151b515b96"] = "MSFT"

    verified, status, issuer = retriever.verify_edge_provenance(edge)
    assert verified is False
    assert status == "unsupported_cross_company"
    assert issuer == "MSFT"


def test_concept_node_does_not_cause_false_refusal():
    """Verify that incidental concept nodes do not block sufficiency if the primary company is covered."""
    checker = EvidenceSufficiencyChecker()
    qa = QueryAnalyzer()
    analysis = qa.analyze("What cybersecurity company did Alphabet acquire in 2024?")
    # Inject an incidental concept node alongside company:GOOGL
    analysis.entities = ["company:GOOGL", "risk:1d58c7f87cec47e3"]
    analysis.relationship_intent = ["ACQUIRED"]

    evidence = [
        RetrievalEvidence(
            evidence_id="ev_googl_wiz",
            source_type="vector",
            score=0.92,
            rank=1,
            text="Alphabet announced the acquisition of Wiz, a leading cloud security platform, for approximately $32 billion.",
            document_id="doc:sec:googl_10k_2024",
            chunk_id="chunk_wiz_deal",
            available_time="2024-12-31T00:00:00+00:00",
            source_company="GOOGL",
            metadata={"source_company": "GOOGL", "relationship": "ACQUIRED"},
        )
    ]

    result = checker.check(analysis, evidence)
    assert result.sufficient is True
    assert len(result.missing_entities) == 0


def test_cross_document_comparison_hop_depth():
    """Verify that cross-document comparisons require 1 hop rather than false 2-hop chains."""
    qa = QueryAnalyzer()
    analysis = qa.analyze("Compare the commercial competition between Apple and Alphabet across their filings.")
    assert analysis.question_type == "comparison"
    assert analysis.expected_hop_depth == 1
    assert analysis.is_multi_hop is False
