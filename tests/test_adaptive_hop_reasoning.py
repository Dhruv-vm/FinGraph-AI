"""
Tests for Workstream B (Adaptive Hop-Depth Reasoning) and Workstream C (Regulatory Entity Linking).

Validates:
1. Direct relationships: verified 1-hop direct relationship satisfies expected 2-hop query.
2. Genuine two-hop paths: consecutive connected valid path accepted.
3. Genuine three-hop paths: consecutive connected 3-hop landlord/cloud path accepted.
4. Reversed edges: reversed relationship in compound chain fails closed.
5. Disconnected paths: gap in intermediate nodes fails closed.
6. Unsupported intermediaries: intermediary without verified provenance fails closed.
7. Regulatory entity linking: FTC, DOJ, European Commission recognized and require regulatory context.
8. Document framing: 'SEC filings' does not inject missing entity requirements.
"""

from __future__ import annotations

import pytest
from src.qa.evidence_sufficiency import EvidenceSufficiencyChecker, SufficiencyResult
from src.qa.query_analysis import QueryAnalysis
from src.retrieval.evidence import RetrievalEvidence
from src.retrieval.graph import GraphRetriever


def make_evidence(
    evidence_id: str,
    text: str,
    source_company: str | None = None,
    provenance_verified: bool = True,
    attribution_status: str = "verified_issuer",
    edges: list[dict] | None = None,
    path: list[str] | None = None,
    hop_count: int = 1,
) -> RetrievalEvidence:
    return RetrievalEvidence(
        evidence_id=evidence_id,
        source_type="hybrid" if edges else "vector",
        score=0.9,
        rank=1,
        text=text,
        document_id=f"doc:{source_company or 'corp'}:10k",
        chunk_id=f"{evidence_id}:chunk:0",
        available_time="2025-01-01T00:00:00+00:00",
        event_time=None,
        publication_date="2025-01-01",
        source_url=None,
        source_reference=None,
        retrieval_sources=["vector", "graph"] if edges else ["vector"],
        original_scores={},
        original_ranks={},
        metadata={
            "source_company": source_company,
            "provenance_verified": provenance_verified,
            "attribution_status": attribution_status,
            "graph_edges": edges or [],
            "path": path or [],
            "hop_count": hop_count,
        },
        source_company=source_company,
        provenance_verified=provenance_verified,
        attribution_status=attribution_status,
    )


def test_adaptive_direct_relationship_accepted():
    """A verified direct relationship between two companies satisfies reasoning depth in adaptive mode."""
    checker = EvidenceSufficiencyChecker(hop_mode="adaptive")
    analysis = QueryAnalysis(
        original_query="How is NVIDIA connected to TSMC through supply and manufacturing relationships?",
        entities=["company:NVDA", "company:TSM"],
        relationship_intent=["SUPPLIES", "DEPENDS_ON"],
        expected_hop_depth=2,
        is_multi_hop=True,
    )

    # Ev 1 is an NVIDIA filing directly naming TSMC as third-party foundry
    ev = make_evidence(
        evidence_id="ev_nvda_tsmc",
        text="We utilize foundries, such as Taiwan Semiconductor Manufacturing Company (TSMC), to manufacture our semiconductor wafers.",
        source_company="NVDA",
        provenance_verified=True,
        attribution_status="verified_issuer",
        edges=[{"source": "company:NVDA", "relationship": "DEPENDS_ON", "target": "company:TSM"}],
        hop_count=1,
    )

    res = checker.check(analysis=analysis, evidence=[ev])
    assert res.sufficient is True, f"Expected sufficient, got: {res.reason}"
    assert res.missing_entities == []


def test_adaptive_genuine_two_hop_path_accepted():
    """A genuine two-hop connected path (Target <- SUPPLIES - Supplier - HAS_RISK -> Risk) is accepted."""
    checker = EvidenceSufficiencyChecker(hop_mode="adaptive")
    analysis = QueryAnalysis(
        original_query="What risks affect suppliers that provide components to Apple?",
        entities=["company:AAPL"],
        relationship_intent=["SUPPLIES", "HAS_RISK"],
        expected_hop_depth=2,
        is_multi_hop=True,
    )

    # Edge 1: Foxconn supplies Apple
    # Edge 2: Foxconn has risk labor shortages
    edges = [
        {"source": "company:FOXCONN", "relationship": "SUPPLIES", "target": "company:AAPL"},
        {"source": "company:FOXCONN", "relationship": "HAS_RISK", "target": "risk:labor_shortage"},
    ]
    ev = make_evidence(
        evidence_id="ev_foxconn_risk",
        text="Foxconn supplies components to Apple and experiences risks of severe labor shortages and facility disruptions.",
        source_company="FOXCONN",
        provenance_verified=True,
        attribution_status="verified_issuer",
        edges=edges,
        path=["company:AAPL", "company:FOXCONN", "risk:labor_shortage"],
        hop_count=2,
    )

    res = checker.check(analysis=analysis, evidence=[ev])
    assert res.sufficient is True, f"Expected sufficient, got: {res.reason}"


def test_adaptive_genuine_three_hop_path_accepted():
    """A genuine 3-hop landlord-cloud-partner-risk chain is accepted."""
    checker = EvidenceSufficiencyChecker(hop_mode="adaptive")
    analysis = QueryAnalysis(
        original_query="What risks affect the real estate landlords leasing data centers to Microsoft's cloud infrastructure partners?",
        entities=["company:MSFT"],
        relationship_intent=["PARTNERS_WITH", "HAS_RISK"],
        expected_hop_depth=3,
        is_multi_hop=True,
    )

    ev_partner = make_evidence(
        evidence_id="ev_partner",
        text="Equinix partners with Microsoft for cloud infrastructure hosting.",
        source_company="MSFT",
        edges=[{"source": "company:EQUINIX", "relationship": "PARTNERS_WITH", "target": "company:MSFT"}],
        path=["company:EQUINIX", "company:MSFT"],
        hop_count=1,
    )
    ev_landlord = make_evidence(
        evidence_id="ev_landlord",
        text="Digital Realty leases data centers to Equinix and faces power supply shortage risks.",
        source_company="DLR",
        edges=[
            {"source": "company:DLR", "relationship": "LEASES_TO", "target": "company:EQUINIX"},
            {"source": "company:DLR", "relationship": "HAS_RISK", "target": "risk:power_shortage"},
        ],
        path=["company:DLR", "company:EQUINIX"],
        hop_count=2,
    )

    res = checker.check(analysis=analysis, evidence=[ev_partner, ev_landlord])
    assert res.sufficient is True, f"Expected sufficient, got: {res.reason}"


def test_adaptive_reversed_edges_rejected():
    """A reversed edge in compound supplier risk (Apple SUPPLIES Supplier instead of Supplier SUPPLIES Apple) fails closed."""
    checker = EvidenceSufficiencyChecker(hop_mode="adaptive")
    analysis = QueryAnalysis(
        original_query="What risks affect suppliers that provide components to Apple?",
        entities=["company:AAPL"],
        relationship_intent=["SUPPLIES", "HAS_RISK"],
        expected_hop_depth=2,
        is_multi_hop=True,
    )

    # Erroneous reversed edge: Apple supplies Foxconn
    edges = [
        {"source": "company:AAPL", "relationship": "SUPPLIES", "target": "company:FOXCONN"},
        {"source": "company:FOXCONN", "relationship": "HAS_RISK", "target": "risk:labor_shortage"},
    ]
    ev = make_evidence(
        evidence_id="ev_reversed",
        text="Apple supplies software to Foxconn.",
        source_company="AAPL",
        edges=edges,
        hop_count=1,
    )

    res = checker.check(analysis=analysis, evidence=[ev])
    assert res.sufficient is False
    assert "Insufficient compound relationship coverage" in res.reason


def test_adaptive_disconnected_paths_rejected():
    """A disconnected intermediary gap in compound multi-hop queries fails closed."""
    checker = EvidenceSufficiencyChecker(hop_mode="adaptive")
    analysis = QueryAnalysis(
        original_query="What risks affect suppliers that provide components to Apple?",
        entities=["company:AAPL"],
        relationship_intent=["SUPPLIES", "HAS_RISK"],
        expected_hop_depth=2,
        is_multi_hop=True,
    )

    # Disconnected edges: Supplier A supplies Apple, but risk applies to unrelated Supplier B
    edges = [
        {"source": "company:SUPPLIER_A", "relationship": "SUPPLIES", "target": "company:AAPL"},
        {"source": "company:SUPPLIER_B", "relationship": "HAS_RISK", "target": "risk:earthquake"},
    ]
    ev = make_evidence(
        evidence_id="ev_disconnected",
        text="Supplier A supplies Apple. In unrelated news, Supplier B experiences earthquake risks.",
        source_company="SUPPLIER_A",
        edges=edges,
        hop_count=1,
    )

    res = checker.check(analysis=analysis, evidence=[ev])
    assert res.sufficient is False
    assert "Insufficient compound relationship coverage" in res.reason


def test_adaptive_unsupported_intermediaries_rejected():
    """Intermediaries with unverified provenance or contradictory attribution fail closed."""
    checker = EvidenceSufficiencyChecker(hop_mode="adaptive")
    analysis = QueryAnalysis(
        original_query="What risks affect suppliers that provide components to Apple?",
        entities=["company:AAPL"],
        relationship_intent=["SUPPLIES", "HAS_RISK"],
        expected_hop_depth=2,
        is_multi_hop=True,
    )

    edges = [
        {"source": "company:SUPPLIER_X", "relationship": "SUPPLIES", "target": "company:AAPL"},
        {"source": "company:SUPPLIER_X", "relationship": "HAS_RISK", "target": "risk:default"},
    ]
    # Mark provenance_verified=False
    ev = make_evidence(
        evidence_id="ev_unverified",
        text="Supplier X supplies Apple and has default risks.",
        source_company="MSFT",
        provenance_verified=False,
        attribution_status="unsupported_cross_company",
        edges=edges,
        hop_count=2,
    )

    res = checker.check(analysis=analysis, evidence=[ev])
    assert res.sufficient is False


def test_regulatory_entity_linking_and_action_verification():
    """Regulatory entities (FTC, European Commission) require regulatory action context to be covered."""
    checker = EvidenceSufficiencyChecker(hop_mode="adaptive")
    analysis = QueryAnalysis(
        original_query="What software products manufactured by Meta Platforms are affected by regulatory actions from the European Commission or FTC?",
        entities=["company:META", "regulator:FTC", "regulator:EC"],
        relationship_intent=["AFFECTED_BY", "MANUFACTURES"],
        expected_hop_depth=2,
        is_multi_hop=True,
    )

    # 1. Incidental mention of FTC without regulatory action does not satisfy
    ev_incidental = make_evidence(
        evidence_id="ev_incidental",
        text="We attended an industry conference hosted near the Federal Trade Commission headquarters in Washington DC.",
        source_company="META",
    )
    res_incidental = checker.check(analysis=analysis, evidence=[ev_incidental])
    assert res_incidental.sufficient is False
    assert any("regulator:" in ent for ent in res_incidental.missing_entities)

    # 2. Substantive regulatory action disclosure regarding FTC satisfies disjunctive requirement
    ev_action = make_evidence(
        evidence_id="ev_action",
        text="The Federal Trade Commission (FTC) initiated an administrative proceeding and enforcement action alleging COPPA compliance violations regarding Instagram.",
        source_company="META",
        edges=[{"source": "company:META", "relationship": "AFFECTED_BY", "target": "regulator:FTC"}],
    )
    res_action = checker.check(analysis=analysis, evidence=[ev_action])
    assert res_action.sufficient is True, f"Expected sufficient, got: {res_action.reason}"


def test_sec_filing_document_framing_not_treated_as_missing_company():
    """SEC filing document framing ('across their separate SEC filings') does not demand SEC company evidence."""
    checker = EvidenceSufficiencyChecker(hop_mode="adaptive")
    analysis = QueryAnalysis(
        original_query="How do Microsoft and Amazon describe their competition in cloud services across their separate SEC filings?",
        entities=["company:MSFT", "company:AMZN"],
        relationship_intent=["COMPETES_WITH"],
        expected_hop_depth=1,
        is_multi_hop=False,
    )

    ev_msft = make_evidence(
        evidence_id="ev_msft",
        text="Microsoft competes with Amazon Web Services across enterprise commercial cloud.",
        source_company="MSFT",
        edges=[{"source": "company:MSFT", "relationship": "COMPETES_WITH", "target": "company:AMZN"}],
    )
    ev_amzn = make_evidence(
        evidence_id="ev_amzn",
        text="Amazon Web Services experiences intense commercial cloud competition with Microsoft Azure.",
        source_company="AMZN",
        edges=[{"source": "company:AMZN", "relationship": "COMPETES_WITH", "target": "company:MSFT"}],
    )

    res = checker.check(analysis=analysis, evidence=[ev_msft, ev_amzn])
    assert res.sufficient is True, f"Expected sufficient, got: {res.reason}"
    assert res.missing_entities == []
