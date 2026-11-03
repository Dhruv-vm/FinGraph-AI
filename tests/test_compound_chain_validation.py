import pytest
from src.qa.evidence_sufficiency import EvidenceSufficiencyChecker
from src.qa.query_analysis import QueryAnalysis
from src.retrieval.evidence import RetrievalEvidence


def test_valid_3hop_compound_chain_accepted():
    """A valid, connected 3-hop landlord-partner-target and landlord-risk chain is accepted."""
    checker = EvidenceSufficiencyChecker(relationship_mode="strict", hop_mode="soft", entity_mode="strict")
    analysis = QueryAnalysis(
        original_query="What risks affect the real estate landlords leasing data centers to Microsoft's cloud infrastructure partners?",
        entities=["company:MSFT"],
        relationship_intent=["PARTNERS_WITH", "HAS_RISK"],
        expected_hop_depth=3,
        is_multi_hop=True,
    )

    evidence = [
        RetrievalEvidence(
            evidence_id="ev_partner",
            source_type="graph",
            score=0.9,
            rank=1,
            text="Equinix partners with Microsoft to provide Azure express routes.",
            document_id="sec:msft:10k:2026",
            chunk_id="chunk:01",
            available_time="2026-01-01T00:00:00+00:00",
            event_time=None,
            publication_date="2026-01-01",
            source_url=None,
            source_reference=None,
            retrieval_sources=["graph"],
            original_scores={},
            original_ranks={},
            metadata={
                "graph_edges": [
                    {"source": "Equinix", "relationship": "PARTNERS_WITH", "target": "company:MSFT"},
                ],
                "path": ["Equinix", "company:MSFT"],
                "hop_count": 1,
            },
        ),
        RetrievalEvidence(
            evidence_id="ev_landlord",
            source_type="graph",
            score=0.85,
            rank=2,
            text="Digital Realty leases data centers to Equinix for cloud infrastructure deployment.",
            document_id="sec:msft:10k:2026",
            chunk_id="chunk:02",
            available_time="2026-01-01T00:00:00+00:00",
            event_time=None,
            publication_date="2026-01-01",
            source_url=None,
            source_reference=None,
            retrieval_sources=["graph"],
            original_scores={},
            original_ranks={},
            metadata={
                "graph_edges": [
                    {"source": "Digital Realty", "relationship": "LEASES_TO", "target": "Equinix"},
                ],
                "path": ["Digital Realty", "Equinix"],
                "hop_count": 1,
            },
        ),
        RetrievalEvidence(
            evidence_id="ev_risk",
            source_type="graph",
            score=0.8,
            rank=3,
            text="Digital Realty is affected by power grid stability risks and interest rate escalations.",
            document_id="sec:msft:10k:2026",
            chunk_id="chunk:03",
            available_time="2026-01-01T00:00:00+00:00",
            event_time=None,
            publication_date="2026-01-01",
            source_url=None,
            source_reference=None,
            retrieval_sources=["graph"],
            original_scores={},
            original_ranks={},
            metadata={
                "graph_edges": [
                    {"source": "Digital Realty", "relationship": "HAS_RISK", "target": "Power Grid Risks"},
                ],
                "path": ["Digital Realty", "Power Grid Risks"],
                "hop_count": 1,
            },
        ),
    ]

    res = checker.check(analysis, evidence)
    assert res.sufficient is True


def test_missing_intermediate_hop_rejected():
    """If the landlord-to-partner leasing hop is missing, strict mode rejects."""
    checker = EvidenceSufficiencyChecker(relationship_mode="strict", hop_mode="soft", entity_mode="strict")
    analysis = QueryAnalysis(
        original_query="What risks affect the real estate landlords leasing data centers to Microsoft's cloud infrastructure partners?",
        entities=["company:MSFT"],
        relationship_intent=["PARTNERS_WITH", "HAS_RISK"],
        expected_hop_depth=3,
        is_multi_hop=True,
    )

    # Missing the landlord-to-partner edge
    evidence = [
        RetrievalEvidence(
            evidence_id="ev_partner",
            source_type="graph",
            score=0.9,
            rank=1,
            text="Equinix partners with Microsoft.",
            document_id="sec:msft:10k:2026",
            chunk_id="chunk:01",
            available_time="2026-01-01T00:00:00+00:00",
            event_time=None,
            publication_date="2026-01-01",
            source_url=None,
            source_reference=None,
            retrieval_sources=["graph"],
            original_scores={},
            original_ranks={},
            metadata={
                "graph_edges": [
                    {"source": "Equinix", "relationship": "PARTNERS_WITH", "target": "company:MSFT"},
                ],
            },
        ),
        RetrievalEvidence(
            evidence_id="ev_risk",
            source_type="graph",
            score=0.8,
            rank=2,
            text="Digital Realty faces power risks.",
            document_id="sec:msft:10k:2026",
            chunk_id="chunk:02",
            available_time="2026-01-01T00:00:00+00:00",
            event_time=None,
            publication_date="2026-01-01",
            source_url=None,
            source_reference=None,
            retrieval_sources=["graph"],
            original_scores={},
            original_ranks={},
            metadata={
                "graph_edges": [
                    {"source": "Digital Realty", "relationship": "HAS_RISK", "target": "Power Risks"},
                ],
            },
        ),
    ]

    res = checker.check(analysis, evidence)
    assert res.sufficient is False
    assert "compound relationship" in res.reason.lower()


def test_disconnected_entities_rejected():
    """Landlord leases to Partner A, but Partner B connects to Microsoft -> Rejected."""
    checker = EvidenceSufficiencyChecker(relationship_mode="strict", hop_mode="soft", entity_mode="strict")
    analysis = QueryAnalysis(
        original_query="What risks affect the real estate landlords leasing data centers to Microsoft's cloud infrastructure partners?",
        entities=["company:MSFT"],
        relationship_intent=["PARTNERS_WITH", "HAS_RISK"],
        expected_hop_depth=3,
        is_multi_hop=True,
    )

    evidence = [
        RetrievalEvidence(
            evidence_id="ev_partner",
            source_type="graph",
            score=0.9,
            rank=1,
            text="Equinix partners with Microsoft.",
            document_id="sec:msft:10k:2026",
            chunk_id="chunk:01",
            available_time="2026-01-01T00:00:00+00:00",
            event_time=None,
            publication_date="2026-01-01",
            source_url=None,
            source_reference=None,
            retrieval_sources=["graph"],
            original_scores={},
            original_ranks={},
            metadata={
                "graph_edges": [
                    {"source": "Equinix", "relationship": "PARTNERS_WITH", "target": "company:MSFT"},
                ],
            },
        ),
        RetrievalEvidence(
            evidence_id="ev_landlord",
            source_type="graph",
            score=0.85,
            rank=2,
            text="Digital Realty leases to CyrusOne (unrelated partner).",
            document_id="sec:msft:10k:2026",
            chunk_id="chunk:02",
            available_time="2026-01-01T00:00:00+00:00",
            event_time=None,
            publication_date="2026-01-01",
            source_url=None,
            source_reference=None,
            retrieval_sources=["graph"],
            original_scores={},
            original_ranks={},
            metadata={
                "graph_edges": [
                    {"source": "Digital Realty", "relationship": "LEASES_TO", "target": "CyrusOne"},
                ],
            },
        ),
        RetrievalEvidence(
            evidence_id="ev_risk",
            source_type="graph",
            score=0.8,
            rank=3,
            text="Digital Realty faces power risks.",
            document_id="sec:msft:10k:2026",
            chunk_id="chunk:03",
            available_time="2026-01-01T00:00:00+00:00",
            event_time=None,
            publication_date="2026-01-01",
            source_url=None,
            source_reference=None,
            retrieval_sources=["graph"],
            original_scores={},
            original_ranks={},
            metadata={
                "graph_edges": [
                    {"source": "Digital Realty", "relationship": "HAS_RISK", "target": "Power Risks"},
                ],
            },
        ),
    ]

    res = checker.check(analysis, evidence)
    assert res.sufficient is False
    assert "compound relationship" in res.reason.lower()


def test_rmh_13_actual_corpus_fails_closed():
    """In actual corpus evidence where Microsoft 10-K lacks landlord risk chains, RMH-13 fails closed."""
    checker = EvidenceSufficiencyChecker(relationship_mode="strict", hop_mode="soft", entity_mode="strict")
    analysis = QueryAnalysis(
        original_query="What risks affect the real estate landlords leasing data centers to Microsoft's cloud infrastructure partners?",
        entities=["company:MSFT"],
        relationship_intent=["PARTNERS_WITH", "HAS_RISK"],
        expected_hop_depth=3,
        is_multi_hop=True,
    )

    # Actual retrieved evidence only mentions Microsoft cloud revenue and general Microsoft partners
    evidence = [
        RetrievalEvidence(
            evidence_id="ev1",
            source_type="vector",
            score=0.8,
            rank=1,
            text="Microsoft Cloud revenue was $35 billion, driven by Azure cloud partnerships.",
            document_id="sec:msft:10k:2026",
            chunk_id="chunk:42",
            available_time="2026-01-01T00:00:00+00:00",
            event_time=None,
            publication_date="2026-01-01",
            source_url=None,
            source_reference=None,
            retrieval_sources=["vector"],
            original_scores={},
            original_ranks={},
            metadata={"graph_edges": []},
        )
    ]

    res = checker.check(analysis, evidence)
    assert res.sufficient is False
    assert "compound relationship" in res.reason.lower()


def test_preserves_supplier_risk_queries_rmh_05_11_12():
    """Existing supplier-risk behavior for queries like RMH-11 remains intact and properly rejected."""
    checker = EvidenceSufficiencyChecker(relationship_mode="strict", hop_mode="soft", entity_mode="strict")
    analysis = QueryAnalysis(
        original_query="What internal manufacturing facility risks affect the contract semiconductor fabricators that supply Apple?",
        entities=["company:AAPL"],
        relationship_intent=["SUPPLIES", "HAS_RISK"],
        expected_hop_depth=3,
        is_multi_hop=True,
    )

    # Disconnected supplier text
    evidence = [
        RetrievalEvidence(
            evidence_id="ev_supp",
            source_type="graph",
            score=0.9,
            rank=1,
            text="Foxconn supplies assembly services to Apple.",
            document_id="sec:aapl:10k:2026",
            chunk_id="chunk:10",
            available_time="2026-01-01T00:00:00+00:00",
            event_time=None,
            publication_date="2026-01-01",
            source_url=None,
            source_reference=None,
            retrieval_sources=["graph"],
            original_scores={},
            original_ranks={},
            metadata={
                "graph_edges": [
                    {"source": "Foxconn", "relationship": "SUPPLIES", "target": "company:AAPL"},
                ],
            },
        ),
        RetrievalEvidence(
            evidence_id="ev_risk",
            source_type="graph",
            score=0.8,
            rank=2,
            text="Intel faces internal manufacturing facility delays.",
            document_id="sec:aapl:10k:2026",
            chunk_id="chunk:11",
            available_time="2026-01-01T00:00:00+00:00",
            event_time=None,
            publication_date="2026-01-01",
            source_url=None,
            source_reference=None,
            retrieval_sources=["graph"],
            original_scores={},
            original_ranks={},
            metadata={
                "graph_edges": [
                    {"source": "Intel", "relationship": "HAS_RISK", "target": "Facility Delays"},
                ],
            },
        ),
    ]

    res = checker.check(analysis, evidence)
    assert res.sufficient is False
    assert "compound relationship" in res.reason.lower()
    assert "disconnected" in res.reason.lower()
