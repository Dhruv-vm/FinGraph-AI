from __future__ import annotations

from datetime import datetime, timezone
import pytest

from src.data.graph.schema import GraphEdge, GraphNode, TemporalGraph
from src.retrieval.evidence import GraphEvidence
from src.retrieval.graph import GraphRetriever


@pytest.fixture
def sample_temporal_graph() -> TemporalGraph:
    """Fixture providing a known small temporal graph."""
    nodes = [
        GraphNode(
            node_id="company:NVDA",
            node_type="Company",
            properties={"canonical_name": "NVIDIA Corp", "name": "NVIDIA", "ticker": "NVDA"},
            available_time="2024-01-01T00:00:00+00:00",
        ),
        GraphNode(
            node_id="company:TSM",
            node_type="Company",
            properties={"canonical_name": "Taiwan Semiconductor Manufacturing Co", "name": "TSMC", "ticker": "TSM"},
            available_time="2024-01-01T00:00:00+00:00",
        ),
        GraphNode(
            node_id="company:ASML",
            node_type="Company",
            properties={"canonical_name": "ASML Holding NV", "name": "ASML", "ticker": "ASML"},
            available_time="2024-01-01T00:00:00+00:00",
        ),
        GraphNode(
            node_id="company:AAPL",
            node_type="Company",
            properties={"canonical_name": "Apple Inc", "name": "Apple", "ticker": "AAPL"},
            available_time="2024-01-01T00:00:00+00:00",
        ),
        GraphNode(
            node_id="risk:geopolitical_taiwan",
            node_type="Risk",
            properties={"canonical_name": "Taiwan Strait Geopolitical Tension"},
            available_time="2024-01-01T00:00:00+00:00",
        ),
        GraphNode(
            node_id="risk:cyber",
            node_type="Risk",
            properties={"canonical_name": "Cybersecurity Vulnerability"},
            available_time="2024-01-01T00:00:00+00:00",
        ),
        GraphNode(
            node_id="product:gpu",
            node_type="Product",
            properties={"canonical_name": "Hopper GPU"},
            available_time="2024-01-01T00:00:00+00:00",
        ),
    ]

    edges = [
        # TSM supplies NVDA (available in Q1 2024)
        GraphEdge(
            edge_id="e1_tsm_supplies_nvda",
            source="company:TSM",
            target="company:NVDA",
            relationship="SUPPLIES",
            available_time="2024-02-01T12:00:00+00:00",
            properties={
                "document_id": "doc_nvda_10k_2023",
                "chunk_id": "chunk_nvda_suppliers",
                "fiscal_period": "FY2023",
                "publication_date": "2024-02-01",
                "confidence": 0.95,
            },
        ),
        # ASML supplies TSM (available in Q1 2024)
        GraphEdge(
            edge_id="e2_asml_supplies_tsm",
            source="company:ASML",
            target="company:TSM",
            relationship="SUPPLIES",
            available_time="2024-02-15T12:00:00+00:00",
            properties={
                "document_id": "doc_tsm_20f_2023",
                "chunk_id": "chunk_tsm_equipment",
                "fiscal_period": "FY2023",
                "publication_date": "2024-02-15",
                "confidence": 0.90,
            },
        ),
        # Future edge: TSM affected by risk (available only in Q3 2024)
        GraphEdge(
            edge_id="e3_tsm_risk_future",
            source="company:TSM",
            target="risk:geopolitical_taiwan",
            relationship="AFFECTED_BY",
            available_time="2024-08-01T12:00:00+00:00",
            properties={
                "document_id": "doc_tsm_6k_q2_2024",
                "chunk_id": "chunk_tsm_taiwan_risk",
                "fiscal_period": "Q2 2024",
                "publication_date": "2024-08-01",
                "confidence": 0.85,
            },
        ),
        # Cycle edge for cycle testing: AAPL partners with NVDA and vice-versa
        GraphEdge(
            edge_id="e4_aapl_partners_nvda",
            source="company:AAPL",
            target="company:NVDA",
            relationship="PARTNERS_WITH",
            available_time="2024-01-10T12:00:00+00:00",
            properties={"confidence": 0.80},
        ),
        GraphEdge(
            edge_id="e5_nvda_partners_aapl",
            source="company:NVDA",
            target="company:AAPL",
            relationship="PARTNERS_WITH",
            available_time="2024-01-10T12:00:00+00:00",
            properties={"confidence": 0.80},
        ),
        # NVDA manufactures GPU (confidence 0.99)
        GraphEdge(
            edge_id="e6_nvda_manufactures_gpu",
            source="company:NVDA",
            target="product:gpu",
            relationship="MANUFACTURES",
            available_time="2024-01-15T12:00:00+00:00",
            properties={"confidence": 0.99},
        ),
        # NVDA has risk (confidence 0.90)
        GraphEdge(
            edge_id="e7_nvda_has_risk_cyber",
            source="company:NVDA",
            target="risk:cyber",
            relationship="HAS_RISK",
            available_time="2024-01-15T12:00:00+00:00",
            properties={"confidence": 0.90},
        ),
        # NVDA depends on TSM (confidence 0.92)
        GraphEdge(
            edge_id="e8_nvda_depends_tsm",
            source="company:NVDA",
            target="company:TSM",
            relationship="DEPENDS_ON",
            available_time="2024-01-15T12:00:00+00:00",
            properties={"confidence": 0.92},
        ),
        # AAPL competes with NVDA (confidence 0.88)
        GraphEdge(
            edge_id="e9_aapl_competes_nvda",
            source="company:AAPL",
            target="company:NVDA",
            relationship="COMPETES_WITH",
            available_time="2024-01-15T12:00:00+00:00",
            properties={"confidence": 0.88},
        ),
    ]

    return TemporalGraph(nodes=nodes, edges=edges)


@pytest.fixture
def mock_universe() -> list[dict[str, str]]:
    return [
        {
            "entity_id": "company:NVDA",
            "ticker": "NVDA",
            "company": "NVIDIA Corporation",
            "sector": "Information Technology",
            "cik": "0001045810",
        },
        {
            "entity_id": "company:TSM",
            "ticker": "TSM",
            "company": "Taiwan Semiconductor Manufacturing Co Ltd",
            "sector": "Information Technology",
            "cik": "0001046179",
        },
        {
            "entity_id": "company:ASML",
            "ticker": "ASML",
            "company": "ASML Holding NV",
            "sector": "Information Technology",
            "cik": "0000937966",
        },
        {
            "entity_id": "company:AAPL",
            "ticker": "AAPL",
            "company": "Apple Inc",
            "sector": "Information Technology",
            "cik": "0000320193",
        },
    ]


def test_entity_linking_exact_ticker(sample_temporal_graph, mock_universe):
    retriever = GraphRetriever(graph=sample_temporal_graph, company_universe=mock_universe)
    resolved, unresolved = retriever.link_entities("What are NVDA and AAPL supply risks?")

    assert "company:NVDA" in resolved
    assert "company:AAPL" in resolved
    assert len(unresolved) == 0


def test_entity_linking_company_name_and_aliases(sample_temporal_graph, mock_universe):
    retriever = GraphRetriever(graph=sample_temporal_graph, company_universe=mock_universe)
    resolved, _ = retriever.link_entities("Does Taiwan Semiconductor supply Nvidia?")

    assert "company:NVDA" in resolved
    assert "company:TSM" in resolved


def test_entity_linking_unresolved_entities(sample_temporal_graph, mock_universe):
    retriever = GraphRetriever(graph=sample_temporal_graph, company_universe=mock_universe)
    resolved, unresolved = retriever.link_entities("How does QuantumFab affect NVDA?")

    assert "company:NVDA" in resolved
    assert "QuantumFab" in unresolved


def test_one_hop_retrieval(sample_temporal_graph, mock_universe):
    retriever = GraphRetriever(graph=sample_temporal_graph, company_universe=mock_universe)
    evidence = retriever.retrieve(
        query="Who supplies NVIDIA?",
        seed_node_ids=["company:NVDA"],
        max_hops=1,
    )

    assert len(evidence) >= 1
    # Check that TSM -> NVDA is found
    tsm_edge = next((ev for ev in evidence if "company:TSM" in ev.path and "company:NVDA" in ev.path), None)
    assert tsm_edge is not None
    assert tsm_edge.hop_count == 1
    assert tsm_edge.relationship == "SUPPLIES"
    assert tsm_edge.chunk_id == "chunk_nvda_suppliers"


def test_multi_hop_two_hop_traversal(sample_temporal_graph, mock_universe):
    retriever = GraphRetriever(graph=sample_temporal_graph, company_universe=mock_universe)
    # ASML -> TSM -> NVDA is a 2-hop path to NVDA
    evidence = retriever.retrieve(
        query="What equipment makers indirectly supply NVIDIA?",
        seed_node_ids=["company:NVDA"],
        max_hops=2,
    )

    two_hop_paths = [ev for ev in evidence if ev.hop_count == 2]
    assert len(two_hop_paths) > 0

    asml_path = next((ev for ev in two_hop_paths if "company:ASML" in ev.path), None)
    assert asml_path is not None
    assert asml_path.hop_count == 2
    assert "company:TSM" in asml_path.path
    assert asml_path.confidence == pytest.approx(0.95 * 0.90)


def test_max_hops_constraint(sample_temporal_graph, mock_universe):
    retriever = GraphRetriever(graph=sample_temporal_graph, company_universe=mock_universe)
    evidence_hop1 = retriever.retrieve(query="NVDA", max_hops=1)
    assert all(ev.hop_count == 1 for ev in evidence_hop1)

    evidence_hop2 = retriever.retrieve(query="NVDA", max_hops=2)
    assert any(ev.hop_count == 2 for ev in evidence_hop2)


def test_cycle_handling_does_not_loop(sample_temporal_graph, mock_universe):
    retriever = GraphRetriever(graph=sample_temporal_graph, company_universe=mock_universe)
    # NVDA <-> AAPL form a cycle
    evidence = retriever.retrieve(
        query="AAPL NVDA partnership",
        seed_node_ids=["company:AAPL"],
        max_hops=3,
    )

    # All paths should have distinct nodes in path
    for ev in evidence:
        assert len(ev.path) == len(set(ev.path)), f"Cycle detected in path: {ev.path}"


def test_point_in_time_single_hop_filtering(sample_temporal_graph, mock_universe):
    retriever = GraphRetriever(graph=sample_temporal_graph, company_universe=mock_universe)

    # Query before TSM supplies NVDA became available (2024-02-01)
    evidence_early = retriever.retrieve(
        query="TSMC",
        reference_time="2024-01-15T00:00:00+00:00",
        max_hops=1,
    )
    # e1_tsm_supplies_nvda was available 2024-02-01, should NOT be present
    assert not any(ev.evidence_id == "edge:e1_tsm_supplies_nvda" for ev in evidence_early)

    # Query after TSM supplies NVDA became available
    evidence_later = retriever.retrieve(
        query="TSMC",
        reference_time="2024-02-05T00:00:00+00:00",
        max_hops=1,
    )
    assert any(ev.evidence_id == "edge:e1_tsm_supplies_nvda" for ev in evidence_later)


def test_point_in_time_multi_hop_prunes_future_edges(sample_temporal_graph, mock_universe):
    retriever = GraphRetriever(graph=sample_temporal_graph, company_universe=mock_universe)

    # Reference time is Q1 2024 (2024-03-01):
    # e1 (TSM->NVDA, 2024-02-01) is valid.
    # e3 (TSM->risk, 2024-08-01) is in the FUTURE.
    evidence_as_of_march = retriever.retrieve(
        query="NVDA risks",
        seed_node_ids=["company:NVDA"],
        reference_time="2024-03-01T00:00:00+00:00",
        max_hops=2,
    )

    # The 2-hop path NVDA -> TSM -> risk:geopolitical_taiwan MUST NOT be formed
    assert not any("risk:geopolitical_taiwan" in ev.path for ev in evidence_as_of_march)

    # Reference time is Q4 2024 (2024-09-01):
    # Now e3 is historical and should be reachable via 2 hops from NVDA
    evidence_as_of_sept = retriever.retrieve(
        query="NVDA risks",
        seed_node_ids=["company:NVDA"],
        reference_time="2024-09-01T00:00:00+00:00",
        max_hops=2,
    )
    assert any("risk:geopolitical_taiwan" in ev.path for ev in evidence_as_of_sept)


def test_provenance_preservation(sample_temporal_graph, mock_universe):
    retriever = GraphRetriever(graph=sample_temporal_graph, company_universe=mock_universe)
    evidence = retriever.retrieve(query="TSMC", max_hops=1)

    target_ev = next((ev for ev in evidence if ev.evidence_id == "edge:e1_tsm_supplies_nvda"), None)
    assert target_ev is not None
    assert target_ev.document_id == "doc_nvda_10k_2023"
    assert target_ev.chunk_id == "chunk_nvda_suppliers"
    assert target_ev.fiscal_period == "FY2023"
    assert target_ev.publication_date == "2024-02-01"
    assert target_ev.available_time == "2024-02-01T12:00:00+00:00"


def test_ranking_determinism(sample_temporal_graph, mock_universe):
    retriever = GraphRetriever(graph=sample_temporal_graph, company_universe=mock_universe)
    run1 = retriever.retrieve(query="NVDA suppliers and partners", max_hops=2, top_k=5)
    run2 = retriever.retrieve(query="NVDA suppliers and partners", max_hops=2, top_k=5)

    assert len(run1) == len(run2)
    assert [ev.evidence_id for ev in run1] == [ev.evidence_id for ev in run2]
    assert [ev.text for ev in run1] == [ev.text for ev in run2]


def test_relationship_intent_risks_prioritized(sample_temporal_graph, mock_universe):
    """'What risks affect NVIDIA?' must rank risk edges above unrelated edges like MANUFACTURES."""
    retriever = GraphRetriever(graph=sample_temporal_graph, company_universe=mock_universe)
    evidence = retriever.retrieve("What risks affect NVIDIA?", top_k=5)

    assert len(evidence) > 0
    # Top ranked evidence should be a risk relationship
    top_ev = evidence[0]
    assert top_ev.relationship in ("HAS_RISK", "AFFECTED_BY")
    assert top_ev.target_node == "risk:cyber"

    # MANUFACTURES edge should still be present, but ranked strictly below risk edges
    risk_indices = [i for i, ev in enumerate(evidence) if ev.relationship in ("HAS_RISK", "AFFECTED_BY")]
    mfg_indices = [i for i, ev in enumerate(evidence) if ev.relationship == "MANUFACTURES"]
    assert len(risk_indices) > 0
    if mfg_indices:
        assert min(risk_indices) < min(mfg_indices)


def test_relationship_intent_dependencies_prioritized(sample_temporal_graph, mock_universe):
    """'What companies does NVIDIA depend on?' must rank DEPENDS_ON / SUPPLIES above MANUFACTURES."""
    retriever = GraphRetriever(graph=sample_temporal_graph, company_universe=mock_universe)
    evidence = retriever.retrieve("What companies does NVIDIA depend on?", top_k=5)

    assert len(evidence) > 0
    top_ev = evidence[0]
    assert top_ev.relationship in ("DEPENDS_ON", "SUPPLIES")

    dep_indices = [i for i, ev in enumerate(evidence) if ev.relationship in ("DEPENDS_ON", "SUPPLIES")]
    mfg_indices = [i for i, ev in enumerate(evidence) if ev.relationship == "MANUFACTURES"]
    assert len(dep_indices) > 0
    if mfg_indices:
        assert min(dep_indices) < min(mfg_indices)


def test_relationship_intent_competitors_prioritized(sample_temporal_graph, mock_universe):
    """'Which companies compete with Apple?' must prioritize COMPETES_WITH."""
    retriever = GraphRetriever(graph=sample_temporal_graph, company_universe=mock_universe)
    evidence = retriever.retrieve("Which companies compete with Apple?", top_k=5)

    assert len(evidence) > 0
    top_ev = evidence[0]
    assert top_ev.relationship == "COMPETES_WITH"


def test_relationship_intent_neutral_broad_retrieval(sample_temporal_graph, mock_universe):
    """'How is NVIDIA connected to TSMC?' without relation keyword retains broad retrieval."""
    retriever = GraphRetriever(graph=sample_temporal_graph, company_universe=mock_universe)
    evidence = retriever.retrieve("How is NVIDIA connected to TSMC?", max_hops=2, top_k=10)

    assert len(evidence) > 0
    relationships = {ev.relationship for ev in evidence}
    # Multiple distinct relationship types should remain accessible
    assert len(relationships) >= 2


def test_mixed_historical_and_future_branches(sample_temporal_graph, mock_universe):
    """Historical branch remains accessible while future branch is pruned."""
    retriever = GraphRetriever(graph=sample_temporal_graph, company_universe=mock_universe)

    # Reference time: 2024-03-01
    # Historical: e1 (TSM->NVDA, 2024-02-01), e2 (ASML->TSM, 2024-02-15)
    # Future: e3 (TSM->risk:geopolitical_taiwan, 2024-08-01)
    results = retriever.retrieve(
        query="TSMC",
        reference_time="2024-03-01T00:00:00+00:00",
        max_hops=2,
        top_k=10,
    )

    # Historical edge and multi-hop paths should be present
    assert any("company:NVDA" in ev.path for ev in results)
    assert any("company:ASML" in ev.path for ev in results)

    # Future branch MUST NOT be present
    assert not any("risk:geopolitical_taiwan" in ev.path for ev in results)
    for ev in results:
        assert ev.available_time <= "2024-03-01T00:00:00+00:00"
