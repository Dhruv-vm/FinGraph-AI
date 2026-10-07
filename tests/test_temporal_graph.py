from datetime import datetime, timezone

from src.data.graph.schema import GraphEdge, GraphNode, TemporalGraph
from src.data.graph.snapshot import build_temporal_snapshot


def test_graph_node_preserves_temporal_metadata():
    node = GraphNode(
        node_id="company:AAPL",
        node_type="company",
        properties={"ticker": "AAPL"},
        event_time=None,
        available_time="2026-01-01T00:00:00+00:00",
    )

    data = node.to_dict()

    assert data["node_id"] == "company:AAPL"
    assert data["node_type"] == "company"
    assert data["properties"]["ticker"] == "AAPL"
    assert data["available_time"] == "2026-01-01T00:00:00+00:00"


def test_graph_edge_preserves_temporal_metadata():
    edge = GraphEdge(
        edge_id="company:AAPL->SUPPLIES->company:TSM",
        source="company:AAPL",
        target="supplier:TSM",
        relationship="SUPPLIES",
        event_time="2026-01-01T00:00:00+00:00",
        available_time="2026-01-02T00:00:00+00:00",
        properties={"confidence": 0.95},
    )

    data = edge.to_dict()

    assert data["relationship"] == "SUPPLIES"
    assert data["event_time"] == "2026-01-01T00:00:00+00:00"
    assert data["available_time"] == "2026-01-02T00:00:00+00:00"
    assert data["properties"]["confidence"] == 0.95


def test_temporal_snapshot_excludes_future_nodes():
    graph = TemporalGraph(
        nodes=[
            GraphNode(
                node_id="company:AAPL",
                node_type="company",
            ),
            GraphNode(
                node_id="doc:past",
                node_type="Document",
                available_time="2026-01-01T00:00:00+00:00",
            ),
            GraphNode(
                node_id="doc:future",
                node_type="Document",
                available_time="2026-08-01T00:00:00+00:00",
            ),
        ]
    )

    snapshot = build_temporal_snapshot(
        graph,
        "2026-07-01T00:00:00+00:00",
    )

    node_ids = {node.node_id for node in snapshot.nodes}

    assert "company:AAPL" in node_ids
    assert "doc:past" in node_ids
    assert "doc:future" not in node_ids


def test_temporal_snapshot_excludes_future_edges():
    graph = TemporalGraph(
        nodes=[
            GraphNode(
                node_id="company:AAPL",
                node_type="company",
            ),
            GraphNode(
                node_id="company:TSM",
                node_type="company",
            ),
            GraphNode(
                node_id="company:AMD",
                node_type="company",
            ),
        ],
        edges=[
            GraphEdge(
                edge_id="edge:past",
                source="company:AAPL",
                target="company:TSM",
                relationship="SUPPLIES",
                available_time="2026-01-01T00:00:00+00:00",
            ),
            GraphEdge(
                edge_id="edge:future",
                source="company:AAPL",
                target="company:AMD",
                relationship="SUPPLIES",
                available_time="2026-08-01T00:00:00+00:00",
            ),
        ],
    )

    snapshot = build_temporal_snapshot(
        graph,
        "2026-07-01T00:00:00+00:00",
    )

    edge_ids = {edge.edge_id for edge in snapshot.edges}

    assert "edge:past" in edge_ids
    assert "edge:future" not in edge_ids


def test_temporal_snapshot_includes_exact_boundary():
    graph = TemporalGraph(
        nodes=[
            GraphNode(
                node_id="company:AAPL",
                node_type="company",
            ),
            GraphNode(
                node_id="company:TSM",
                node_type="company",
                available_time="2026-07-01T00:00:00+00:00",
            ),
        ],
        edges=[
            GraphEdge(
                edge_id="edge:boundary",
                source="company:AAPL",
                target="company:TSM",
                relationship="SUPPLIES",
                available_time="2026-07-01T00:00:00+00:00",
            )
        ],
    )

    reference_time = datetime(
        2026,
        7,
        1,
        tzinfo=timezone.utc,
    )

    snapshot = build_temporal_snapshot(
        graph,
        reference_time,
    )

    node_ids = {node.node_id for node in snapshot.nodes}
    edge_ids = {edge.edge_id for edge in snapshot.edges}

    assert "company:TSM" in node_ids
    assert "edge:boundary" in edge_ids


def test_temporal_snapshot_rejects_unknown_availability():
    graph = TemporalGraph(
        nodes=[
            GraphNode(
                node_id="company:AAPL",
                node_type="company",
            ),
            GraphNode(
                node_id="doc:unknown",
                node_type="Document",
                available_time=None,
            ),
        ]
    )

    snapshot = build_temporal_snapshot(
        graph,
        "2026-07-01T00:00:00+00:00",
    )

    node_ids = {node.node_id for node in snapshot.nodes}

    assert "company:AAPL" in node_ids
    assert "doc:unknown" not in node_ids


def test_snapshot_requires_both_edge_endpoints_to_be_available():
    graph = TemporalGraph(
        nodes=[
            GraphNode(
                node_id="company:AAPL",
                node_type="company",
            ),
            GraphNode(
                node_id="supplier:TSM",
                node_type="Supplier",
                available_time="2026-08-01T00:00:00+00:00",
            ),
        ],
        edges=[
            GraphEdge(
                edge_id="edge:future-target",
                source="company:AAPL",
                target="supplier:TSM",
                relationship="SUPPLIES",
                available_time="2026-01-01T00:00:00+00:00",
            )
        ],
    )

    snapshot = build_temporal_snapshot(
        graph,
        "2026-07-01T00:00:00+00:00",
    )

    assert snapshot.edges == []


def test_entity_id_compatibility():
    """Test A: Extracted entities and relations must not lose edges due to ID mismatch."""
    from src.data.extraction.entities import build_entity
    from src.data.extraction.relations import build_relation
    from src.data.graph.builder import build_semantic_graph

    nvidia = build_entity("Company", "NVIDIA").to_dict()
    tsmc = build_entity("Supplier", "TSMC").to_dict()

    relation = build_relation(
        source_entity_id=nvidia["entity_id"],
        target_entity_id=tsmc["entity_id"],
        relationship="SUPPLIES",
        confidence=0.95,
    ).to_dict()

    graph = build_semantic_graph(
        entities=[nvidia, tsmc],
        relations=[relation],
    )

    assert graph.node_count == 2
    assert graph.edge_count == 1
    edge = graph.edges[0]
    assert edge.relationship == "SUPPLIES"
    assert edge.source in {nvidia["entity_id"], "company:NVDA"}
    assert edge.target == tsmc["entity_id"]


def test_entity_type_normalization():
    """Test B: Equivalent entity-type casings must normalize to canonical TitleCase."""
    from src.data.graph.builder import build_semantic_graph
    from src.data.graph.relationships import is_valid_node_type, normalize_entity_type

    assert normalize_entity_type("company") == "Company"
    assert normalize_entity_type("COMPANY") == "Company"
    assert normalize_entity_type("Company") == "Company"
    assert normalize_entity_type("financialmetric") == "FinancialMetric"
    assert normalize_entity_type("newsarticle") == "NewsArticle"
    assert is_valid_node_type("company")
    assert is_valid_node_type("COMPANY")
    assert is_valid_node_type("Supplier")

    entities = [
        {"entity_type": "company", "name": "OpenAI", "canonical_name": "OpenAI"},
        {"entity_type": "SUPPLIER", "name": "TSMC", "canonical_name": "TSMC"},
    ]
    graph = build_semantic_graph(entities=entities, relations=[])
    types = {node.node_type for node in graph.nodes}
    assert types == {"Company", "Supplier"}


def test_self_referential_relations_rejected():
    """Test C: Self-referential relations must be rejected/skipped."""
    from src.data.graph.builder import build_semantic_graph

    entities = [
        {"entity_type": "Company", "name": "NVIDIA", "canonical_name": "NVIDIA"}
    ]
    relations = [
        {
            "source": "NVIDIA",
            "target": "NVIDIA",
            "relationship": "PARTNERS_WITH",
        }
    ]
    graph = build_semantic_graph(entities=entities, relations=relations)
    assert graph.node_count == 1
    assert graph.edge_count == 0


def test_unknown_relationship_rejected():
    """Test D: Unknown relationship types must not enter the KG."""
    from src.data.graph.builder import build_semantic_graph

    entities = [
        {"entity_type": "Company", "name": "NVIDIA", "canonical_name": "NVIDIA"},
        {"entity_type": "Product", "name": "GPU", "canonical_name": "GPU"},
    ]
    relations = [
        {
            "source": "NVIDIA",
            "target": "GPU",
            "relationship": "UNKNOWN_CUSTOM_RELATION",
        }
    ]
    graph = build_semantic_graph(entities=entities, relations=relations)
    assert graph.node_count == 2
    assert graph.edge_count == 0


def test_dangling_relation_rejected():
    """Test E: Relations referencing missing entities must not create dangling edges."""
    from src.data.graph.builder import build_semantic_graph

    entities = [
        {"entity_type": "Company", "name": "NVIDIA", "canonical_name": "NVIDIA"}
    ]
    relations = [
        {
            "source": "NVIDIA",
            "target": "supplier:non_existent_id",
            "relationship": "SUPPLIES",
        }
    ]
    graph = build_semantic_graph(entities=entities, relations=relations)
    assert graph.node_count == 1
    assert graph.edge_count == 0


def test_temporal_metadata_preservation():
    """Test F: event_time and available_time must survive extraction -> KG construction."""
    from src.data.graph.builder import build_semantic_graph

    entities = [
        {
            "entity_type": "Company",
            "name": "NVIDIA",
            "canonical_name": "NVIDIA",
            "event_time": "2026-01-10T00:00:00+00:00",
            "available_time": "2026-01-15T12:00:00+00:00",
        },
        {
            "entity_type": "Supplier",
            "name": "TSMC",
            "canonical_name": "TSMC",
            "event_time": "2026-01-10T00:00:00+00:00",
            "available_time": "2026-01-15T12:00:00+00:00",
        },
    ]
    relations = [
        {
            "source": "NVIDIA",
            "target": "TSMC",
            "relationship": "SUPPLIES",
            "event_time": "2026-01-10T00:00:00+00:00",
            "available_time": "2026-01-15T12:00:00+00:00",
        }
    ]
    graph = build_semantic_graph(entities=entities, relations=relations)
    assert graph.edge_count == 1
    edge = graph.edges[0]
    assert edge.event_time == "2026-01-10T00:00:00+00:00"
    assert edge.available_time == "2026-01-15T12:00:00+00:00"


def test_provenance_preservation():
    """Test G: Resulting node/edge must remain traceable to source document/chunk."""
    from src.data.graph.builder import build_semantic_graph

    entities = [
        {
            "entity_type": "Company",
            "name": "NVIDIA",
            "canonical_name": "NVIDIA",
            "properties": {"document_id": "doc:sec:123", "chunk_id": "doc:sec:123:chunk:0"},
        },
        {
            "entity_type": "Supplier",
            "name": "TSMC",
            "canonical_name": "TSMC",
            "properties": {"document_id": "doc:sec:123", "chunk_id": "doc:sec:123:chunk:0"},
        },
    ]
    relations = [
        {
            "source": "NVIDIA",
            "target": "TSMC",
            "relationship": "SUPPLIES",
            "properties": {
                "document_id": "doc:sec:123",
                "chunk_id": "doc:sec:123:chunk:0",
                "source_url": "https://sec.gov/filing.htm",
            },
        }
    ]
    graph = build_semantic_graph(entities=entities, relations=relations)
    assert graph.edge_count == 1
    edge = graph.edges[0]
    assert edge.properties["document_id"] == "doc:sec:123"
    assert edge.properties["chunk_id"] == "doc:sec:123:chunk:0"
    assert edge.properties["source_url"] == "https://sec.gov/filing.htm"


def test_multi_document_assembly():
    """Test H: Shared entities across documents resolve consistently."""
    from src.data.graph.builder import build_semantic_graph

    # Doc 1: NVIDIA Corporation -> MANUFACTURES -> H100
    doc1_entities = [
        {"entity_type": "Company", "name": "NVIDIA Corporation", "canonical_name": "NVIDIA Corporation"},
        {"entity_type": "Product", "name": "H100", "canonical_name": "H100"},
    ]
    doc1_relations = [
        {"source": "NVIDIA Corporation", "target": "H100", "relationship": "MANUFACTURES"}
    ]

    # Doc 2: NVIDIA -> SUPPLIES -> Microsoft
    doc2_entities = [
        {"entity_type": "Company", "name": "NVIDIA", "canonical_name": "NVIDIA"},
        {"entity_type": "Company", "name": "Microsoft", "canonical_name": "Microsoft"},
    ]
    doc2_relations = [
        {"source": "NVIDIA", "target": "Microsoft", "relationship": "SUPPLIES"}
    ]

    graph = build_semantic_graph(
        entities=doc1_entities + doc2_entities,
        relations=doc1_relations + doc2_relations,
    )

    # NVIDIA Corporation and NVIDIA must resolve to the SAME node company:NVDA
    node_ids = {node.node_id for node in graph.nodes}
    assert "company:NVDA" in node_ids
    assert "company:MSFT" in node_ids
    assert graph.node_count == 3  # company:NVDA, company:MSFT, product:H100

    # 2 edges, both connected to company:NVDA
    assert graph.edge_count == 2
    nvda_edges = [e for e in graph.edges if e.source == "company:NVDA"]
    assert len(nvda_edges) == 2
    relationships = {e.relationship for e in nvda_edges}
    assert relationships == {"MANUFACTURES", "SUPPLIES"}

