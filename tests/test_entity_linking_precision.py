import pytest
from src.data.graph.schema import GraphEdge, GraphNode, TemporalGraph
from src.retrieval.graph import GraphRetriever


@pytest.fixture
def precision_test_graph():
    """Graph with a mix of company, generic, numeric, and legitimate domain nodes."""
    nodes = {
        "company:NVDA": GraphNode(
            node_id="company:NVDA",
            node_type="Company",
            properties={"name": "NVIDIA Corporation", "canonical_name": "NVIDIA", "ticker": "NVDA"},
        ),
        "company:AAPL": GraphNode(
            node_id="company:AAPL",
            node_type="Company",
            properties={"name": "Apple Inc.", "canonical_name": "Apple", "ticker": "AAPL"},
        ),
        "company:MSFT": GraphNode(
            node_id="company:MSFT",
            node_type="Company",
            properties={"name": "Microsoft Corporation", "canonical_name": "Microsoft", "ticker": "MSFT"},
        ),
        "company:TSM": GraphNode(
            node_id="company:TSM",
            node_type="Company",
            properties={"name": "Taiwan Semiconductor Manufacturing Company", "canonical_name": "TSMC", "ticker": "TSM"},
        ),
        # Incidental generic nodes extracted from SEC text
        "company:generic_firm": GraphNode(
            node_id="company:generic_firm",
            node_type="Company",
            properties={"name": "Firm", "canonical_name": "Firm"},
        ),
        "risk:competition": GraphNode(
            node_id="risk:competition",
            node_type="Risk",
            properties={"name": "Competition", "canonical_name": "Competition"},
        ),
        "event:acquisition": GraphNode(
            node_id="event:acquisition",
            node_type="Event",
            properties={"name": "acquisition", "canonical_name": "acquisition"},
        ),
        "financialmetric:one_dollar": GraphNode(
            node_id="financialmetric:one_dollar",
            node_type="FinancialMetric",
            properties={"name": "$1", "canonical_name": "$1"},
        ),
        "event:year_2025": GraphNode(
            node_id="event:year_2025",
            node_type="Event",
            properties={"name": "2025", "canonical_name": "2025"},
        ),
        # Legitimate domain non-company entities
        "product:fsd": GraphNode(
            node_id="product:fsd",
            node_type="Product",
            properties={"name": "Full Self-Driving", "canonical_name": "Full Self-Driving (FSD)"},
        ),
        "product:gemini": GraphNode(
            node_id="product:gemini",
            node_type="Product",
            properties={"name": "Gemini app", "canonical_name": "Gemini"},
        ),
        "risk:export_restrictions": GraphNode(
            node_id="risk:export_restrictions",
            node_type="Risk",
            properties={"name": "Export Restrictions", "canonical_name": "Export Restrictions"},
        ),
    }

    edges = [
        GraphEdge(
            edge_id="e1",
            source="company:TSM",
            target="company:NVDA",
            relationship="SUPPLIES",
            available_time="2026-01-01T00:00:00+00:00",
        ),
    ]
    return TemporalGraph(nodes=list(nodes.values()), edges=edges)


def test_rejects_generic_words_firm_and_competition(precision_test_graph):
    retriever = GraphRetriever(graph=precision_test_graph)
    query = "What cybersecurity firm did Apple face competition from?"
    resolved, _ = retriever.link_entities(query)

    assert "company:AAPL" in resolved
    assert "company:generic_firm" not in resolved
    assert "risk:competition" not in resolved


def test_rejects_standalone_numbers_and_date_tokens(precision_test_graph):
    retriever = GraphRetriever(graph=precision_test_graph)
    query = "What risks affected Apple as of August 1, 2025?"
    resolved, _ = retriever.link_entities(query)

    assert "company:AAPL" in resolved
    assert "financialmetric:one_dollar" not in resolved
    assert "event:year_2025" not in resolved


def test_links_known_companies_and_aliases(precision_test_graph):
    retriever = GraphRetriever(graph=precision_test_graph)
    query = "Does Taiwan Semiconductor supply Nvidia and Microsoft?"
    resolved, _ = retriever.link_entities(query)

    assert "company:TSM" in resolved
    assert "company:NVDA" in resolved
    assert "company:MSFT" in resolved


def test_preserves_legitimate_domain_entities(precision_test_graph):
    retriever = GraphRetriever(graph=precision_test_graph)
    query = "Compare Full Self-Driving and Gemini app disclosures."
    resolved, _ = retriever.link_entities(query)

    assert "product:fsd" in resolved
    assert "product:gemini" in resolved


def test_queries_with_multiple_entity_types(precision_test_graph):
    retriever = GraphRetriever(graph=precision_test_graph)
    query = "Does TSMC face Export Restrictions when supplying Nvidia?"
    resolved, _ = retriever.link_entities(query)

    assert "company:TSM" in resolved
    assert "company:NVDA" in resolved
    assert "risk:export_restrictions" in resolved


def test_short_valid_ticker_preserved(precision_test_graph):
    retriever = GraphRetriever(graph=precision_test_graph)
    query = "Analyze NVDA and TSM financial disclosures."
    resolved, _ = retriever.link_entities(query)

    assert "company:NVDA" in resolved
    assert "company:TSM" in resolved
