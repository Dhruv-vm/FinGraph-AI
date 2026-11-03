import pytest
from src.evaluation.retrieval import RetrievalEvaluator, _candidate_path_matches, _edge_matches


def test_valid_directed_multi_hop_path():
    """A valid 2-hop directed chain A -> B -> C matches the expected path."""
    exp_path = [
        ["company:NVDA", "DEPENDS_ON", "USG"],
        ["USG", "HAS_RISK", "Export Restrictions"],
    ]
    ret_path = [
        {"source": "company:NVDA", "relationship": "DEPENDS_ON", "target": "USG"},
        {"source": "USG", "relationship": "HAS_RISK", "target": "Export Restrictions"},
    ]

    assert _candidate_path_matches(ret_path, exp_path) is True


def test_reversed_edge_rejected():
    """A reversed edge where source and target are swapped must be rejected."""
    exp_path = [
        ["company:NVDA", "DEPENDS_ON", "USG"],
        ["USG", "HAS_RISK", "Export Restrictions"],
    ]
    # Edge 1 is reversed (USG -> NVDA instead of NVDA -> USG)
    reversed_ret_path = [
        {"source": "USG", "relationship": "DEPENDS_ON", "target": "company:NVDA"},
        {"source": "USG", "relationship": "HAS_RISK", "target": "Export Restrictions"},
    ]

    assert _candidate_path_matches(reversed_ret_path, exp_path) is False


def test_missing_relationship_rejected():
    """If the relationship predicate does not match, the path must be rejected."""
    exp_path = [
        ["company:NVDA", "DEPENDS_ON", "USG"],
        ["USG", "HAS_RISK", "Export Restrictions"],
    ]
    missing_rel_path = [
        {"source": "company:NVDA", "relationship": "COMPETES_WITH", "target": "USG"},
        {"source": "USG", "relationship": "HAS_RISK", "target": "Export Restrictions"},
    ]

    assert _candidate_path_matches(missing_rel_path, exp_path) is False


def test_disconnected_path_rejected():
    """If hop 1 ends at entity X and hop 2 starts at unrelated entity Y, path is rejected."""
    exp_path = [
        ["company:NVDA", "DEPENDS_ON", "USG"],
        ["USG", "HAS_RISK", "Export Restrictions"],
    ]
    disconnected_ret_path = [
        {"source": "company:NVDA", "relationship": "DEPENDS_ON", "target": "TSMC"},
        {"source": "USG", "relationship": "HAS_RISK", "target": "Export Restrictions"},
    ]

    assert _candidate_path_matches(disconnected_ret_path, exp_path) is False


def test_empty_or_unavailable_path_rejected():
    """An empty candidate path or None path returns False."""
    exp_path = [
        ["company:NVDA", "DEPENDS_ON", "USG"],
    ]

    assert _candidate_path_matches([], exp_path) is False
    assert _candidate_path_matches(None, exp_path) is False


def test_multiple_candidate_paths_one_valid():
    """When multiple candidate paths are retrieved, having one valid path succeeds."""
    evaluator = RetrievalEvaluator()
    exp_path = [
        ["company:NVDA", "DEPENDS_ON", "USG"],
        ["USG", "HAS_RISK", "Export Restrictions"],
    ]
    candidate_paths = [
        # Irrelevant path
        [{"source": "company:AAPL", "relationship": "COMPETES_WITH", "target": "company:GOOGL"}],
        # Disconnected path
        [
            {"source": "company:NVDA", "relationship": "DEPENDS_ON", "target": "TSMC"},
            {"source": "USG", "relationship": "HAS_RISK", "target": "Export Restrictions"},
        ],
        # Valid connected path
        [
            {"source": "company:NVDA", "relationship": "DEPENDS_ON", "target": "USG"},
            {"source": "USG", "relationship": "HAS_RISK", "target": "Export Restrictions"},
        ],
    ]

    res = evaluator.evaluate_single(
        question_id="TEST-01",
        retrieved_chunk_ids=[],
        relevant_chunk_ids=[],
        retrieved_doc_ids=[],
        relevant_doc_ids=[],
        retrieved_graph_paths=candidate_paths,
        expected_graph_paths=exp_path,
        is_answerable=True,
    )

    assert res.multi_hop_path_accuracy == 1.0


def test_multiple_candidate_paths_all_invalid():
    """When all candidate paths fail directionality or connectivity, accuracy is 0.0."""
    evaluator = RetrievalEvaluator()
    exp_path = [
        ["company:NVDA", "DEPENDS_ON", "USG"],
        ["USG", "HAS_RISK", "Export Restrictions"],
    ]
    candidate_paths = [
        [{"source": "USG", "relationship": "DEPENDS_ON", "target": "company:NVDA"}],
        [{"source": "company:AAPL", "relationship": "COMPETES_WITH", "target": "company:GOOGL"}],
    ]

    res = evaluator.evaluate_single(
        question_id="TEST-02",
        retrieved_chunk_ids=[],
        relevant_chunk_ids=[],
        retrieved_doc_ids=[],
        relevant_doc_ids=[],
        retrieved_graph_paths=candidate_paths,
        expected_graph_paths=exp_path,
        is_answerable=True,
    )

    assert res.multi_hop_path_accuracy == 0.0
