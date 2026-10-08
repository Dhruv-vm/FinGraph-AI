from __future__ import annotations

import pytest

from src.qa.query_analysis import QueryAnalyzer


@pytest.fixture
def analyzer() -> QueryAnalyzer:
    return QueryAnalyzer()


def test_explicit_reference_time_month_date_year(analyzer):
    analysis = analyzer.analyze("What risks affected NVIDIA as of June 1, 2025?")
    assert analysis.reference_time == "2025-06-01T00:00:00+00:00"
    assert analysis.temporal_window is not None
    assert analysis.temporal_window["end"] == "2025-06-01T00:00:00+00:00"


def test_explicit_reference_time_iso_date(analyzer):
    analysis = analyzer.analyze("What companies did Apple supply before 2024-03-31?")
    assert analysis.reference_time == "2024-03-31T00:00:00+00:00"


def test_explicit_reference_time_year_phrase(analyzer):
    analysis = analyzer.analyze("What were Microsoft's primary risks in 2024?")
    assert analysis.reference_time == "2024-12-31T23:59:59+00:00"
    assert analysis.temporal_window["start"] == "2024-01-01T00:00:00+00:00"


def test_no_reference_time_remains_none(analyzer):
    analysis = analyzer.analyze("What companies does NVIDIA depend on?")
    assert analysis.reference_time is None
    assert analysis.temporal_window is None


def test_question_type_classification(analyzer):
    # Factual / Relationship
    q1 = analyzer.analyze("What products does NVIDIA manufacture?")
    assert q1.question_type in ("factual", "relationship")

    # Comparison
    q2 = analyzer.analyze("Which company had higher revenue, Apple or Microsoft?")
    assert q2.question_type == "comparison"

    # Temporal
    q3 = analyzer.analyze("What risks affected NVIDIA in 2024?")
    assert "temporal" in [q3.question_type] + q3.secondary_question_types

    # Multi-hop
    q4 = analyzer.analyze("How is NVIDIA connected to TSMC through its suppliers?")
    assert q4.question_type == "multi-hop" or "multi-hop" in q4.secondary_question_types


def test_expected_hop_depth_one_hop(analyzer):
    analysis = analyzer.analyze("What company supplies NVIDIA?")
    assert analysis.expected_hop_depth == 1
    assert analysis.is_multi_hop is False


def test_expected_hop_depth_two_hop(analyzer):
    analysis = analyzer.analyze("What supplier does NVIDIA depend on for a product?")
    assert analysis.expected_hop_depth == 2
    assert analysis.is_multi_hop is True


def test_multi_hop_query_detected(analyzer):
    analysis = analyzer.analyze("Which company is connected to NVIDIA through its supplier?")
    assert analysis.expected_hop_depth >= 2
    assert analysis.is_multi_hop is True


def test_q6_supplier_risks_classified_as_multi_hop(analyzer):
    analysis = analyzer.analyze("What risks affect companies that supply NVIDIA?")
    assert analysis.expected_hop_depth >= 2
    assert analysis.is_multi_hop is True
    assert analysis.question_type == "multi-hop"
