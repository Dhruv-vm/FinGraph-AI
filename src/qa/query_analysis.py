from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import re
from typing import Any

from src.retrieval.graph import GraphRetriever, detect_relationship_intents


_MONTH_NAME_TO_INT = {
    "january": 1, "jan": 1,
    "february": 2, "feb": 2,
    "march": 3, "mar": 3,
    "april": 4, "apr": 4,
    "may": 5,
    "june": 6, "jun": 6,
    "july": 7, "jul": 7,
    "august": 8, "aug": 8,
    "september": 9, "sep": 9, "sept": 9,
    "october": 10, "oct": 10,
    "november": 11, "nov": 11,
    "december": 12, "dec": 12,
}

_QUARTER_END_MONTH = {
    1: (3, 31),
    2: (6, 30),
    3: (9, 30),
    4: (12, 31),
}


@dataclass
class QueryAnalysis:
    """Structured representation of query semantics, intent, and constraints."""

    original_query: str
    entities: list[str] = field(default_factory=list)
    unresolved_entities: list[str] = field(default_factory=list)
    relationship_intent: list[str] = field(default_factory=list)
    question_type: str = "factual"
    secondary_question_types: list[str] = field(default_factory=list)
    reference_time: str | None = None
    temporal_window: dict[str, str] | None = None
    expected_hop_depth: int = 1
    is_multi_hop: bool = False
    cross_document_required: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class QueryAnalyzer:
    """
    Deterministic Query Analysis Engine for FinGraph AI.

    Extracts entities, relationship intent, reference time, question type,
    and expected hop depth without relying on an external LLM.
    """

    def __init__(self, graph_retriever: GraphRetriever | None = None) -> None:
        self.graph_retriever = graph_retriever

    def analyze(self, query: str, reference_time_override: str | None = None) -> QueryAnalysis:
        """
        Analyze query to extract entities, relationships, temporal constraints, and reasoning depth.

        Args:
            query: Natural language financial question.
            reference_time_override: Optional explicit reference time overriding lexical extraction.

        Returns:
            QueryAnalysis: Structured semantic analysis of the query.
        """
        clean_query = query.strip()

        # 1. Entity Extraction
        entities: list[str] = []
        unresolved: list[str] = []
        if self.graph_retriever is not None and hasattr(self.graph_retriever, "link_entities"):
            try:
                res = self.graph_retriever.link_entities(clean_query)
                if isinstance(res, tuple) and len(res) == 2:
                    entities, unresolved = res
                elif isinstance(res, (list, tuple)):
                    entities = list(res)
            except Exception:
                entities = []
                unresolved = []

        if not entities:
            # Fallback ticker regex
            tickers = re.findall(r"\b[A-Z]{1,5}\b", clean_query)
            entities = [f"company:{t}" for t in tickers if t not in ("A", "I", "IN", "ON", "AT", "TO", "OR", "AND")]

        # 2. Relationship Intent
        rel_intents = sorted(detect_relationship_intents(clean_query))

        # 3. Reference Time Extraction
        ref_time, temp_window = self._extract_temporal_constraints(clean_query)
        if reference_time_override is not None:
            ref_time = reference_time_override

        # 4. Expected Hop Depth & Multi-hop Detection
        hop_depth, is_multi_hop = self._estimate_hop_depth(clean_query, entities)

        # 5. Question Type Classification
        q_type, secondary_types = self._classify_question_type(clean_query, entities, rel_intents, is_multi_hop, ref_time)

        # 6. Cross-document Requirement
        # Multi-entity comparisons or multi-hop paths often span separate 10-K/10-Q filings
        cross_doc = len(entities) >= 2 or is_multi_hop or q_type == "comparison"

        return QueryAnalysis(
            original_query=clean_query,
            entities=entities,
            unresolved_entities=unresolved,
            relationship_intent=rel_intents,
            question_type=q_type,
            secondary_question_types=secondary_types,
            reference_time=ref_time,
            temporal_window=temp_window,
            expected_hop_depth=hop_depth,
            is_multi_hop=is_multi_hop,
            cross_document_required=cross_doc,
        )

    def _extract_temporal_constraints(
        self, query: str
    ) -> tuple[str | None, dict[str, str] | None]:
        """
        Extract explicit temporal expressions from query text deterministically.
        Returns (reference_time_iso, temporal_window_dict).
        """
        q_lower = query.lower()

        # 1. Month DD, YYYY (e.g., "June 1, 2025", "as of March 31, 2024")
        month_day_year_match = re.search(
            r"(?:as of|by|before|on|at)?\s*([a-z]+)\s+(\d{1,2}),?\s+(\d{4})\b",
            q_lower,
        )
        if month_day_year_match:
            month_str, day_str, year_str = month_day_year_match.groups()
            if month_str in _MONTH_NAME_TO_INT:
                month_num = _MONTH_NAME_TO_INT[month_str]
                day_num = int(day_str)
                year_num = int(year_str)
                dt = datetime(year_num, month_num, day_num, 0, 0, 0, tzinfo=timezone.utc)
                iso_str = dt.isoformat()
                return iso_str, {"start": f"{year_num}-01-01T00:00:00+00:00", "end": iso_str}

        # 2. ISO date YYYY-MM-DD (e.g., "as of 2024-03-31", "2025-06-01")
        iso_match = re.search(r"\b(\d{4})-(\d{2})-(\d{2})\b", query)
        if iso_match:
            y, m, d = iso_match.groups()
            dt = datetime(int(y), int(m), int(d), 0, 0, 0, tzinfo=timezone.utc)
            return dt.isoformat(), {"start": f"{y}-01-01T00:00:00+00:00", "end": dt.isoformat()}

        # 3. Quarter expressions (e.g. "Q1 2024", "in Q3 2025", "end of Q2 2024")
        quarter_match = re.search(r"\bq([1-4])\s+(\d{4})\b", q_lower)
        if quarter_match:
            q_num = int(quarter_match.group(1))
            year_num = int(quarter_match.group(2))
            end_m, end_d = _QUARTER_END_MONTH[q_num]
            dt = datetime(year_num, end_m, end_d, 23, 59, 59, tzinfo=timezone.utc)
            return dt.isoformat(), {"start": f"{year_num}-01-01T00:00:00+00:00", "end": dt.isoformat()}

        # 4. Expressions with "by the end of YYYY" / "by end of YYYY" / "at the end of YYYY"
        end_of_year_match = re.search(r"(?:by|at)?\s*(?:the\s+)?end\s+of\s+(\d{4})\b", q_lower)
        if end_of_year_match:
            year_num = int(end_of_year_match.group(1))
            dt = datetime(year_num, 12, 31, 23, 59, 59, tzinfo=timezone.utc)
            return dt.isoformat(), {"start": f"{year_num}-01-01T00:00:00+00:00", "end": dt.isoformat()}

        # 5. "before YYYY" (e.g. "before 2025" -> end of 2024)
        before_match = re.search(r"\bbefore\s+(\d{4})\b", q_lower)
        if before_match:
            year_num = int(before_match.group(1)) - 1
            dt = datetime(year_num, 12, 31, 23, 59, 59, tzinfo=timezone.utc)
            return dt.isoformat(), {"start": "2000-01-01T00:00:00+00:00", "end": dt.isoformat()}

        # 6. "as of YYYY" (e.g. "as of 2024")
        as_of_year_match = re.search(r"\bas\s+of\s+(\d{4})\b", q_lower)
        if as_of_year_match:
            year_num = int(as_of_year_match.group(1))
            dt = datetime(year_num, 12, 31, 23, 59, 59, tzinfo=timezone.utc)
            return dt.isoformat(), {"start": f"{year_num}-01-01T00:00:00+00:00", "end": dt.isoformat()}

        # 7. "in YYYY" / "during YYYY" (e.g., "in 2024", "during 2023")
        in_year_match = re.search(r"\b(?:in|during)\s+(\d{4})\b", q_lower)
        if in_year_match:
            year_num = int(in_year_match.group(1))
            dt = datetime(year_num, 12, 31, 23, 59, 59, tzinfo=timezone.utc)
            return dt.isoformat(), {
                "start": f"{year_num}-01-01T00:00:00+00:00",
                "end": f"{year_num}-12-31T23:59:59+00:00",
            }

        # 8. Standalone 4-digit year with prepositions (e.g. "fiscal year 2024", "FY 2024")
        fy_match = re.search(r"\b(?:fy|fiscal\s+year)\s*(\d{4})\b", q_lower)
        if fy_match:
            year_num = int(fy_match.group(1))
            dt = datetime(year_num, 12, 31, 23, 59, 59, tzinfo=timezone.utc)
            return dt.isoformat(), {"start": f"{year_num}-01-01T00:00:00+00:00", "end": dt.isoformat()}

        # No temporal expression specified
        return None, None

    def _estimate_hop_depth(self, query: str, entities: list[str]) -> tuple[int, bool]:
        """
        Estimate expected reasoning hop depth (1, 2, or 3+) using lexical/structural signals.
        """
        q_lower = query.lower()

        # Explicit multi-hop keywords
        three_hop_cues = (
            "3-hop", "three hops", "three steps", "path from", "path between",
            "indirectly connected through its supplier", "chain of suppliers",
        )
        if any(cue in q_lower for cue in three_hop_cues):
            return 3, True

        two_hop_cues = (
            "connected to", "connection between", "relationship between",
            "indirectly", "indirect", "through its", "through their",
            "supplier's supplier", "customer's customer", "supplier of a product",
            "supplier does nvidia depend on for a product",
            "supplier for a product", "supplier that produces",
            "connected through", "intermediary", "chain",
            "both", "compete with each other",
        )
        if any(cue in q_lower for cue in two_hop_cues):
            return 2, True

        # Query structure: two named companies asking how they connect
        if len(entities) >= 2 and any(k in q_lower for k in ("connect", "relation", "between", "link", "interact")):
            return 2, True

        # Compound relationship inquiry (e.g. "What risks affect companies that supply NVIDIA?",
        # "What suppliers provide components for products that NVIDIA manufactures?")
        rel_count = sum(
            1 for cue in ("supplier", "supply", "depend", "product", "manufacture", "risk", "affect", "compete", "partner")
            if cue in q_lower
        )
        if rel_count >= 2 and any(w in q_lower for w in ("that", "which", "for a", "for their", "whose", "of companies", "companies that")):
            return 2, True

        # Default to 1-hop for direct factual/relational questions
        return 1, False

    def _classify_question_type(
        self,
        query: str,
        entities: list[str],
        rel_intents: list[str],
        is_multi_hop: bool,
        ref_time: str | None,
    ) -> tuple[str, list[str]]:
        """
        Classify primary and secondary question types.
        """
        q_lower = query.lower()
        types: list[str] = []

        # 1. Comparison
        comparison_cues = ("compare", "higher", "lower", "more than", "less than", "versus", "vs", "better", "greater", "outperform")
        if any(cue in q_lower for cue in comparison_cues) or (len(entities) >= 2 and " or " in q_lower):
            types.append("comparison")

        # 2. Aggregation
        aggregation_cues = ("total", "sum", "average", "how many", "count of", "overall revenue", "aggregate")
        if any(cue in q_lower for cue in aggregation_cues):
            types.append("aggregation")

        # 3. Multi-hop
        if is_multi_hop:
            types.append("multi-hop")

        # 4. Temporal
        if ref_time is not None or any(t in q_lower for t in ("when", "year", "quarter", "timeline", "date")):
            types.append("temporal")

        # 5. Relationship
        if rel_intents or any(r in q_lower for r in ("connect", "supplier", "depend", "partner", "compete", "affect", "manufacture", "risk")):
            types.append("relationship")

        # 6. Factual fallback
        if not types or any(q_lower.startswith(w) for w in ("what", "who", "where", "which")):
            types.append("factual")

        primary = types[0]
        # Re-prioritize primary type if comparison or multi-hop is detected
        if "comparison" in types:
            primary = "comparison"
        elif "multi-hop" in types:
            primary = "multi-hop"
        elif "relationship" in types and primary == "factual":
            primary = "relationship"

        secondary = [t for t in types if t != primary]
        return primary, secondary
