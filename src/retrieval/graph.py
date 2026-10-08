from __future__ import annotations

from collections import deque
from datetime import datetime, timezone
from pathlib import Path
import re
from typing import Any

from src.data.entities.resolution import (
    DEFAULT_COMPANY_UNIVERSE,
    build_entity_indexes,
    load_company_universe,
    normalize_text,
    resolve_company,
    strip_company_legal_suffix,
)
from src.data.graph.relationships import normalize_entity_type
from src.data.graph.schema import GraphEdge, GraphNode, TemporalGraph
from src.data.graph.snapshot import is_available, parse_datetime
from src.data.graph.storage import load_graph
from src.retrieval.evidence import GraphEvidence


_QUERY_STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "did",
    "do",
    "does",
    "for",
    "from",
    "had",
    "has",
    "have",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
}


_ADDITIONAL_COMPANY_ALIASES: dict[str, str] = {
    "tsmc": "TSM",
    "taiwan semiconductor": "TSM",
    "taiwan semi": "TSM",
    "nvidia": "NVDA",
    "apple": "AAPL",
    "microsoft": "MSFT",
    "amazon": "AMZN",
    "google": "GOOGL",
    "meta": "META",
    "intel": "INTC",
    "asml": "ASML",
    "amd": "AMD",
    "broadcom": "AVGO",
    "qualcomm": "QCOM",
}


RELATIONSHIP_INTENTS: list[tuple[tuple[str, ...], set[str]]] = [
    (
        (
            "depend on",
            "depends on",
            "dependency",
            "dependencies",
            "dependent on",
            "reliant on",
            "relies on",
        ),
        {"DEPENDS_ON", "SUPPLIES"},
    ),
    (
        (
            "supplier",
            "suppliers",
            "supply",
            "supplies",
            "supply chain",
            "supplied by",
            "vendor",
            "vendors",
        ),
        {"SUPPLIES", "DEPENDS_ON"},
    ),
    (
        (
            "compete with",
            "competes with",
            "competitor",
            "competitors",
            "competition",
            "compete",
            "rival",
            "rivals",
        ),
        {"COMPETES_WITH"},
    ),
    (
        (
            "risk",
            "risks",
            "threat",
            "threats",
            "danger",
            "dangers",
            "vulnerability",
            "vulnerabilities",
            "hazard",
            "hazards",
        ),
        {"HAS_RISK", "AFFECTED_BY"},
    ),
    (
        (
            "partner",
            "partners",
            "partnership",
            "partnerships",
            "alliance",
            "alliances",
            "collaboration",
            "collaborate",
        ),
        {"PARTNERS_WITH"},
    ),
    (
        (
            "acquired",
            "acquisition",
            "acquisitions",
            "acquire",
            "merger",
            "merged with",
            "buyout",
        ),
        {"ACQUIRED"},
    ),
    (
        (
            "invested",
            "invested in",
            "investment",
            "investments",
            "investor",
            "investors",
            "stake in",
        ),
        {"INVESTED_IN"},
    ),
    (
        (
            "manufactures",
            "manufacture",
            "manufactured by",
            "product",
            "products",
            "produces",
            "produce",
            "makes",
        ),
        {"MANUFACTURES"},
    ),
    (
        (
            "located in",
            "headquarters",
            "headquartered in",
            "facilities in",
            "based in",
        ),
        {"LOCATED_IN"},
    ),
]


def detect_relationship_intents(query: str) -> set[str]:
    """Detect FinGraph relationship intents from natural language query."""
    if not query:
        return set()

    lower_query = query.lower()
    intents: set[str] = set()

    for phrases, rels in RELATIONSHIP_INTENTS:
        for phrase in phrases:
            if phrase in lower_query:
                intents.update(rels)
                break

    return intents


class GraphRetriever:
    """
    Temporal Graph Retrieval engine for FinGraph AI.

    Performs query entity linking, multi-hop BFS path traversal,
    cycle prevention, and strict point-in-time temporal filtering.
    """

    def __init__(
        self,
        graph: TemporalGraph | dict[str, Any] | str | Path,
        company_universe: str | Path | list[dict[str, str]] | None = DEFAULT_COMPANY_UNIVERSE,
    ) -> None:
        if isinstance(graph, (str, Path)):
            self.graph: TemporalGraph = load_graph(graph)
        elif isinstance(graph, dict):
            self.graph = TemporalGraph.from_dict(graph)
        elif isinstance(graph, TemporalGraph):
            self.graph = graph
        else:
            raise TypeError(f"Unsupported graph input type: {type(graph)}")

        # Initialize company universe resolution indexes
        self.universe_records: list[dict[str, str]] = []
        self.universe_indexes: dict[str, dict[str, dict[str, str]]] = {
            "ticker": {},
            "cik": {},
            "name": {},
        }
        if company_universe is not None:
            if isinstance(company_universe, list):
                self.universe_records = company_universe
                self.universe_indexes = build_entity_indexes(self.universe_records)
            elif isinstance(company_universe, (str, Path)):
                universe_path = Path(company_universe)
                if universe_path.exists():
                    self.universe_records = load_company_universe(universe_path)
                    self.universe_indexes = build_entity_indexes(self.universe_records)

        # Build in-memory graph lookup structures
        self.nodes: dict[str, GraphNode] = {}
        self.out_edges: dict[str, list[GraphEdge]] = {}
        self.in_edges: dict[str, list[GraphEdge]] = {}
        self.node_text_lookup: dict[str, set[str]] = {}

        self._build_graph_indexes()

    def _build_graph_indexes(self) -> None:
        """Index nodes, incident edges, and textual lookup keys."""
        for node in self.graph.nodes:
            self.nodes[node.node_id] = node
            self.out_edges.setdefault(node.node_id, [])
            self.in_edges.setdefault(node.node_id, [])

            # Index by normalized node_id
            self._register_node_lookup(node.node_id, node.node_id)
            if ":" in node.node_id:
                _, raw_id = node.node_id.split(":", 1)
                self._register_node_lookup(raw_id, node.node_id)

            # Index canonical name, name, and ticker
            for field_name in ("canonical_name", "name", "ticker"):
                val = node.properties.get(field_name)
                if val:
                    self._register_node_lookup(str(val), node.node_id)
                    stripped = strip_company_legal_suffix(str(val))
                    if stripped:
                        self._register_node_lookup(stripped, node.node_id)

            # Index explicit aliases if present in node properties
            aliases = node.properties.get("aliases")
            if isinstance(aliases, (list, tuple, set)):
                for alias in aliases:
                    if alias:
                        self._register_node_lookup(str(alias), node.node_id)

        # Register additional well-known corporate financial aliases
        for alias, ticker in _ADDITIONAL_COMPANY_ALIASES.items():
            candidate_id = f"company:{ticker}"
            if candidate_id in self.nodes:
                self._register_node_lookup(alias, candidate_id)

        for edge in self.graph.edges:
            self.out_edges.setdefault(edge.source, []).append(edge)
            self.in_edges.setdefault(edge.target, []).append(edge)

    def _register_node_lookup(self, text: str, node_id: str) -> None:
        """Register a normalized text key mapping to a node ID."""
        normalized = normalize_text(text)
        if normalized:
            self.node_text_lookup.setdefault(normalized, set()).add(node_id)

    def link_entities(self, query: str) -> tuple[list[str], list[str]]:
        """
        Link entities mentioned in a natural language query to KG nodes.

        Returns:
            tuple[list[str], list[str]]: (resolved_node_ids, unresolved_mentions)
        """
        resolved: set[str] = set()
        unresolved: set[str] = set()

        if not query or not query.strip():
            return [], []

        # 1. Match uppercase ticker symbols
        candidate_tickers = re.findall(r"\b[A-Z]{1,5}\b", query)
        for ticker in candidate_tickers:
            if ticker in self.universe_indexes["ticker"]:
                resolved.add(f"company:{ticker}")
            elif f"company:{ticker}" in self.nodes:
                resolved.add(f"company:{ticker}")
            elif normalize_text(ticker) in self.node_text_lookup:
                for node_id in self.node_text_lookup[normalize_text(ticker)]:
                    if node_id in self.nodes:
                        resolved.add(node_id)

        # 2. Extract potential entity phrases via sliding n-grams (5 down to 1)
        tokens = query.split()
        max_n = min(5, len(tokens))

        for n in range(max_n, 0, -1):
            for i in range(len(tokens) - n + 1):
                phrase = " ".join(tokens[i : i + n])
                norm_phrase = normalize_text(phrase)
                if not norm_phrase or norm_phrase in _QUERY_STOP_WORDS:
                    continue

                # Check additional known aliases
                if norm_phrase in _ADDITIONAL_COMPANY_ALIASES:
                    ticker = _ADDITIONAL_COMPANY_ALIASES[norm_phrase]
                    candidate_node_id = f"company:{ticker}"
                    if candidate_node_id in self.nodes:
                        resolved.add(candidate_node_id)
                        continue

                # Check universe company resolution
                if self.universe_indexes["name"]:
                    resolved_company = resolve_company(
                        company=phrase,
                        indexes=self.universe_indexes,
                    )
                    if resolved_company and resolved_company["entity_id"] in self.nodes:
                        resolved.add(resolved_company["entity_id"])
                        continue

                # Check KG node text lookup
                if norm_phrase in self.node_text_lookup:
                    for node_id in self.node_text_lookup[norm_phrase]:
                        if node_id in self.nodes:
                            resolved.add(node_id)
                    continue

                stripped = strip_company_legal_suffix(phrase)
                if stripped and stripped in self.node_text_lookup:
                    for node_id in self.node_text_lookup[stripped]:
                        if node_id in self.nodes:
                            resolved.add(node_id)
                    continue

                # Multi-word prefix matching
                if n >= 2:
                    for key, node_ids in self.node_text_lookup.items():
                        if key.startswith(norm_phrase + " "):
                            for node_id in node_ids:
                                if node_id in self.nodes:
                                    resolved.add(node_id)

        # 3. Detect unresolved mentions: Title-cased or capitalized terms not matched
        capitalized_spans = re.findall(r"\b[A-Z][a-zA-Z0-9_\-\.]*\b", query)
        for span in capitalized_spans:
            if span.lower() in _QUERY_STOP_WORDS:
                continue
            norm_span = normalize_text(span)
            # Check if this span contributed to any resolved node
            span_resolved = False
            for r_node_id in resolved:
                node = self.nodes.get(r_node_id)
                if not node:
                    continue
                node_name = normalize_text(
                    node.properties.get("canonical_name") or node.properties.get("name") or ""
                )
                if norm_span in node_name or norm_span in r_node_id.lower():
                    span_resolved = True
                    break
            if not span_resolved and norm_span not in self.node_text_lookup:
                unresolved.add(span)

        return sorted(resolved), sorted(unresolved)

    def _get_node_name(self, node_id: str) -> str:
        """Return the best human-readable name for a node."""
        node = self.nodes.get(node_id)
        if not node:
            return node_id
        return (
            node.properties.get("canonical_name")
            or node.properties.get("name")
            or node_id
        )

    def _verbalize_edge(self, edge: GraphEdge) -> str:
        """Generate a concise textual statement for a single edge."""
        src_name = self._get_node_name(edge.source)
        tgt_name = self._get_node_name(edge.target)
        rel = edge.relationship

        parts = [f"{src_name} --[{rel}]--> {tgt_name}"]
        props = edge.properties
        extra: list[str] = []
        if props.get("fiscal_period"):
            extra.append(f"period: {props['fiscal_period']}")
        if edge.available_time:
            extra.append(f"as of {edge.available_time[:10]}")
        if extra:
            parts.append(f"({', '.join(extra)})")

        return " ".join(parts)

    def _verbalize_path(self, nodes: list[str], edges: list[GraphEdge]) -> str:
        """Generate a concise textual statement for a multi-hop path."""
        if len(edges) == 1:
            return self._verbalize_edge(edges[0])

        chain = []
        for i, edge in enumerate(edges):
            u_name = self._get_node_name(edge.source)
            v_name = self._get_node_name(edge.target)
            chain.append(f"{u_name} --[{edge.relationship}]--> {v_name}")
        return " -> ".join(chain)

    def retrieve(
        self,
        query: str,
        reference_time: str | datetime | None = None,
        max_hops: int = 2,
        top_k: int = 10,
        seed_node_ids: list[str] | None = None,
    ) -> list[GraphEvidence]:
        """
        Retrieve relational evidence and multi-hop paths from the KG.

        Args:
            query: Query string for entity linking.
            reference_time: Temporal cutoff datetime or ISO string.
            max_hops: Maximum path length (1 to 3).
            top_k: Number of evidence items to return.
            seed_node_ids: Optional explicit seed node IDs.

        Returns:
            list[GraphEvidence]: Point-in-time filtered, ranked graph evidence.
        """
        if max_hops < 1:
            max_hops = 1
        elif max_hops > 3:
            max_hops = 3

        # Parse reference datetime
        ref_dt: datetime | None = None
        if reference_time is not None:
            if isinstance(reference_time, str):
                ref_dt = parse_datetime(reference_time)
                if ref_dt is None:
                    raise ValueError(f"Invalid reference_time format: {reference_time}")
            elif isinstance(reference_time, datetime):
                ref_dt = (
                    reference_time
                    if reference_time.tzinfo is not None
                    else reference_time.replace(tzinfo=timezone.utc)
                )

        # Detect relationship intent and target entity constraints
        intents = detect_relationship_intents(query)
        wants_company = any(
            w in query.lower()
            for w in ("company", "companies", "firm", "firms", "corporation")
        )

        # Determine seed nodes
        if seed_node_ids is None:
            seed_nodes, _ = self.link_entities(query)
        else:
            seed_nodes = list(seed_node_ids)

        active_seeds = [
            s
            for s in seed_nodes
            if s in self.nodes
            and (
                ref_dt is None
                or not self.nodes[s].available_time
                or is_available(self.nodes[s].available_time, ref_dt)
            )
        ]
        if not active_seeds:
            return []

        candidates: dict[str, GraphEvidence] = {}

        # Perform BFS traversal from seeds
        for seed in active_seeds:
            # Queue elements: (current_node, path_nodes, path_edges)
            queue: deque[tuple[str, list[str], list[GraphEdge]]] = deque()
            queue.append((seed, [seed], []))

            while queue:
                curr_node, path_nodes, path_edges = queue.popleft()
                curr_hop = len(path_edges)

                if curr_hop >= max_hops:
                    continue

                # Explore incident edges (outgoing and incoming)
                out_edges = self.out_edges.get(curr_node, [])
                in_edges = self.in_edges.get(curr_node, [])

                # Outgoing transitions: curr_node -> next_node
                for edge in out_edges:
                    next_node = edge.target
                    if next_node in path_nodes:
                        continue  # cycle prevention

                    # Strict point-in-time check on this edge
                    if ref_dt is not None and not is_available(edge.available_time, ref_dt):
                        continue

                    # Check target node temporal validity
                    target_node_obj = self.nodes.get(next_node)
                    if (
                        ref_dt is not None
                        and target_node_obj
                        and target_node_obj.available_time
                        and not is_available(target_node_obj.available_time, ref_dt)
                    ):
                        continue

                    new_edges = path_edges + [edge]
                    new_nodes = path_nodes + [next_node]
                    evidence = self._create_evidence(new_nodes, new_edges)

                    if ref_dt is not None and not is_available(evidence.available_time, ref_dt):
                        continue

                    if evidence.evidence_id not in candidates:
                        candidates[evidence.evidence_id] = evidence

                    if len(new_edges) < max_hops:
                        queue.append((next_node, new_nodes, new_edges))

                # Incoming transitions: next_node -> curr_node
                for edge in in_edges:
                    next_node = edge.source
                    if next_node in path_nodes:
                        continue  # cycle prevention

                    # Strict point-in-time check on this edge
                    if ref_dt is not None and not is_available(edge.available_time, ref_dt):
                        continue

                    # Check source node temporal validity
                    source_node_obj = self.nodes.get(next_node)
                    if (
                        ref_dt is not None
                        and source_node_obj
                        and source_node_obj.available_time
                        and not is_available(source_node_obj.available_time, ref_dt)
                    ):
                        continue

                    new_edges = path_edges + [edge]
                    new_nodes = path_nodes + [next_node]
                    evidence = self._create_evidence(new_nodes, new_edges)

                    if ref_dt is not None and not is_available(evidence.available_time, ref_dt):
                        continue

                    if evidence.evidence_id not in candidates:
                        candidates[evidence.evidence_id] = evidence

                    if len(new_edges) < max_hops:
                        queue.append((next_node, new_nodes, new_edges))

        # Score and rank candidates deterministically with relationship-intent weighting
        def score_evidence(item: GraphEvidence) -> float:
            hop_discount = 1.0 / (1.0 + 0.5 * (item.hop_count - 1))
            base_score = float(item.confidence) * hop_discount

            if not intents:
                return base_score

            path_rels = item.relationship.split(" -> ")
            matched_rels = [r for r in path_rels if r in intents]

            if matched_rels:
                multiplier = 5.0
                if len(matched_rels) == len(path_rels):
                    multiplier = 8.0
            else:
                multiplier = 0.2

            if wants_company:
                target_node = self.nodes.get(item.target_node)
                if target_node and target_node.node_type.casefold() == "company":
                    multiplier *= 1.5

            return base_score * multiplier

        scored = [
            (score_evidence(item), item)
            for item in candidates.values()
            if ref_dt is None or is_available(item.available_time, ref_dt)
        ]

        # Sort: score DESC, hop_count ASC, available_time DESC, evidence_id ASC
        scored.sort(
            key=lambda pair: (
                -pair[0],
                pair[1].hop_count,
                -(len(pair[1].available_time or "")),
                pair[1].evidence_id,
            )
        )

        return [item for _, item in scored[:top_k]]

    def _create_evidence(
        self,
        nodes: list[str],
        edges: list[GraphEdge],
    ) -> GraphEvidence:
        """Construct a GraphEvidence dataclass from traversed nodes and edges."""
        hop_count = len(edges)
        if hop_count == 1:
            edge = edges[0]
            evidence_id = f"edge:{edge.edge_id}"
            relationship = edge.relationship
            avail_time = edge.available_time
            event_time = edge.event_time
            doc_id = edge.properties.get("document_id")
            chunk_id = edge.properties.get("chunk_id")
            source = edge.properties.get("source")
            source_url = edge.properties.get("source_url")
            source_ref = edge.properties.get("source_reference")
            pub_date = edge.properties.get("publication_date")
            fiscal_period = edge.properties.get("fiscal_period")
            confidence = float(edge.properties.get("confidence", 1.0))
            props = dict(edge.properties)
        else:
            evidence_id = f"path:{':'.join(e.edge_id for e in edges)}"
            relationship = " -> ".join(e.relationship for e in edges)
            # Latest available_time in the chain
            avail_times = [e.available_time for e in edges if e.available_time]
            avail_time = max(avail_times) if avail_times else None
            event_times = [e.event_time for e in edges if e.event_time]
            event_time = max(event_times) if event_times else None
            last_edge = edges[-1]
            doc_id = last_edge.properties.get("document_id")
            chunk_id = last_edge.properties.get("chunk_id")
            source = last_edge.properties.get("source")
            source_url = last_edge.properties.get("source_url")
            source_ref = last_edge.properties.get("source_reference")
            pub_date = last_edge.properties.get("publication_date")
            fiscal_period = last_edge.properties.get("fiscal_period")
            # Combined confidence as product of edge confidences
            conf = 1.0
            for e in edges:
                conf *= float(e.properties.get("confidence", 1.0))
            confidence = conf
            props = {
                "edge_ids": [e.edge_id for e in edges],
                "documents": [e.properties.get("document_id") for e in edges if e.properties.get("document_id")],
                "chunks": [e.properties.get("chunk_id") for e in edges if e.properties.get("chunk_id")],
            }

        text = self._verbalize_path(nodes, edges)

        return GraphEvidence(
            evidence_id=evidence_id,
            source_node=nodes[0],
            target_node=nodes[-1],
            relationship=relationship,
            path=nodes,
            hop_count=hop_count,
            text=text,
            available_time=avail_time,
            event_time=event_time,
            confidence=confidence,
            document_id=doc_id,
            chunk_id=chunk_id,
            source=source,
            source_url=source_url,
            source_reference=source_ref,
            publication_date=pub_date,
            fiscal_period=fiscal_period,
            properties=props,
        )
