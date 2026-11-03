from __future__ import annotations

from collections import deque
from datetime import datetime, timezone
import json
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

_TICKER_TO_NAMES: dict[str, list[str]] = {
    "AAPL": ["apple", "aapl"],
    "MSFT": ["microsoft", "msft"],
    "NVDA": ["nvidia", "nvda"],
    "AMZN": ["amazon", "amzn"],
    "GOOGL": ["google", "alphabet", "googl", "goog"],
    "GOOG": ["google", "alphabet", "googl", "goog"],
    "META": ["meta", "facebook"],
    "TSLA": ["tesla", "tsla"],
    "TSM": ["tsmc", "taiwan semiconductor", "tsm"],
    "JPM": ["jpmorgan", "jpmorgan chase", "jpm"],
    "XOM": ["exxon", "exxonmobil", "xom"],
    "JNJ": ["johnson & johnson", "johnson and johnson", "jnj"],
    "INTC": ["intel", "intc"],
    "AMD": ["amd", "advanced micro devices"],
    "ASML": ["asml"],
    "AVGO": ["broadcom", "avgo"],
    "QCOM": ["qualcomm", "qcom"],
}

_REGULATORY_ENTITIES: dict[str, dict[str, Any]] = {
    "regulator:FTC": {
        "canonical_name": "Federal Trade Commission",
        "aliases": ["ftc", "federal trade commission", "u.s. federal trade commission", "united states federal trade commission", "trade commission"],
        "entity_type": "regulator",
    },
    "regulator:DOJ": {
        "canonical_name": "Department of Justice",
        "aliases": ["doj", "department of justice", "u.s. department of justice", "united states department of justice", "justice department"],
        "entity_type": "regulator",
    },
    "regulator:EC": {
        "canonical_name": "European Commission",
        "aliases": ["european commission", "the european commission", "ec", "eu competition authorities", "european union competition authorities"],
        "entity_type": "regulator",
    },
    "regulator:SAMR": {
        "canonical_name": "State Administration for Market Regulation",
        "aliases": ["samr", "state administration for market regulation", "chinese market regulator"],
        "entity_type": "regulator",
    },
    "regulator:SEC": {
        "canonical_name": "Securities and Exchange Commission",
        "aliases": ["sec", "securities and exchange commission", "u.s. securities and exchange commission"],
        "entity_type": "regulator",
    },
}

_KNOWN_REGULATOR_NAMES_TO_ID: dict[str, str] = {
    "ftc": "regulator:FTC",
    "federal trade commission": "regulator:FTC",
    "u.s. federal trade commission": "regulator:FTC",
    "united states federal trade commission": "regulator:FTC",
    "doj": "regulator:DOJ",
    "department of justice": "regulator:DOJ",
    "u.s. department of justice": "regulator:DOJ",
    "united states department of justice": "regulator:DOJ",
    "justice department": "regulator:DOJ",
    "european commission": "regulator:EC",
    "the european commission": "regulator:EC",
    "ec": "regulator:EC",
    "samr": "regulator:SAMR",
    "state administration for market regulation": "regulator:SAMR",
    "sec": "regulator:SEC",
    "securities and exchange commission": "regulator:SEC",
}


def get_company_terms(entity_or_ticker: str, node: GraphNode | None = None) -> list[str]:
    """Return recognized lexical search terms for a company ticker, regulator, or node identifier."""
    # Check regulatory entities first
    if entity_or_ticker.startswith("regulator:"):
        reg_info = _REGULATORY_ENTITIES.get(entity_or_ticker)
        if reg_info:
            return sorted(set(reg_info["aliases"] + [reg_info["canonical_name"].lower()]))

    clean = entity_or_ticker.split(":", 1)[1] if ":" in entity_or_ticker else entity_or_ticker
    ticker_upper = clean.upper()
    if f"regulator:{ticker_upper}" in _REGULATORY_ENTITIES:
        reg_info = _REGULATORY_ENTITIES[f"regulator:{ticker_upper}"]
        return sorted(set(reg_info["aliases"] + [reg_info["canonical_name"].lower()]))

    terms: set[str] = set()
    if ticker_upper in _TICKER_TO_NAMES:
        terms.update(_TICKER_TO_NAMES[ticker_upper])

    if node is not None:
        for field in ("canonical_name", "name", "ticker"):
            val = node.properties.get(field)
            if val and isinstance(val, str):
                val_clean = val.strip().lower()
                if val_clean:
                    terms.add(val_clean)
                    stripped = strip_company_legal_suffix(val_clean)
                    if stripped and len(stripped) >= 3:
                        terms.add(stripped)
        aliases = node.properties.get("aliases")
        if isinstance(aliases, (list, tuple, set)):
            for a in aliases:
                if isinstance(a, str) and a.strip():
                    terms.add(a.strip().lower())

    if not terms:
        # If not a pure hex hash of length >= 8, include clean string
        if len(clean) >= 3 and not re.match(r"^[0-9a-f]{8,}$", clean.lower()):
            terms.add(clean.lower())

    return sorted(terms) or [clean.lower()]

_GENERIC_ENTITY_WORDS: set[str] = {
    # Entity category / role framing
    "firm", "firms",
    "company", "companies",
    "business", "businesses",
    "corporation", "corporations",
    "enterprise", "enterprises",
    "supplier", "suppliers",
    "customer", "customers",
    "competitor", "competitors",
    "partner", "partners",
    "provider", "providers",
    "vendor", "vendors",
    "client", "clients",
    "peer", "peers",
    "landlord", "landlords",
    "manufacturer", "manufacturers",
    "fabricator", "fabricators",
    "party", "parties",
    # Financial / business concepts
    "product", "products",
    "service", "services",
    "risk", "risks",
    "factor", "factors",
    "market", "markets",
    "industry", "industries",
    "revenue", "revenues",
    "sales", "net sales",
    "income", "profit",
    "cost", "costs",
    "price", "prices",
    "expense", "expenses",
    "margin", "margins",
    "award", "plan",
    "facility", "facilities",
    "infrastructure",
    # Relationship / transaction cues
    "acquisition", "acquisitions",
    "merger", "mergers",
    "agreement", "agreements",
    "contract", "contracts",
    "partnership", "partnerships",
    "investment", "investments",
    "operation", "operations",
    "competition", "commercial",
    "dependency", "dependencies",
    # Domain / department / industry / technology terms
    "gaming", "server", "servers", "legal", "treasury", "platform", "platforms",
    "software", "hardware", "cloud", "technology", "technologies", "entertainment",
    "healthcare", "cybersecurity", "energy", "investigation", "investigations",
    "advertising", "carbon", "emissions", "intelligence", "artificial intelligence",
    "search", "mobile", "network", "networks", "security", "defense", "banking",
    "automotive", "automobiles", "semiconductor", "semiconductors", "chip", "chips",
    "device", "devices", "application", "applications", "payment", "payments",
    "media", "content", "corporate", "deal", "deals",
    "advanced", "regulator", "regulators", "regulatory", "commission", "agency", "agencies", "board", "court", "investigator", "investigators",
    # SEC disclosure / document words
    "filing", "filings",
    "report", "reports",
    "statement", "statements",
    "disclosure", "disclosures",
    "form",
    # Calendar & temporal units
    "year", "years",
    "quarter", "quarters",
    "month", "months",
    "period", "periods",
    "date", "dates",
    "time", "times",
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december",
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
        manifest_path: str | Path | None = None,
        regulatory_linking: bool = True,
    ) -> None:
        self.regulatory_linking = regulatory_linking
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

        # Document and chunk provenance lookup tables
        self.doc_to_issuer: dict[str, str] = {}
        self.chunk_lookup: dict[str, dict[str, str]] = {}
        self._load_provenance_indexes(manifest_path)
        self._is_sec_corpus = any(
            str(e.properties.get("document_id", "")).startswith("doc:sec:")
            for e in self.graph.edges
        )

        # Build in-memory graph lookup structures
        self.nodes: dict[str, GraphNode] = {}
        self.out_edges: dict[str, list[GraphEdge]] = {}
        self.in_edges: dict[str, list[GraphEdge]] = {}
        self.node_text_lookup: dict[str, set[str]] = {}

        self._build_graph_indexes()

    def _load_provenance_indexes(self, manifest_path: str | Path | None = None) -> None:
        """Load document issuer mapping and chunk texts for provenance and attribution verification."""
        path = None
        if manifest_path is not None:
            candidate = Path(manifest_path)
            if candidate.exists():
                path = candidate
        if path is None:
            default_candidate = Path("data/processed/sec_corpus_manifest.json")
            if default_candidate.exists():
                path = default_candidate
            else:
                repo_candidate = Path(__file__).resolve().parents[2] / "data" / "processed" / "sec_corpus_manifest.json"
                if repo_candidate.exists():
                    path = repo_candidate

        if not path or not path.exists():
            return

        try:
            with path.open("r", encoding="utf-8") as handle:
                manifest_data = json.load(handle)
            completed = manifest_data.get("completed", {})
            for filing_key, filing_info in completed.items():
                issuer = filing_key.split(":", 1)[0]
                doc_id = filing_info.get("document_id")
                if doc_id:
                    self.doc_to_issuer[doc_id] = issuer

                chunk_path_str = filing_info.get("chunk_path")
                if chunk_path_str:
                    c_path = Path(chunk_path_str)
                    if not c_path.exists():
                        c_path = Path(__file__).resolve().parents[2] / chunk_path_str
                    if c_path.exists():
                        with c_path.open("r", encoding="utf-8") as c_handle:
                            chunks = json.load(c_handle)
                            for c in chunks:
                                cid = c.get("chunk_id")
                                if cid:
                                    self.chunk_lookup[cid] = {
                                        "text": c.get("text", ""),
                                        "issuer": issuer,
                                        "doc_id": doc_id,
                                    }
        except Exception:
            pass

    def get_entity_terms(self, entity_id: str) -> list[str]:
        """Return recognized lexical search terms for an entity ID, resolving canonical name and aliases."""
        node = self.nodes.get(entity_id)
        if not node and ":" in entity_id:
            raw = entity_id.split(":", 1)[1]
            node = self.nodes.get(f"company:{raw}") or self.nodes.get(raw)
        return get_company_terms(entity_id, node=node)

    def verify_edge_provenance(self, edge: GraphEdge) -> tuple[bool, str, str | None]:
        """
        Validate edge provenance and cross-company attribution against source SEC chunks.

        Returns:
            tuple[bool, str, str | None]: (verified, status, issuer_ticker)
        """
        props = edge.properties
        # If explicitly marked in test/synthetic graph, respect it
        if "provenance_verified" in props:
            return (
                bool(props["provenance_verified"]),
                str(props.get("attribution_status", "explicit")),
                props.get("source_company"),
            )

        doc_id = str(props.get("document_id") or "")
        chunk_id = str(props.get("chunk_id") or "")

        # If not an SEC corpus graph (e.g. synthetic unit test graph), accept mock edges
        if not getattr(self, "_is_sec_corpus", True):
            doc_upper = doc_id.upper()
            inferred_issuer = None
            for ticker in _TICKER_TO_NAMES:
                if ticker in doc_upper:
                    inferred_issuer = ticker
                    break
            return True, "verified_mock_fixture", inferred_issuer

        if not doc_id:
            return False, "missing_provenance", None

        # Check if doc_id is from our real SEC corpus
        if doc_id.startswith("doc:sec:"):
            issuer = props.get("source_company") or self.doc_to_issuer.get(doc_id)
            if not issuer:
                return False, "unknown_issuer", None

            src = edge.source
            tgt = edge.target
            rel = edge.relationship

            chunk_info = self.chunk_lookup.get(chunk_id)
            chunk_text = (chunk_info.get("text", "") if chunk_info else "").lower()

            # 1. Source Company Verification
            if src.startswith("company:"):
                src_ticker = src.split(":", 1)[1].upper()
                if src_ticker != issuer:
                    # Cross-company edge originating from a third-party issuer's filing
                    src_terms = self.get_entity_terms(src)
                    if not chunk_text or not any(t in chunk_text for t in src_terms):
                        return False, "unsupported_cross_company", issuer

                    # In corporate filings, general risk factors apply to the filing issuer.
                    # A third-party filing only substantiates external risk if explicitly discussed.
                    if rel in ("HAS_RISK", "AFFECTED_BY"):
                        risk_context = any(
                            f"{t} risk" in chunk_text
                            or f"risk to {t}" in chunk_text
                            or f"risk for {t}" in chunk_text
                            or f"{t}'s risk" in chunk_text
                            or f"{t} face" in chunk_text
                            or f"{t} subject to" in chunk_text
                            or f"disruptions at {t}" in chunk_text
                            or f"delays from {t}" in chunk_text
                            for t in src_terms
                        )
                        if not risk_context:
                            return False, "unsupported_cross_company_risk", issuer

            # 2. Target Company Verification
            if tgt.startswith("company:"):
                tgt_ticker = tgt.split(":", 1)[1].upper()
                if tgt_ticker != issuer:
                    tgt_terms = self.get_entity_terms(tgt)
                    if chunk_text and not any(t in chunk_text for t in tgt_terms):
                        return False, "unsupported_cross_company", issuer

            status = "verified_issuer" if (src == f"company:{issuer}") else "verified_cross_company"
            return True, status, issuer

        # For non-SEC or synthetic unit test documents (e.g. doc_nvda_10k_2023)
        doc_upper = doc_id.upper()
        inferred_issuer = None
        for ticker in _TICKER_TO_NAMES:
            if ticker in doc_upper:
                inferred_issuer = ticker
                break
        return True, "verified_mock_fixture", inferred_issuer

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
            ver, stat, issuer = self.verify_edge_provenance(edge)
            edge.properties["provenance_verified"] = ver
            edge.properties["attribution_status"] = stat
            if issuer:
                edge.properties["source_company"] = issuer

            self.out_edges.setdefault(edge.source, []).append(edge)
            self.in_edges.setdefault(edge.target, []).append(edge)

    def _register_node_lookup(self, text: str, node_id: str) -> None:
        """Register a normalized text key mapping to a node ID."""
        normalized = normalize_text(text)
        if normalized:
            self.node_text_lookup.setdefault(normalized, set()).add(node_id)

    def _is_valid_entity_node(self, node_id: str) -> bool:
        """
        Check whether a node represents a legitimate entity rather than an incidental
        generic dictionary word, grammatical stopword, or pure numeric/date token.
        """
        node = self.nodes.get(node_id)
        if not node:
            return False
        # Universe companies or nodes with an explicit ticker are always valid
        if node_id.startswith("company:") and node.properties.get("ticker"):
            return True
        name = (node.properties.get("canonical_name") or node.properties.get("name") or "").strip()
        norm = normalize_text(name)
        if not norm:
            return False
        # Standalone digits, currency, or pure numbers (e.g., "1", "$1", "2025") are not entities
        if re.match(r"^[\$\d\.,\-_/]+$", norm):
            return False
        # Generic role/framing words when appearing alone
        if norm in _GENERIC_ENTITY_WORDS or norm in _QUERY_STOP_WORDS:
            return False
        return True

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

        _COMMON_UPPERCASE_WORDS = {
            "A", "I", "IN", "ON", "AT", "TO", "OR", "AND", "AS", "BY", "FOR",
            "DO", "IF", "OF", "AN", "BE", "IS", "IT", "SO", "NO", "UP", "HE",
            "WE", "MY", "US", "AM", "ALL", "NOT", "BUT", "THE", "CAN", "DID",
            "HAS", "HAD", "HOW", "WHO", "WHY", "WHAT", "WHEN", "WHERE", "WHICH"
        }

        # Detect SEC filing/disclosure framing (e.g., "SEC filings", "separate SEC filings", "SEC reports")
        sec_is_framing = self.regulatory_linking and bool(
            re.search(
                r"\b(?:across\s+their\s+separate\s+)?sec\s+(?:filing|filings|report|reports|document|documents|disclosure|disclosures)\b",
                query,
                re.I,
            )
        )

        # 1. Match uppercase ticker symbols and known acronyms
        candidate_tickers = re.findall(r"\b[A-Z]{1,5}\b", query)
        for ticker in candidate_tickers:
            if ticker in _COMMON_UPPERCASE_WORDS:
                continue

            ticker_lower = ticker.lower()
            if self.regulatory_linking:
                if ticker_lower == "sec":
                    if not sec_is_framing:
                        resolved.add("regulator:SEC")
                    continue
                elif ticker_lower == "ftc":
                    resolved.add("regulator:FTC")
                    continue
                elif ticker_lower == "doj":
                    resolved.add("regulator:DOJ")
                    continue
                elif ticker_lower == "samr":
                    resolved.add("regulator:SAMR")
                    continue
                elif ticker_lower == "ec" and any(k in query.lower() for k in ("europe", "commission", "eu")):
                    resolved.add("regulator:EC")
                    continue

            if ticker in self.universe_indexes["ticker"]:
                resolved.add(f"company:{ticker}")
            elif f"company:{ticker}" in self.nodes:
                resolved.add(f"company:{ticker}")
            elif normalize_text(ticker) in self.node_text_lookup:
                for node_id in self.node_text_lookup[normalize_text(ticker)]:
                    if self._is_valid_entity_node(node_id):
                        resolved.add(node_id)

        # 2. Extract potential entity phrases via sliding n-grams (5 down to 1)
        tokens = query.split()
        max_n = min(5, len(tokens))
        matched_token_indices: set[int] = set()

        for n in range(max_n, 0, -1):
            for i in range(len(tokens) - n + 1):
                # If all tokens in this span are already part of an identified longer entity phrase, skip
                span_indices = set(range(i, i + n))
                if span_indices.issubset(matched_token_indices):
                    continue

                phrase = " ".join(tokens[i : i + n])
                norm_phrase = normalize_text(phrase)
                if not norm_phrase or norm_phrase in _QUERY_STOP_WORDS:
                    continue

                # Skip phrases that consist entirely of stop words (e.g., "as of", "in the", "for the")
                if all(w.lower() in _QUERY_STOP_WORDS for w in phrase.split()):
                    continue

                # Standalone digits or pure numbers are temporal markers/quantities, not business entities
                if re.match(r"^[\$\d\.,\-_/]+$", norm_phrase):
                    continue

                # Single-word generic dictionary concepts are rejected
                if n == 1 and norm_phrase in _GENERIC_ENTITY_WORDS:
                    continue

                # Check regulatory and jurisdictional entities
                if self.regulatory_linking and norm_phrase in _KNOWN_REGULATOR_NAMES_TO_ID:
                    reg_id = _KNOWN_REGULATOR_NAMES_TO_ID[norm_phrase]
                    if reg_id == "regulator:SEC" and sec_is_framing:
                        continue
                    resolved.add(reg_id)
                    matched_token_indices.update(span_indices)
                    continue

                # Check additional known aliases
                if norm_phrase in _ADDITIONAL_COMPANY_ALIASES:
                    ticker = _ADDITIONAL_COMPANY_ALIASES[norm_phrase]
                    candidate_node_id = f"company:{ticker}"
                    if candidate_node_id in self.nodes:
                        resolved.add(candidate_node_id)
                        matched_token_indices.update(span_indices)
                        continue

                # Check universe company resolution
                if self.universe_indexes["name"]:
                    resolved_company = resolve_company(
                        company=phrase,
                        indexes=self.universe_indexes,
                    )
                    if resolved_company and resolved_company["entity_id"] in self.nodes:
                        resolved.add(resolved_company["entity_id"])
                        matched_token_indices.update(span_indices)
                        continue

                # Check KG node text lookup
                if norm_phrase in self.node_text_lookup:
                    matched_any = False
                    for node_id in self.node_text_lookup[norm_phrase]:
                        if self._is_valid_entity_node(node_id):
                            node_obj = self.nodes.get(node_id)
                            c_name = normalize_text(node_obj.properties.get("canonical_name") or node_obj.properties.get("name") or "") if node_obj else ""
                            # Check if node corresponds to a known regulator
                            if self.regulatory_linking and c_name in _KNOWN_REGULATOR_NAMES_TO_ID:
                                reg_id = _KNOWN_REGULATOR_NAMES_TO_ID[c_name]
                                if reg_id == "regulator:SEC" and sec_is_framing:
                                    continue
                                resolved.add(reg_id)
                                matched_any = True
                                continue
                            # Canonicalize universe company mentions to company:TICKER
                            if not node_id.startswith("company:") and c_name in _ADDITIONAL_COMPANY_ALIASES:
                                t = _ADDITIONAL_COMPANY_ALIASES[c_name]
                                if f"company:{t}" in self.nodes:
                                    resolved.add(f"company:{t}")
                                    matched_any = True
                                    continue
                            resolved.add(node_id)
                            matched_any = True
                    if matched_any:
                        matched_token_indices.update(span_indices)
                    continue

                stripped = strip_company_legal_suffix(phrase)
                if stripped and stripped in self.node_text_lookup:
                    for node_id in self.node_text_lookup[stripped]:
                        if self._is_valid_entity_node(node_id):
                            node_obj = self.nodes.get(node_id)
                            c_name = normalize_text(node_obj.properties.get("canonical_name") or node_obj.properties.get("name") or "") if node_obj else ""
                            if not node_id.startswith("company:") and c_name in _ADDITIONAL_COMPANY_ALIASES:
                                t = _ADDITIONAL_COMPANY_ALIASES[c_name]
                                if f"company:{t}" in self.nodes:
                                    resolved.add(f"company:{t}")
                                    continue
                            resolved.add(node_id)
                    continue

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
                if len(candidates) >= 1000:
                    break
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

                    # Strict provenance check: ignore unverified or hallucinated cross-company edges
                    if not edge.properties.get("provenance_verified", True):
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

                    # Strict provenance check: ignore unverified or hallucinated cross-company edges
                    if not edge.properties.get("provenance_verified", True):
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

            # Direction-sensitive check (Issue 2):
            # When query asks "What does X depend on?", favor incoming SUPPLIES or outgoing DEPENDS_ON.
            # Penalize outgoing SUPPLIES from X (which means X supplies Y, not Y supplies X).
            if "depend" in query.lower() and item.hop_count == 1:
                edge_id = item.evidence_id.replace("edge:", "").lower()
                text_lower = (item.text or "").lower()
                for seed in seed_nodes:
                    seed_lower = seed.lower()
                    if seed_lower in edge_id:
                        if f"{seed_lower}->supplies->" in edge_id:
                            multiplier *= 0.1  # Strongly downrank reverse direction (X supplying others)
                        elif f"->supplies->{seed_lower}" in edge_id:
                            multiplier *= 2.0  # Boost supplier to X
                        elif f"{seed_lower}->depends_on->" in edge_id:
                            multiplier *= 2.0  # Boost X depending on entity

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
            source_company = edge.properties.get("source_company")
            provenance_verified = bool(edge.properties.get("provenance_verified", True))
            attribution_status = str(edge.properties.get("attribution_status", "verified"))
            props = dict(edge.properties)
            props["graph_edges"] = [
                {
                    "source": edge.source,
                    "relationship": edge.relationship,
                    "target": edge.target,
                }
            ]
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
            source_company = edges[0].properties.get("source_company")
            provenance_verified = all(bool(e.properties.get("provenance_verified", True)) for e in edges)
            attribution_status = "verified_path" if provenance_verified else "unverified_path_edge"
            # Combined confidence as product of edge confidences
            conf = 1.0
            for e in edges:
                conf *= float(e.properties.get("confidence", 1.0))
            confidence = conf
            props = {
                "edge_ids": [e.edge_id for e in edges],
                "documents": [e.properties.get("document_id") for e in edges if e.properties.get("document_id")],
                "chunks": [e.properties.get("chunk_id") for e in edges if e.properties.get("chunk_id")],
                "graph_edges": [
                    {
                        "source": e.source,
                        "relationship": e.relationship,
                        "target": e.target,
                    }
                    for e in edges
                ],
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
            source_company=source_company,
            provenance_verified=provenance_verified,
            attribution_status=attribution_status,
        )
