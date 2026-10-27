from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any

from src.data.graph.snapshot import is_available, parse_datetime
from src.qa.query_analysis import QueryAnalysis
from src.retrieval.evidence import RetrievalEvidence

_STOP_WORDS = {"the", "and", "inc", "corp", "corporation", "ltd", "company", "co", "llc", "group", "holdings"}


@dataclass
class SufficiencyResult:
    """Structured result evaluating evidence sufficiency."""

    sufficient: bool
    reason: str
    missing_entities: list[str] = field(default_factory=list)
    missing_relationships: list[str] = field(default_factory=list)
    required_hops: int = 1
    observed_hops: int = 1
    evidence_count: int = 0
    temporal_valid: bool = True
    provenance_complete: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class EvidenceSufficiencyChecker:
    """
    Deterministic Evidence Sufficiency Evaluator.

    Verifies whether retrieved evidence adequately covers query entities,
    relationship intent, reasoning depth, temporal constraints, and provenance.
    """

    def __init__(
        self,
        min_evidence_items: int = 1,
        graph_retriever: Any = None,
        entity_mode: str = "strict",
        relationship_mode: str = "strict",
        hop_mode: str = "strict",
        gate_enabled: bool = True,
    ) -> None:
        self.min_evidence_items = min_evidence_items
        self.graph_retriever = graph_retriever
        self.entity_mode = entity_mode
        self.relationship_mode = relationship_mode
        self.hop_mode = hop_mode
        self.gate_enabled = gate_enabled

    def _get_entity_terms(self, ent: str) -> set[str]:
        raw = ent.split(":", 1)[1] if ":" in ent else ent
        terms = {raw.lower(), ent.lower()}
        ticker_upper = raw.upper()
        if ticker_upper == "NVDA":
            terms.update(["nvidia", "nvda"])
        elif ticker_upper == "AAPL":
            terms.update(["apple", "aapl"])
        elif ticker_upper == "MSFT":
            terms.update(["microsoft", "msft"])
        elif ticker_upper in ("TSM", "TSMC"):
            terms.update(["tsmc", "taiwan semiconductor", "tsm"])
        elif ticker_upper in ("GOOGL", "GOOG"):
            terms.update(["google", "alphabet"])
        elif ticker_upper == "AMZN":
            terms.update(["amazon"])
        elif ticker_upper == "META":
            terms.update(["meta", "facebook"])
        elif ticker_upper == "TSLA":
            terms.update(["tesla"])
        elif ticker_upper == "JPM":
            terms.update(["jpmorgan", "jpmorgan chase"])
        elif ticker_upper == "XOM":
            terms.update(["exxon", "exxonmobil"])
        return terms

    def _validate_compound_supplier_risk_chain(
        self,
        analysis: QueryAnalysis,
        evidence: list[RetrievalEvidence],
    ) -> tuple[bool, str]:
        """
        Validate that the retrieved evidence contains a defensible, connected,
        and directionally valid graph-edge chain for compound supplier-risk queries.
        """
        if not analysis.entities:
            return (
                False,
                "multi-hop query requires connecting supplier risks to a target entity, "
                "but no target entity was identified in the query.",
            )

        # Collect target entity terms
        target_terms: set[str] = set()
        for ent in analysis.entities:
            target_terms.update(self._get_entity_terms(ent))

        def matches_target(node_str: str) -> bool:
            if not node_str:
                return False
            n_clean = node_str.lower().strip()
            raw = n_clean.split(":", 1)[1] if ":" in n_clean else n_clean
            return n_clean in target_terms or raw in target_terms or any(t == raw or t == n_clean for t in target_terms)

        def supplier_nodes_match(s1: str, s2: str) -> bool:
            if not s1 or not s2:
                return False
            s1_lower = s1.lower().strip()
            s2_lower = s2.lower().strip()
            if s1_lower == s2_lower:
                return True
            s1_clean = s1_lower.split(":", 1)[1] if ":" in s1_lower else s1_lower
            s2_clean = s2_lower.split(":", 1)[1] if ":" in s2_lower else s2_lower
            if s1_clean == s2_clean:
                return True
            s1_terms = self._get_entity_terms(s1)
            s2_terms = self._get_entity_terms(s2)
            return bool(s1_terms.intersection(s2_terms))

        # Collect candidate paths and candidate edges from evidence
        candidate_paths: list[list[dict[str, Any]]] = []
        all_edges: list[dict[str, Any]] = []

        for ev in evidence[:10]:
            # 1. Inspect graph_edge_paths
            edge_paths = ev.metadata.get("graph_edge_paths") or []
            for path in edge_paths:
                if isinstance(path, list) and path:
                    valid_path_edges = []
                    for e in path:
                        if isinstance(e, dict) and e.get("source") and e.get("target") and e.get("relationship"):
                            valid_path_edges.append(e)
                    if valid_path_edges:
                        candidate_paths.append(valid_path_edges)
                        all_edges.extend(valid_path_edges)

            # 2. Inspect graph_edges in metadata or properties
            g_edges = ev.metadata.get("graph_edges") or ev.metadata.get("properties", {}).get("graph_edges") or []
            if isinstance(g_edges, list):
                for e in g_edges:
                    if isinstance(e, dict) and e.get("source") and e.get("target") and e.get("relationship"):
                        all_edges.append(e)

            # 3. Fallback for single edge directly in metadata
            src = ev.metadata.get("source_node") or ev.metadata.get("properties", {}).get("source_node")
            tgt = ev.metadata.get("target_node") or ev.metadata.get("properties", {}).get("target_node")
            rel = ev.metadata.get("relationship")
            if src and tgt and rel and " -> " not in str(rel):
                all_edges.append({"source": str(src), "relationship": str(rel), "target": str(tgt)})

        # First, check if any multi-edge path satisfies the connected chain
        for path in candidate_paths:
            if len(path) >= 2:
                for i in range(len(path)):
                    for j in range(len(path)):
                        if i == j:
                            continue
                        e_supp = path[i]
                        e_risk = path[j]
                        if str(e_supp.get("relationship", "")).upper() == "SUPPLIES":
                            risk_rel = str(e_risk.get("relationship", "")).upper()
                            if risk_rel in ("HAS_RISK", "AFFECTED_BY"):
                                supp_target = str(e_supp.get("target", ""))
                                supp_source = str(e_supp.get("source", ""))
                                risk_source = str(e_risk.get("source", ""))
                                if matches_target(supp_target) and not matches_target(supp_source):
                                    if supplier_nodes_match(supp_source, risk_source):
                                        return True, "Valid connected compound supplier-risk path verified."

        # Second, search across all formal edges
        valid_supp_edges: list[tuple[str, str]] = []  # (supplier_node, target_node)
        valid_risk_edges: list[tuple[str, str, str]] = []  # (supplier_node, relationship, risk_node)

        for e in all_edges:
            rel = str(e.get("relationship", "")).upper()
            src = str(e.get("source", "")).strip()
            tgt = str(e.get("target", "")).strip()
            if not src or not tgt or not rel:
                continue

            if rel == "SUPPLIES":
                # Supplier must supply target company: src (Supplier) -> tgt (Target)
                if matches_target(tgt) and not matches_target(src):
                    valid_supp_edges.append((src, tgt))
            elif rel in ("HAS_RISK", "AFFECTED_BY"):
                # Risk affecting supplier: src (Supplier) -> tgt (Risk)
                # Ensure the risk is not attached to the target company itself
                if not matches_target(src):
                    valid_risk_edges.append((src, rel, tgt))

        if not valid_supp_edges and not valid_risk_edges:
            return (
                False,
                "multi-hop query requires formal graph edges linking risks affecting supplier companies, "
                "but evidence lacks formal supplier and risk edge metadata.",
            )
        if not valid_supp_edges:
            return (
                False,
                "formal supplier relationship (SUPPLIES) to the requested entity is missing in graph metadata.",
            )
        if not valid_risk_edges:
            return (
                False,
                "formal risk relationship (HAS_RISK or AFFECTED_BY) for suppliers is missing in graph metadata.",
            )

        # Check connectivity: does any supplier from valid_supp_edges match a supplier from valid_risk_edges?
        for supp_node, _ in valid_supp_edges:
            for risk_supp_node, _, _ in valid_risk_edges:
                if supplier_nodes_match(supp_node, risk_supp_node):
                    return True, "Valid connected compound supplier-risk chain verified."

        return (
            False,
            "formal supplier and risk relationships were retrieved but are disconnected "
            "(the identified supplier does not match the entity affected by the risk).",
        )

    def check(
        self,
        analysis: QueryAnalysis,
        evidence: list[RetrievalEvidence],
    ) -> SufficiencyResult:
        """
        Evaluate if retrieved evidence is sufficient to answer the query.

        Args:
            analysis: Analyzed query semantics and constraints.
            evidence: List of retrieved evidence items.

        Returns:
            SufficiencyResult: Structured sufficiency evaluation.
        """
        evidence_count = len(evidence)
        required_hops = analysis.expected_hop_depth

        if evidence_count < self.min_evidence_items:
            return SufficiencyResult(
                sufficient=False,
                reason="No evidence items retrieved.",
                missing_entities=list(analysis.entities),
                missing_relationships=list(analysis.relationship_intent),
                required_hops=required_hops,
                observed_hops=0,
                evidence_count=0,
                temporal_valid=True,
                provenance_complete=False,
            )

        # 1. Temporal Validity Verification
        temporal_valid = True
        if analysis.reference_time is not None:
            ref_dt = parse_datetime(analysis.reference_time)
            if ref_dt is not None:
                for ev in evidence:
                    if not is_available(ev.available_time, ref_dt):
                        temporal_valid = False
                        break

        if not temporal_valid:
            return SufficiencyResult(
                sufficient=False,
                reason="Retrieved evidence contains temporal violations exceeding the reference time.",
                missing_entities=[],
                missing_relationships=[],
                required_hops=required_hops,
                observed_hops=0,
                evidence_count=evidence_count,
                temporal_valid=False,
                provenance_complete=True,
            )

        # 2. Provenance Completeness
        provenance_complete = any(
            bool(ev.document_id and (ev.chunk_id or ev.available_time))
            for ev in evidence
        )
        if not provenance_complete:
            return SufficiencyResult(
                sufficient=False,
                reason="Retrieved evidence lacks complete SEC document provenance.",
                missing_entities=[],
                missing_relationships=[],
                required_hops=required_hops,
                observed_hops=1,
                evidence_count=evidence_count,
                temporal_valid=True,
                provenance_complete=False,
            )

        # 3. Short-circuit if gate is disabled (CFG5: no-gate unconstrained ablation)
        if not self.gate_enabled:
            return SufficiencyResult(
                sufficient=True,
                reason="Sufficiency gate disabled (no-gate unconstrained generation ablation).",
                missing_entities=[],
                missing_relationships=[],
                required_hops=required_hops,
                observed_hops=1,
                evidence_count=evidence_count,
                temporal_valid=True,
                provenance_complete=True,
            )

        # 4. Entity Coverage
        missing_entities: list[str] = []
        for ent in analysis.entities:
            # Build search terms for this entity (ticker, name, corporate aliases)
            raw = ent.split(":", 1)[1] if ":" in ent else ent
            candidate_terms = {raw.lower(), ent.lower()}

            if (
                self.graph_retriever
                and hasattr(self.graph_retriever, "nodes")
                and ent in self.graph_retriever.nodes
            ):
                node = self.graph_retriever.nodes[ent]
                name = node.properties.get("canonical_name") or node.properties.get("name")
                if name:
                    candidate_terms.add(name.lower())
                    for part in name.lower().split():
                        if len(part) >= 3 and part not in _STOP_WORDS:
                            candidate_terms.add(part)

            ticker_upper = raw.upper()
            if ticker_upper == "NVDA":
                candidate_terms.update(["nvidia", "nvda"])
            elif ticker_upper == "AAPL":
                candidate_terms.update(["apple", "aapl"])
            elif ticker_upper == "MSFT":
                candidate_terms.update(["microsoft", "msft"])
            elif ticker_upper in ("TSM", "TSMC"):
                candidate_terms.update(["tsmc", "taiwan semiconductor", "tsm"])
            elif ticker_upper in ("GOOGL", "GOOG"):
                candidate_terms.update(["google", "alphabet"])
            elif ticker_upper == "AMZN":
                candidate_terms.update(["amazon"])
            elif ticker_upper == "META":
                candidate_terms.update(["meta", "facebook"])

            entity_covered = False
            for ev in evidence:
                path = ev.metadata.get("path") or []
                if ent in path or raw in path:
                    entity_covered = True
                    break

                text_lower = (ev.text or "").lower()
                path_str = " ".join(str(p).lower() for p in path)
                metadata_str = str(ev.metadata).lower()

                if any(
                    term in text_lower or term in path_str or term in metadata_str
                    for term in candidate_terms
                ):
                    entity_covered = True
                    break

            if not entity_covered:
                missing_entities.append(ent)

        # 4. Relationship Intent Coverage
        RELATIONSHIP_PHRASE_MAP: dict[str, list[str]] = {
            "SUPPLIES": ["suppl", "vendor", "provide", "furnish"],
            "MANUFACTURES": ["manufactur", "produc", "build", "assembl", "fabricat"],
            "HAS_RISK": ["risk", "advers", "threat", "hazard", "vulnerab"],
            "AFFECTED_BY": ["affect", "impact", "subject to", "disrupt"],
            "DEPENDS_ON": ["depend", "reli", "subject to"],
            "PARTNERS_WITH": ["partner", "collaborat", "joint venture", "alliance"],
            "COMPETES_WITH": ["compet", "rival"],
            "ACQUIRED": ["acquir", "merg", "bought", "acquisition"],
            "LOCATED_IN": ["headquarter", "locat", "operat in", "facility"],
        }

        missing_relationships: list[str] = []
        if analysis.relationship_intent:
            rels_found: set[str] = set()
            for ev in evidence:
                item_rels: list[str] = []
                if "graph_relations" in ev.metadata and ev.metadata["graph_relations"]:
                    item_rels.extend([str(r).upper() for r in ev.metadata["graph_relations"]])
                if "relationship" in ev.metadata and ev.metadata["relationship"]:
                    item_rels.append(str(ev.metadata["relationship"]).upper())
                for e in ev.metadata.get("graph_edges", []) or []:
                    if isinstance(e, dict) and e.get("relationship"):
                        item_rels.append(str(e["relationship"]).upper())
                for path in ev.metadata.get("graph_edge_paths", []) or []:
                    if isinstance(path, list):
                        for e in path:
                            if isinstance(e, dict) and e.get("relationship"):
                                item_rels.append(str(e["relationship"]).upper())

                for rel in analysis.relationship_intent:
                    rel_upper = rel.upper()
                    # 1. Formal edge metadata check (valid in both strict and textual modes)
                    if any(rel_upper == r or rel_upper in r for r in item_rels):
                        rels_found.add(rel)
                    # 2. Textual mode fallback: inspect documented relationship phrases/keywords in retrieved snippet
                    elif self.relationship_mode == "textual":
                        text_lower = (ev.text or "").lower()
                        keywords = RELATIONSHIP_PHRASE_MAP.get(rel_upper, [rel.lower()])
                        if any(kw in text_lower for kw in keywords) or rel.lower() in text_lower:
                            rels_found.add(rel)

            for rel in analysis.relationship_intent:
                if rel not in rels_found:
                    missing_relationships.append(rel)

        # 5. Observed Hop Depth
        observed_hops = 1
        for ev in evidence:
            hop = int(ev.metadata.get("hop_count", 1))
            if hop > observed_hops:
                observed_hops = hop

        # Multi-hop sufficiency check according to hop_mode
        if self.hop_mode == "strict":
            hop_sufficient = (not analysis.is_multi_hop) or (observed_hops >= required_hops)
        else:
            # Soft or disabled hop mode: allows single-hop evidence synthesis
            hop_sufficient = True

        # 6. Overall Sufficiency Decision
        # A. Entity Coverage Gate
        if self.entity_mode != "disabled":
            if self.entity_mode == "strict" and missing_entities:
                return SufficiencyResult(
                    sufficient=False,
                    reason=f"Missing evidence coverage for key query entities: {missing_entities}",
                    missing_entities=missing_entities,
                    missing_relationships=missing_relationships,
                    required_hops=required_hops,
                    observed_hops=observed_hops,
                    evidence_count=evidence_count,
                    temporal_valid=True,
                    provenance_complete=True,
                )
            elif self.entity_mode == "partial" and len(analysis.entities) > 0:
                covered_count = len(analysis.entities) - len(missing_entities)
                # In partial mode, fail only if 0 entities are covered or strictly < 50% for multi-entity sets
                if covered_count == 0 or (len(analysis.entities) > 1 and covered_count < (len(analysis.entities) / 2.0)):
                    return SufficiencyResult(
                        sufficient=False,
                        reason=f"Insufficient partial entity coverage ({covered_count}/{len(analysis.entities)} covered): missing {missing_entities}",
                        missing_entities=missing_entities,
                        missing_relationships=missing_relationships,
                        required_hops=required_hops,
                        observed_hops=observed_hops,
                        evidence_count=evidence_count,
                        temporal_valid=True,
                        provenance_complete=True,
                    )

        # B. Relationship Coverage Gate
        query_lower = analysis.original_query.lower()
        wants_supplier = any(w in query_lower for w in ("supply", "supplies", "supplier", "suppliers"))
        wants_risk = any(w in query_lower for w in ("risk", "risks", "affect", "affects", "affected"))
        is_compound_supplier_risk = (
            analysis.is_multi_hop and wants_supplier and wants_risk
        )

        if is_compound_supplier_risk:
            if self.relationship_mode == "strict":
                chain_valid, chain_reason = self._validate_compound_supplier_risk_chain(analysis, evidence)
                if not chain_valid:
                    return SufficiencyResult(
                        sufficient=False,
                        reason=f"Insufficient compound relationship coverage: {chain_reason}",
                        missing_entities=[],
                        missing_relationships=["HAS_RISK", "AFFECTED_BY"],
                        required_hops=required_hops,
                        observed_hops=observed_hops,
                        evidence_count=evidence_count,
                        temporal_valid=True,
                        provenance_complete=True,
                    )
            elif self.relationship_mode == "textual":
                has_text_supp = any(
                    any(kw in (ev.text or "").lower() for kw in ("suppl", "vendor", "foundry", "manufactur"))
                    or "SUPPLIES" in str(ev.metadata.get("relationship", "")).upper()
                    or any("SUPPLIES" in str(r).upper() for r in ev.metadata.get("graph_relations") or [])
                    for ev in evidence[:5]
                )
                has_text_risk = any(
                    any(kw in (ev.text or "").lower() for kw in ("risk", "advers", "threat", "disrupt", "shortage"))
                    or "HAS_RISK" in str(ev.metadata.get("relationship", "")).upper()
                    or "AFFECTED_BY" in str(ev.metadata.get("relationship", "")).upper()
                    or any("HAS_RISK" in str(r).upper() or "AFFECTED_BY" in str(r).upper() for r in ev.metadata.get("graph_relations") or [])
                    for ev in evidence[:5]
                )
                if not (has_text_supp and has_text_risk):
                    return SufficiencyResult(
                        sufficient=False,
                        reason="Insufficient compound relationship coverage: textual mode requires mentions of supplier and risk.",
                        missing_entities=[],
                        missing_relationships=["HAS_RISK", "AFFECTED_BY"],
                        required_hops=required_hops,
                        observed_hops=observed_hops,
                        evidence_count=evidence_count,
                        temporal_valid=True,
                        provenance_complete=True,
                    )

        if self.relationship_mode in ("strict", "textual"):
            if missing_relationships and len(missing_relationships) == len(analysis.relationship_intent):
                return SufficiencyResult(
                    sufficient=False,
                    reason=f"Missing evidence coverage for intended relationships: {missing_relationships}",
                    missing_entities=[],
                    missing_relationships=missing_relationships,
                    required_hops=required_hops,
                    observed_hops=observed_hops,
                    evidence_count=evidence_count,
                    temporal_valid=True,
                    provenance_complete=True,
                )

        # C. Reasoning Hop Gate
        if not hop_sufficient:
            return SufficiencyResult(
                sufficient=False,
                reason=f"Insufficient multi-hop reasoning depth: observed {observed_hops} hops, required {required_hops}.",
                missing_entities=[],
                missing_relationships=missing_relationships,
                required_hops=required_hops,
                observed_hops=observed_hops,
                evidence_count=evidence_count,
                temporal_valid=True,
                provenance_complete=True,
            )

        return SufficiencyResult(
            sufficient=True,
            reason="Retrieved evidence sufficiently covers entities, relationships, temporal constraints, and reasoning depth.",
            missing_entities=[],
            missing_relationships=[],
            required_hops=required_hops,
            observed_hops=observed_hops,
            evidence_count=evidence_count,
            temporal_valid=True,
            provenance_complete=True,
        )
