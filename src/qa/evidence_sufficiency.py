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

    def __init__(self, min_evidence_items: int = 1, graph_retriever: Any = None) -> None:
        self.min_evidence_items = min_evidence_items
        self.graph_retriever = graph_retriever

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

        # 3. Entity Coverage
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
        missing_relationships: list[str] = []
        if analysis.relationship_intent:
            rels_found: set[str] = set()
            for ev in evidence:
                # Check graph_relations, relationship, or verbalized text
                item_rels = []
                if "graph_relations" in ev.metadata:
                    item_rels.extend(ev.metadata["graph_relations"])
                if "relationship" in ev.metadata:
                    item_rels.append(str(ev.metadata["relationship"]))

                for rel in analysis.relationship_intent:
                    if any(rel in r for r in item_rels) or rel.lower() in (ev.text or "").lower():
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

        # Multi-hop sufficiency check
        hop_sufficient = (not analysis.is_multi_hop) or (observed_hops >= required_hops)

        # 6. Overall Sufficiency Decision
        if missing_entities:
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
