from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any

from src.data.graph.snapshot import is_available, parse_datetime
from src.qa.evidence_sufficiency import SufficiencyResult
from src.qa.query_analysis import QueryAnalysis
from src.retrieval.evidence import RetrievalEvidence

if TYPE_CHECKING:
    from src.retrieval.orchestrator import RetrievalOrchestrator


@dataclass
class QAEvidencePackage:
    """Consolidated evidence package for QA reasoning and generation."""

    query: str
    reference_time: str | None
    question_type: str
    expected_hops: int
    evidence: list[RetrievalEvidence]
    graph_paths: list[list[str]]
    sources: list[str]
    retrieval_rounds: int
    temporal_valid: bool
    sufficient: bool
    trace: list[dict[str, Any]] = field(default_factory=list)
    primary_evidence: list[RetrievalEvidence] = field(default_factory=list)
    supporting_evidence: list[RetrievalEvidence] = field(default_factory=list)
    sufficiency_result: SufficiencyResult | None = None
    retrieval_trace: list[dict[str, Any]] = field(default_factory=list)
    analysis: QueryAnalysis | None = None

    def __post_init__(self) -> None:
        if not self.primary_evidence and self.evidence:
            self.primary_evidence = self.evidence[:5]
            self.supporting_evidence = self.evidence[5:]
        if not self.retrieval_trace and self.trace:
            self.retrieval_trace = self.trace
        elif not self.trace and self.retrieval_trace:
            self.trace = self.retrieval_trace

    @property
    def all_evidence(self) -> list[RetrievalEvidence]:
        return self.evidence

    @property
    def evidence_count(self) -> int:
        return len(self.evidence)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["evidence"] = [ev.to_dict() for ev in self.evidence]
        return d


class RetrievalAgent:
    """
    Lightweight, controlled ReAct-style Retrieval Agent.

    Executes a bounded retrieval loop (maximum 2 additional iterations,
    maximum 3 total rounds) to resolve missing entities, relationships,
    or reasoning depth without unrestricted autonomy.
    """

    ALLOWED_ACTIONS = {
        "initial_orchestration",
        "retrieve_missing_relationship",
        "increase_hop_depth",
        "retrieve_missing_entity",
        "broaden_retrieval",
    }

    def __init__(
        self,
        orchestrator: RetrievalOrchestrator,
        max_additional_iterations: int = 2,
    ) -> None:
        self.orchestrator = orchestrator
        self.max_additional_iterations = max_additional_iterations

    def run(
        self,
        query: str,
        reference_time: str | datetime | None = None,
        top_k: int = 10,
    ) -> QAEvidencePackage:
        """
        Execute bounded retrieval loop and return verified evidence package.

        Args:
            query: Question text.
            reference_time: Optional point-in-time cutoff.
            top_k: Target evidence count.

        Returns:
            QAEvidencePackage: Bounded, temporally valid evidence package.
        """
        trace: list[dict[str, Any]] = []
        evidence_pool: dict[str, RetrievalEvidence] = {}

        # Round 1: Initial Orchestration
        current_evidence, analysis, sufficiency = self.orchestrator.orchestrate(
            query=query,
            reference_time=reference_time,
            top_k=top_k,
        )

        for ev in current_evidence:
            evidence_pool[ev.evidence_id] = ev

        trace.append({
            "round": 1,
            "action": "initial_orchestration",
            "reason": "Initial query analysis and hybrid retrieval",
            "query": query,
            "reference_time": analysis.reference_time,
            "result_count": len(current_evidence),
        })

        additional_rounds = 0
        current_hops = analysis.expected_hop_depth

        # Iterative ReAct-style loop: only if evidence is insufficient and rounds remain
        while not sufficiency.sufficient and additional_rounds < self.max_additional_iterations:
            additional_rounds += 1
            round_num = additional_rounds + 1

            new_evidence: list[RetrievalEvidence] = []
            action: str
            reason: str

            # Formulate ONE controlled targeted retrieval action
            if sufficiency.missing_relationships:
                action = "retrieve_missing_relationship"
                target_rels = sufficiency.missing_relationships
                reason = f"Retrieve missing relationships: {target_rels}"
                # Targeted graph retrieval with relationship keywords
                targeted_query = f"{query} {' '.join(target_rels)}"
                new_evidence = self.orchestrator.hybrid_retriever.retrieve(
                    query=targeted_query,
                    reference_time=analysis.reference_time,
                    top_k=top_k,
                    max_hops=current_hops,
                )

            elif sufficiency.observed_hops < sufficiency.required_hops:
                action = "increase_hop_depth"
                current_hops = min(3, current_hops + 1)
                reason = f"Expand graph traversal depth to {current_hops} hops"
                new_evidence = self.orchestrator.hybrid_retriever.retrieve(
                    query=query,
                    reference_time=analysis.reference_time,
                    top_k=top_k,
                    max_hops=current_hops,
                )

            elif sufficiency.missing_entities:
                action = "retrieve_missing_entity"
                missing_ent = sufficiency.missing_entities[0]
                reason = f"Targeted search for missing entity: {missing_ent}"
                new_evidence = self.orchestrator.hybrid_retriever.retrieve(
                    query=f"{query} {missing_ent}",
                    reference_time=analysis.reference_time,
                    top_k=top_k,
                    max_hops=current_hops,
                )

            else:
                action = "broaden_retrieval"
                reason = "Broaden hybrid search candidate pool"
                new_evidence = self.orchestrator.hybrid_retriever.retrieve(
                    query=query,
                    reference_time=analysis.reference_time,
                    top_k=top_k * 2,
                    max_hops=current_hops,
                )

            assert action in self.ALLOWED_ACTIONS, f"Unauthorized agent action: {action}"

            # Strict temporal filtering on additional retrieval results
            if analysis.reference_time is not None:
                ref_dt = parse_datetime(analysis.reference_time)
                if ref_dt is not None:
                    new_evidence = [ev for ev in new_evidence if is_available(ev.available_time, ref_dt)]

            # Merge new evidence into pool
            for ev in new_evidence:
                evidence_pool[ev.evidence_id] = ev

            # Rerank cumulative evidence pool
            reranked_pool = self.orchestrator.reranker.rerank(
                query=query,
                candidates=list(evidence_pool.values()),
                top_k=top_k,
                reference_time=analysis.reference_time,
            )

            # Re-evaluate sufficiency
            sufficiency = self.orchestrator.sufficiency_checker.check(
                analysis=analysis,
                evidence=reranked_pool,
            )

            current_evidence = reranked_pool

            trace.append({
                "round": round_num,
                "action": action,
                "reason": reason,
                "query": query,
                "reference_time": analysis.reference_time,
                "result_count": len(new_evidence),
            })

        # Final Evidence Packaging
        graph_paths: list[list[str]] = []
        sources: list[str] = []

        # Final strict temporal validation and filtering across all returned evidence
        temporal_valid = True
        if analysis.reference_time is not None:
            ref_dt = parse_datetime(analysis.reference_time)
            if ref_dt is not None:
                sanitized_evidence = []
                for ev in current_evidence:
                    if not is_available(ev.available_time, ref_dt):
                        temporal_valid = False
                    else:
                        sanitized_evidence.append(ev)
                current_evidence = sanitized_evidence

        for ev in current_evidence:
            if "path" in ev.metadata and isinstance(ev.metadata["path"], list):
                if ev.metadata["path"] not in graph_paths:
                    graph_paths.append(ev.metadata["path"])
            if ev.document_id and ev.document_id not in sources:
                sources.append(ev.document_id)

        return QAEvidencePackage(
            query=query,
            reference_time=analysis.reference_time,
            question_type=analysis.question_type,
            expected_hops=analysis.expected_hop_depth,
            evidence=current_evidence,
            graph_paths=graph_paths,
            sources=sources,
            retrieval_rounds=len(trace),
            temporal_valid=temporal_valid,
            sufficient=sufficiency.sufficient,
            trace=trace,
            primary_evidence=current_evidence[:5],
            supporting_evidence=current_evidence[5:],
            sufficiency_result=sufficiency,
            retrieval_trace=trace,
            analysis=analysis,
        )


class AnswerGenerator:
    """
    Interface for final answer generation from QA evidence packages.

    Keeps retrieval evaluation separated from generative LLM experiments.
    """

    def generate(self, package: QAEvidencePackage) -> dict[str, Any]:
        """
        Generate structured answer placeholder from evidence package.
        """
        return {
            "answer": None,
            "evidence_package": package.to_dict(),
        }
