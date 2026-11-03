from __future__ import annotations

from datetime import datetime
from typing import Any

from src.qa.evidence_sufficiency import EvidenceSufficiencyChecker, SufficiencyResult
from src.qa.query_analysis import QueryAnalysis, QueryAnalyzer
from src.retrieval.evidence import RetrievalEvidence
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.reranker import BaseReranker, DeterministicReranker


class RetrievalOrchestrator:
    """
    Coordinates Query Analysis, Hybrid Retrieval, Deterministic Reranking,
    and Evidence Sufficiency Evaluation.
    """

    def __init__(
        self,
        hybrid_retriever: HybridRetriever,
        reranker: BaseReranker | None = None,
        query_analyzer: QueryAnalyzer | None = None,
        sufficiency_checker: EvidenceSufficiencyChecker | None = None,
    ) -> None:
        self.hybrid_retriever = hybrid_retriever
        self.reranker = reranker or DeterministicReranker()
        self.query_analyzer = query_analyzer or QueryAnalyzer(
            graph_retriever=hybrid_retriever.graph_retriever
        )
        self.sufficiency_checker = sufficiency_checker or EvidenceSufficiencyChecker()

    def orchestrate(
        self,
        query: str,
        reference_time: str | datetime | None = None,
        top_k: int = 10,
        max_hops: int | None = None,
        candidate_pool_size: int | None = None,
        balanced_fusion: bool = False,
        min_per_entity: int = 3,
    ) -> tuple[list[RetrievalEvidence], QueryAnalysis, SufficiencyResult]:
        """
        Execute coordinated retrieval pipeline.

        Args:
            query: Question text.
            reference_time: Optional explicit temporal cutoff override.
            top_k: Target number of final evidence records.
            max_hops: Optional override for graph traversal hop depth.
            candidate_pool_size: Optional override for candidate pool size entering reranker.
            balanced_fusion: Whether to apply entity-balanced evidence allocation for comparative queries.
            min_per_entity: Minimum candidate quota reserved per required entity.

        Returns:
            tuple[list[RetrievalEvidence], QueryAnalysis, SufficiencyResult]:
                Reranked evidence, query analysis, and sufficiency evaluation.
        """
        # 1. Query Analysis
        ref_override = reference_time.isoformat() if isinstance(reference_time, datetime) else reference_time
        analysis = self.query_analyzer.analyze(
            query=query,
            reference_time_override=ref_override,
        )

        effective_ref_time = analysis.reference_time
        effective_hops = max_hops if max_hops is not None else analysis.expected_hop_depth

        is_comparative = (
            analysis.question_type == "comparison"
            or len(analysis.entities) >= 2
            or analysis.cross_document_required
        )
        use_balanced = balanced_fusion and is_comparative

        # 2. Hybrid Retrieval with Temporal Cutoff
        pool_k = candidate_pool_size if candidate_pool_size is not None else max(top_k * 2, 25 if use_balanced else 20)
        initial_candidates = self.hybrid_retriever.retrieve(
            query=query,
            reference_time=effective_ref_time,
            top_k=pool_k,
            max_hops=effective_hops,
            candidate_fetch_k=max(pool_k * 2, 40 if use_balanced else 20),
            balanced_fusion=use_balanced,
            min_per_entity=min_per_entity,
            target_entities=analysis.entities,
        )

        # 3. Deterministic Reranking with Strict Temporal Enforcement
        reranked_evidence = self.reranker.rerank(
            query=query,
            candidates=initial_candidates,
            top_k=top_k,
            reference_time=effective_ref_time,
            balanced_reranking=use_balanced,
            target_entities=analysis.entities,
            min_per_entity=min_per_entity,
        )

        # 4. Evidence Sufficiency Assessment
        sufficiency = self.sufficiency_checker.check(
            analysis=analysis,
            evidence=reranked_evidence,
        )

        return reranked_evidence, analysis, sufficiency
