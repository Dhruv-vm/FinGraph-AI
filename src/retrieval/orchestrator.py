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
    ) -> tuple[list[RetrievalEvidence], QueryAnalysis, SufficiencyResult]:
        """
        Execute coordinated retrieval pipeline.

        Args:
            query: Question text.
            reference_time: Optional explicit temporal cutoff override.
            top_k: Target number of final evidence records.
            max_hops: Optional override for graph traversal hop depth.

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

        # 2. Hybrid Retrieval with Temporal Cutoff
        initial_candidates = self.hybrid_retriever.retrieve(
            query=query,
            reference_time=effective_ref_time,
            top_k=max(top_k * 2, 20),
            max_hops=effective_hops,
        )

        # 3. Deterministic Reranking with Strict Temporal Enforcement
        reranked_evidence = self.reranker.rerank(
            query=query,
            candidates=initial_candidates,
            top_k=top_k,
            reference_time=effective_ref_time,
        )

        # 4. Evidence Sufficiency Assessment
        sufficiency = self.sufficiency_checker.check(
            analysis=analysis,
            evidence=reranked_evidence,
        )

        return reranked_evidence, analysis, sufficiency
