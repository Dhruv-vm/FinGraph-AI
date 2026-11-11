from __future__ import annotations

import argparse
import json
import os
import sys
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

# Ensure project root is on sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.qa.agent import RetrievalAgent, QAEvidencePackage
from src.qa.evidence_sufficiency import EvidenceSufficiencyChecker
from src.qa.generator import AnswerGenerator, QAAnswer
from src.qa.query_analysis import QueryAnalyzer
from src.qa.validation import AnswerValidator
from src.retrieval.graph import GraphRetriever
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.orchestrator import RetrievalOrchestrator
from src.retrieval.reranker import DeterministicReranker
from src.retrieval.vector import VectorStore

load_dotenv(".env")


class FinGraphPipeline:
    """
    FinGraph AI Full Framework.

    Integrates:
    - Semantic Temporal Knowledge Graph
    - Dense Vector Retrieval over SEC filings
    - Hybrid Retrieval with RRF fusion
    - Deterministic Reranking
    - Query Analysis (Hop-depth estimation, relationship intent)
    - Lightweight Bounded ReAct Retrieval Agent
    - Evidence Sufficiency Gate
    - Structured Answer Generator with AnswerValidator
    """

    def __init__(
        self,
        kg_path: Path | str | None = None,
        vector_path: Path | str | None = None,
        vector_store: VectorStore | None = None,
        model: str | None = None,
        temperature: float = 0.0,
        max_agent_iterations: int = 2,
        hop_mode: str = "strict",
        balanced_fusion: bool = False,
        min_per_entity: int = 3,
        regulatory_linking: bool = True,
    ) -> None:
        self.kg_path = Path(kg_path or (ROOT / "data" / "processed" / "unified" / "fingraph_semantic_kg.json"))
        self.vector_path = Path(vector_path or (ROOT / "data" / "vector_store"))
        self.hop_mode = hop_mode
        self.balanced_fusion = balanced_fusion
        self.min_per_entity = min_per_entity
        self.regulatory_linking = regulatory_linking

        self.graph_retriever = GraphRetriever(self.kg_path, regulatory_linking=self.regulatory_linking)
        if vector_store is not None:
            self.vector_store = vector_store
        else:
            self.vector_store = VectorStore(self.vector_path) if self.vector_path.exists() else None

        self.hybrid_retriever = HybridRetriever(
            vector_store=self.vector_store,
            graph_retriever=self.graph_retriever,
            rrf_k=60,
        )

        self.query_analyzer = QueryAnalyzer(graph_retriever=self.graph_retriever)
        self.reranker = DeterministicReranker()
        self.sufficiency_checker = EvidenceSufficiencyChecker(
            graph_retriever=self.graph_retriever,
            hop_mode=self.hop_mode,
        )

        self.orchestrator = RetrievalOrchestrator(
            hybrid_retriever=self.hybrid_retriever,
            query_analyzer=self.query_analyzer,
            reranker=self.reranker,
            sufficiency_checker=self.sufficiency_checker,
        )

        self.agent = RetrievalAgent(
            orchestrator=self.orchestrator,
            max_additional_iterations=max_agent_iterations,
        )

        self.validator = AnswerValidator()
        self.generator = AnswerGenerator(
            model=model or os.getenv("LLM_MODEL", "qwen3.5:4b"),
            temperature=temperature,
            validator=self.validator,
        )

    def answer(
        self,
        query: str,
        reference_time: str | datetime | None = None,
        top_k: int = 10,
        max_evidence_items: int | None = None,
        candidate_pool_size: int | None = None,
        mode: str = "benchmark",
    ) -> tuple[QAEvidencePackage, QAAnswer]:
        """
        Execute full FinGraph AI pipeline: query analysis, hybrid retrieval, agentic loop,
        evidence packaging, and structured answer generation.
        """
        t_start = time.perf_counter()
        ref_str = reference_time.isoformat() if isinstance(reference_time, datetime) else reference_time
        assist = mode == "assist"
        # Assist mode (live app only): a relationship may be evidenced by filing text, not only by a
        # typed graph edge. Benchmark mode keeps the evaluated Phase 7 configuration unchanged.
        # Callers must serialise calls (the API holds a lock) because the checker is shared.
        previous_rel_mode = self.sufficiency_checker.relationship_mode
        if assist:
            self.sufficiency_checker.relationship_mode = "textual"
            self.hybrid_retriever.issuer_scoped = True
            self.query_analyzer.period_dates_are_cutoffs = False
            # Include filings fetched on demand, and scope retrieval to the company the question
            # names even when it is outside the research corpus (no network call here).
            if self.vector_store is not None:
                self.vector_store.include_live = True
            try:
                from src.data.ingestion.live_sec import resolve_company

                company = resolve_company(query, allow_network=False)
            except Exception:
                company = None
            self.hybrid_retriever.extra_scoped_tickers = [company["ticker"]] if company else []
        try:
            package = self.agent.run(
                query=query,
                reference_time=ref_str,
                top_k=max(top_k, 12) if assist else top_k,
                candidate_pool_size=max(candidate_pool_size or 20, 40) if assist else candidate_pool_size,
                balanced_fusion=self.balanced_fusion,
                min_per_entity=self.min_per_entity,
            )
        finally:
            self.sufficiency_checker.relationship_mode = previous_rel_mode
            self.hybrid_retriever.issuer_scoped = False
            self.query_analyzer.period_dates_are_cutoffs = True
            self.hybrid_retriever.extra_scoped_tickers = []
            if self.vector_store is not None:
                self.vector_store.include_live = False
        t_gen = time.perf_counter()
        answer = self.generator.generate(package, max_evidence_items=max_evidence_items, assist=assist)
        t_done = time.perf_counter()
        logging.getLogger("fingraph.pipeline").info(
            "Timing (%s): retrieval %.1fs over %d round(s), generation %.1fs",
            mode, t_gen - t_start, package.retrieval_rounds, t_done - t_gen,
        )
        return package, answer


def main() -> None:
    parser = argparse.ArgumentParser(description="Run FinGraph AI Full Framework")
    parser.add_argument("--query", "-q", type=str, default="What companies does NVIDIA depend on?")
    parser.add_argument("--reference-time", "-t", type=str, default=None)
    args = parser.parse_args()

    print("=== FinGraph AI Full Framework ===")
    print(f"Query: {args.query}")
    if args.reference_time:
        print(f"Reference Time: {args.reference_time}")

    pipeline = FinGraphPipeline()
    t0 = time.perf_counter()
    pkg, ans = pipeline.answer(query=args.query, reference_time=args.reference_time)
    t1 = time.perf_counter()

    print(f"\nExecution Time: {t1 - t0:.2f}s")
    print(f"Hops Estimated: {pkg.expected_hops} (Multi-hop: {pkg.expected_hops > 1})")
    print(f"Rounds Executed: {pkg.retrieval_rounds}")
    print(f"Sufficient Evidence: {pkg.sufficient}")
    print(f"Evidence Items: {pkg.evidence_count}")
    print(f"\nAnswer: {ans.answer}")
    print(f"Confidence: {ans.confidence}")
    print(f"Evidence IDs Cited: {ans.evidence_ids}")
    print(f"Validation Status: {ans.validation_status}")


if __name__ == "__main__":
    main()
