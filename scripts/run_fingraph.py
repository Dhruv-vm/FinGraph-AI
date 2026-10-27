from __future__ import annotations

import argparse
import json
import os
import sys
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
    ) -> None:
        self.kg_path = Path(kg_path or (ROOT / "data" / "processed" / "unified" / "fingraph_semantic_kg.json"))
        self.vector_path = Path(vector_path or (ROOT / "data" / "vector_store"))

        self.graph_retriever = GraphRetriever(self.kg_path)
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
        self.sufficiency_checker = EvidenceSufficiencyChecker(graph_retriever=self.graph_retriever)

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
    ) -> tuple[QAEvidencePackage, QAAnswer]:
        """
        Execute full FinGraph AI pipeline: query analysis, hybrid retrieval, agentic loop,
        evidence packaging, and structured answer generation.
        """
        ref_str = reference_time.isoformat() if isinstance(reference_time, datetime) else reference_time
        package = self.agent.run(query=query, reference_time=ref_str)
        answer = self.generator.generate(package)
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
