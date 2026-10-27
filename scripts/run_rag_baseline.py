from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import ollama
from dotenv import load_dotenv

# Ensure project root is on sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.graph.snapshot import parse_datetime
from src.qa.agent import QAEvidencePackage
from src.qa.generator import QAAnswer, SYSTEM_PROMPT
from src.qa.validation import AnswerValidator
from src.retrieval.evidence import RetrievalEvidence
from src.retrieval.vector import RetrievedChunk, VectorStore

load_dotenv(".env")


class VectorOnlyRAG:
    """
    Standard Vector-Only RAG Baseline (B1).

    Scientific Isolation Properties:
    - Pure dense semantic retrieval over Qdrant chunk embeddings (MiniLM).
    - ZERO graph retrieval: GraphRetriever, HybridRetriever, and graph traversal are NEVER called.
    - ZERO graph reranking: DeterministicReranker graph degree weighting is NEVER applied.
    - ZERO agentic loop: Single-shot retrieval; no ReAct loop or iterative follow-ups.
    - Identical temporal point-in-time filtering: applied at the vector level via Qdrant date filter.
    - Identical generation conditions: local Ollama qwen3.5:4b, temperature=0.0, top-5 context budget.
    """

    def __init__(
        self,
        vector_store: VectorStore | None = None,
        model: str | None = None,
        host: str | None = None,
        temperature: float = 0.0,
        top_k: int = 5,
        llm_client: Callable[[str, str], str] | None = None,
    ) -> None:
        self.vector_store = vector_store or VectorStore(ROOT / "data" / "vector_store")
        self.model = model or os.getenv("LLM_MODEL", "qwen3.5:4b")
        self.host = host or os.getenv("OLLAMA_HOST", "http://localhost:11434")
        self.temperature = temperature
        self.top_k = top_k
        self.validator = AnswerValidator()
        self.llm_client = llm_client

    def retrieve(
        self,
        query: str,
        reference_time: str | datetime | None = None,
        top_k: int | None = None,
    ) -> list[RetrievalEvidence]:
        """
        Pure dense vector retrieval with optional point-in-time cutoff.
        """
        k = top_k or self.top_k
        ref_dt = parse_datetime(reference_time) if isinstance(reference_time, str) else reference_time

        raw_chunks: list[RetrievedChunk] = self.vector_store.search(
            query=query,
            top_k=k,
            reference_time=ref_dt,
        )

        evidence_items = []
        for rank, chunk in enumerate(raw_chunks, 1):
            payload = chunk.payload
            doc_id = payload.get("document_id", "")
            avail_time = payload.get("available_time", "")
            ev = RetrievalEvidence(
                evidence_id=f"vector:{chunk.chunk_id}",
                source_type="vector",
                text=chunk.text,
                score=chunk.score,
                document_id=doc_id,
                chunk_id=chunk.chunk_id,
                available_time=avail_time,
                metadata={
                    "vector_score": chunk.score,
                    "company": payload.get("company"),
                    "publication_date": payload.get("publication_date"),
                    "hop_count": 1,
                },
                original_scores={"vector": chunk.score},
                rank=rank,
            )
            evidence_items.append(ev)

        return evidence_items

    def answer(
        self,
        query: str,
        reference_time: str | datetime | None = None,
    ) -> QAAnswer:
        """
        Execute vector-only retrieval and evidence-grounded answer generation.
        """
        ref_str = reference_time.isoformat() if isinstance(reference_time, datetime) else reference_time
        evidence = self.retrieve(query=query, reference_time=reference_time)

        # Build QAEvidencePackage representation for validation consistency
        doc_ids = sorted(list({ev.document_id for ev in evidence if ev.document_id}))
        package = QAEvidencePackage(
            query=query,
            reference_time=ref_str,
            question_type="factual",
            expected_hops=1,
            evidence=evidence,
            graph_paths=[],  # Vector-only baseline has ZERO graph paths
            sources=doc_ids,
            retrieval_rounds=1,
            temporal_valid=True,
            sufficient=len(evidence) > 0,
            primary_evidence=evidence[:self.top_k],
            supporting_evidence=[],
        )

        # Refusal if no evidence was retrieved
        if len(evidence) == 0:
            return QAAnswer(
                answer="Insufficient evidence in the retrieved corpus.",
                confidence=0.0,
                evidence_ids=[],
                citations=[],
                temporal_reference=ref_str,
                insufficient_evidence=True,
                reasoning_summary="No vector chunks satisfied query and temporal cutoff.",
                evidence=[],
                sources=[],
                validation_status={"is_valid": True, "errors": []},
                evidence_package=package.to_dict(),
            )

        # Format evidence for prompt injection (identical context format)
        serialized_evidence = []
        for ev in evidence[:self.top_k]:
            snippet = ev.text.strip()[:220]
            if len(ev.text.strip()) > 220:
                snippet += "..."
            serialized_evidence.append({
                "evidence_id": ev.evidence_id,
                "source_type": ev.source_type,
                "text": snippet,
                "available_time": ev.available_time or "N/A",
                "document_id": ev.document_id,
                "chunk_id": ev.chunk_id,
            })

        user_prompt = (
            f"Question:\n{query}\n\n"
            f"Reference time:\n{ref_str or 'No temporal restriction'}\n\n"
            f"Evidence:\n{json.dumps(serialized_evidence, indent=2)}\n\n"
            "Instructions:\n"
            "- Be concise (2-3 sentences max).\n"
            "- Answer directly using the evidence items.\n"
            "- Output valid JSON strictly adhering to schema:\n"
            "{\n"
            '  "answer": "...",\n'
            '  "confidence": 0.0,\n'
            '  "evidence_ids": ["..."],\n'
            '  "reasoning_summary": "..."\n'
            "}"
        )

        raw_response: str
        try:
            if self.llm_client is not None:
                raw_response = self.llm_client(user_prompt, SYSTEM_PROMPT)
            else:
                client = ollama.Client(host=self.host)
                chat_res = client.chat(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    format="json",
                    options={
                        "temperature": self.temperature,
                        "num_predict": 2048,
                    },
                    think=False,
                )
                raw_response = chat_res.message.content
        except Exception as exc:
            return QAAnswer(
                answer=f"Generation failed due to LLM error: {exc}",
                confidence=0.0,
                evidence_ids=[],
                citations=[],
                temporal_reference=ref_str,
                insufficient_evidence=True,
                reasoning_summary=str(exc),
                evidence=[ev.to_dict() for ev in evidence],
                sources=doc_ids,
                validation_status={"is_valid": False, "errors": [str(exc)]},
                evidence_package=package.to_dict(),
            )

        # Parse output
        try:
            parsed = json.loads(raw_response)
        except Exception:
            parsed = {
                "answer": "Malformed non-JSON output from LLM.",
                "confidence": 0.0,
                "evidence_ids": [],
                "reasoning_summary": "JSON decode failure.",
            }

        answer_text = parsed.get("answer", "")
        conf = float(parsed.get("confidence", 0.0))
        eids = [str(x) for x in parsed.get("evidence_ids", [])]
        reasoning = parsed.get("reasoning_summary", "")

        val_result = self.validator.validate(parsed, package)

        ev_map = {ev.evidence_id: ev for ev in evidence}
        citations = []
        for eid in eids:
            if eid in ev_map:
                ev_obj = ev_map[eid]
                citations.append({
                    "evidence_id": ev_obj.evidence_id,
                    "document_id": ev_obj.document_id,
                    "chunk_id": ev_obj.chunk_id,
                    "available_time": ev_obj.available_time,
                    "source_type": ev_obj.source_type,
                    "text_snippet": ev_obj.text[:150] if ev_obj.text else "",
                })

        return QAAnswer(
            answer=answer_text,
            confidence=conf,
            evidence_ids=eids,
            citations=citations,
            temporal_reference=ref_str,
            insufficient_evidence="insufficient evidence" in answer_text.lower(),
            reasoning_summary=reasoning,
            evidence=[ev.to_dict() for ev in evidence],
            sources=doc_ids,
            validation_status=val_result.to_dict(),
            evidence_package=package.to_dict(),
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Vector-Only RAG Baseline (B1)")
    parser.add_argument("--query", "-q", type=str, default="What products does Apple manufacture?")
    parser.add_argument("--reference-time", "-t", type=str, default=None)
    parser.add_argument("--top-k", "-k", type=int, default=5)
    args = parser.parse_args()

    print(f"=== Vector-Only RAG Baseline (B1) ===")
    print(f"Query: {args.query}")
    if args.reference_time:
        print(f"Reference Time: {args.reference_time}")

    rag = VectorOnlyRAG(top_k=args.top_k)
    t0 = time.perf_counter()
    ans = rag.answer(query=args.query, reference_time=args.reference_time)
    t1 = time.perf_counter()

    print(f"\nExecution Time: {t1 - t0:.2f}s")
    print(f"Answer: {ans.answer}")
    print(f"Confidence: {ans.confidence}")
    print(f"Evidence Cited: {ans.evidence_ids}")
    print(f"Citations: {len(ans.citations)}")


if __name__ == "__main__":
    main()
