from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Callable

import ollama
from dotenv import load_dotenv

from src.data.graph.snapshot import is_available, parse_datetime
from src.qa.agent import QAEvidencePackage
from src.qa.validation import AnswerValidator, ValidationResult
from src.retrieval.evidence import RetrievalEvidence

load_dotenv(".env")


@dataclass
class QAAnswer:
    """Structured, evidence-grounded answer with preserved provenance."""

    answer: str
    confidence: float
    evidence_ids: list[str] = field(default_factory=list)
    citations: list[dict[str, Any]] = field(default_factory=list)
    temporal_reference: str | None = None
    insufficient_evidence: bool = False
    reasoning_summary: str | None = None
    evidence: list[dict[str, Any]] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    validation_status: dict[str, Any] | None = None
    evidence_package: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def __contains__(self, key: str) -> bool:
        return hasattr(self, key)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)


SYSTEM_PROMPT = """You are a financial question-answering system.

Answer the user's question using ONLY the supplied evidence.
The evidence has already passed temporal validation.

Rules:
1. Do not use outside knowledge.
2. Do not invent facts.
3. Do not add unsupported relationships.
4. Do not use information not present in the evidence.
5. If the evidence is insufficient, say: "Insufficient evidence in the retrieved corpus."
6. Respect the reference time. Never mention facts that became available after the reference time.
7. Every factual claim must cite one or more evidence IDs.
8. Return ONLY valid JSON with keys: "answer", "confidence", "evidence_ids", "reasoning_summary".
9. Do not include markdown code blocks, reasoning chains, or commentary.
"""


class AnswerGenerator:
    """
    Evidence-grounded generative financial QA layer using local Ollama (Qwen3.5:4B).

    Ensures:
    - Pre-generation temporal assertion (zero future evidence reaches LLM).
    - Insufficient evidence refusal without hallucination.
    - Deterministic JSON output with temperature = 0.
    - Post-generation citation and provenance validation.
    """

    def __init__(
        self,
        model: str | None = None,
        host: str | None = None,
        temperature: float = 0.0,
        validator: AnswerValidator | None = None,
        llm_client: Any | None = None,
    ) -> None:
        self.model = model or os.getenv("LLM_MODEL", "qwen3.5:4b")
        self.host = host or os.getenv("OLLAMA_HOST", "http://localhost:11434")
        self.temperature = temperature
        self.validator = validator or AnswerValidator()
        self.llm_client = llm_client

    def _serialize_evidence(self, evidence: list[RetrievalEvidence]) -> list[dict[str, Any]]:
        """Deterministically serialize evidence items for prompt injection."""
        serialized = []
        for ev in evidence:
            raw_text = ev.text.strip() if ev.text else ""
            # Truncate text snippet to avoid blowing context window and triggering unterminated JSON
            snippet = raw_text[:220].strip()
            if len(raw_text) > 220:
                snippet += "..."
            item = {
                "evidence_id": ev.evidence_id,
                "source_type": ev.source_type,
                "text": snippet,
                "relationship": ev.metadata.get("relationship", "N/A"),
                "hop_count": int(ev.metadata.get("hop_count", 1)),
                "confidence": round(float(ev.metadata.get("confidence", ev.original_scores.get("graph", ev.score))), 4),
                "available_time": ev.available_time or "N/A",
                "document_id": ev.document_id,
                "chunk_id": ev.chunk_id,
            }
            if "path" in ev.metadata and isinstance(ev.metadata["path"], list):
                item["graph_path"] = ev.metadata["path"]
            serialized.append(item)
        return serialized

    def generate(self, package: QAEvidencePackage) -> QAAnswer:
        """
        Generate evidence-grounded structured QAAnswer from QAEvidencePackage.

        Args:
            package: Consolidated, point-in-time filtered QAEvidencePackage.

        Returns:
            QAAnswer: Structured answer with verified citations and provenance.
        """
        # 1. HARD TEMPORAL SAFETY: Assert zero future evidence reaches generation
        ref_dt = None
        if package.reference_time:
            ref_dt = parse_datetime(package.reference_time)
            if ref_dt is not None:
                future_items = [
                    ev for ev in package.all_evidence
                    if not is_available(ev.available_time, ref_dt)
                ]
                if future_items:
                    raise ValueError(
                        f"Temporal safety violation: {len(future_items)} future evidence items "
                        f"present in package (reference_time={package.reference_time}). "
                        "Future evidence must NEVER reach the generator."
                    )

        # 2. INSUFFICIENT EVIDENCE REFUSAL: Never fabricate an answer
        if not package.sufficient or len(package.all_evidence) == 0:
            return QAAnswer(
                answer="Insufficient evidence in the retrieved corpus.",
                confidence=0.0,
                evidence_ids=[],
                citations=[],
                temporal_reference=package.reference_time,
                insufficient_evidence=True,
                reasoning_summary="Retrieved evidence does not satisfy query entity, relationship, or reasoning depth constraints.",
                evidence=[ev.to_dict() for ev in package.all_evidence],
                sources=package.sources,
                validation_status={"is_valid": True, "errors": []},
                evidence_package=package.to_dict(),
            )

        # 3. Deterministically serialize candidate evidence
        # Top 5 items to maintain focused context window and ensure valid completion
        serialized_evidence = self._serialize_evidence(package.all_evidence[:5])

        user_prompt = (
            f"Question:\n{package.query}\n\n"
            f"Reference time:\n{package.reference_time or 'No temporal restriction'}\n\n"
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

        # 4. Invoke Ollama model
        raw_response: str
        try:
            if self.llm_client is not None:
                # Custom/mock callable for unit testing
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
            # Safe failure fallback: structured error result, never fabricated answer
            return QAAnswer(
                answer=f"Generation failed due to LLM backend error: {exc}",
                confidence=0.0,
                evidence_ids=[],
                citations=[],
                temporal_reference=package.reference_time,
                insufficient_evidence=True,
                reasoning_summary=f"Ollama execution failed: {exc}",
                evidence=[ev.to_dict() for ev in package.all_evidence],
                sources=package.sources,
                validation_status={"is_valid": False, "errors": [str(exc)]},
                evidence_package=package.to_dict(),
            )

        # 5. Parse and validate LLM output
        val_result = self.validator.validate(raw_response, package)

        parsed_json: dict[str, Any]
        try:
            parsed_json = json.loads(raw_response)
        except Exception:
            parsed_json = {
                "answer": "Generation produced malformed non-JSON output.",
                "confidence": 0.0,
                "evidence_ids": [],
                "reasoning_summary": "Malformed JSON output.",
            }

        answer_text = parsed_json.get("answer", "")
        confidence_val = float(parsed_json.get("confidence", 0.0))
        evidence_ids_cited = [str(x) for x in parsed_json.get("evidence_ids", [])]
        reasoning_sum = parsed_json.get("reasoning_summary", "")

        # 6. Assemble provenance for cited evidence
        evidence_map = {ev.evidence_id: ev for ev in package.all_evidence}
        citations: list[dict[str, Any]] = []
        for eid in evidence_ids_cited:
            if eid in evidence_map:
                ev_obj = evidence_map[eid]
                citations.append({
                    "evidence_id": ev_obj.evidence_id,
                    "document_id": ev_obj.document_id,
                    "chunk_id": ev_obj.chunk_id,
                    "available_time": ev_obj.available_time,
                    "source_type": ev_obj.source_type,
                    "confidence": float(ev_obj.metadata.get("confidence", ev_obj.original_scores.get("graph", ev_obj.score))),
                    "text_snippet": ev_obj.text[:150] if ev_obj.text else "",
                })

        return QAAnswer(
            answer=answer_text,
            confidence=confidence_val,
            evidence_ids=evidence_ids_cited,
            citations=citations,
            temporal_reference=package.reference_time,
            insufficient_evidence=not val_result.is_valid and len(evidence_ids_cited) == 0,
            reasoning_summary=reasoning_sum,
            evidence=[ev.to_dict() for ev in package.all_evidence],
            sources=package.sources,
            validation_status=val_result.to_dict(),
            evidence_package=package.to_dict(),
        )
