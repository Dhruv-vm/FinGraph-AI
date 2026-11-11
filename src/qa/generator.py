from __future__ import annotations

import json
import logging
import os
import re
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

# Live "assist" mode: same grounding rules, but the model may give a partial answer
# instead of a blanket refusal when the evidence covers only part of the question.
ASSIST_SYSTEM_PROMPT = """You are a financial research assistant answering questions about SEC filings.

Answer using ONLY the supplied evidence passages.

Rules:
1. Never use outside knowledge and never invent facts, figures, dates or relationships.
2. Read every passage fully: figures often appear inside tables or later in a passage.
3. If the evidence answers the question, answer it directly and include the specific figures, names and periods it states.
4. If the evidence answers only part of the question, answer that part and say plainly what the filings provided do not cover.
5. Only if nothing in the evidence is relevant, reply exactly: "Insufficient evidence in the retrieved corpus."
6. Respect the reference time. Never mention facts that became available after it.
7. For "latest" or "most recent" questions, use the passage with the latest "period_end" (or filing date) that contains the figure.
8. Each passage names the company whose filing it comes from ("source_company"). Never attribute a statement, figure or plan from one company's filing to a different company.
9. Every factual claim must cite one or more evidence IDs.
10. Set "coverage" to "full" if the evidence answers the question, "partial" if it answers only part of it, or "none" if it does not answer it at all (then cite nothing).
11. Check the question's premise against the evidence. If the evidence shows it is false (for example, nobody owns a
    majority, i.e. more than 50%), say so plainly, then give what the evidence does show.
12. In 10-Q tables, rows usually list the three-month (quarter) figures first: current year, prior year, then the
    six- or nine-month year-to-date figures. Answer a "quarter" question with the three-month figure, and say which period it is.
13. Keep "reasoning_summary" to one short sentence.
14. Return ONLY valid JSON with keys: "answer", "coverage", "confidence", "evidence_ids", "reasoning_summary".
"""

_NEGATIVE = re.compile(
    r"\b(?:do|does|did)\s+not\s+(?:explicitly\s+)?(?:contain|provide|include|state|mention|list|identify|report|specify|disclose)"
    r"|\bno\s+(?:information|data|figures?|mention|evidence)\b|\bnot\s+(?:available|found|present)\s+in\b",
    re.I,
)
_PIVOT = re.compile(r"\b(?:however|but|only|instead|although|while)\b", re.I)


def _assist_coverage(parsed: dict[str, Any], answer_text: str) -> str:
    """'full', 'partial' or 'none': the model's own declaration, checked against its wording."""
    declared = str(parsed.get("coverage", "")).strip().lower()
    negative = bool(_NEGATIVE.search(answer_text))
    if declared in ("full", "partial", "none"):
        if declared == "full" and negative:
            return "partial" if _PIVOT.search(answer_text) else "none"
        return declared
    if negative:
        return "partial" if _PIVOT.search(answer_text) else "none"
    return "full"


def _parse_json_loose(raw: str) -> dict[str, Any] | None:
    """Parse model output as JSON, falling back to the outermost {...} block."""
    try:
        v = json.loads(raw)
        return v if isinstance(v, dict) else None
    except Exception:
        pass
    a, b = raw.find("{"), raw.rfind("}")
    if a != -1 and b > a:
        try:
            v = json.loads(raw[a : b + 1])
            if isinstance(v, dict):
                return v
        except Exception:
            pass
    # Truncated output: salvage the fields that did complete.
    ans = re.search(r'"answer"\s*:\s*"((?:[^"\\]|\\.)*)"', raw)
    if not ans:
        return None
    out: dict[str, Any] = {"answer": json.loads(f'"{ans.group(1)}"')}
    cov = re.search(r'"coverage"\s*:\s*"(full|partial|none)"', raw)
    if cov:
        out["coverage"] = cov.group(1)
    conf = re.search(r'"confidence"\s*:\s*([0-9.]+)', raw)
    if conf:
        out["confidence"] = float(conf.group(1))
    ids = re.search(r'"evidence_ids"\s*:\s*\[(.*?)\]', raw, re.S)
    out["evidence_ids"] = re.findall(r'"([^"]+)"', ids.group(1)) if ids else []
    out["reasoning_summary"] = ""
    return out


_WINDOW_STOP = {
    "what", "which", "when", "where", "does", "did", "the", "and", "for", "with", "from", "that", "this",
    "their", "they", "have", "has", "was", "were", "are", "its", "about", "into", "most", "recent", "company",
    "filings", "filing", "report", "reported", "say", "said", "how", "who", "whom", "much", "many",
}


def _relevant_window(text: str, query: str, width: int) -> str:
    """Return the `width`-character window of `text` that mentions the most query terms.

    Used only in assist mode so a figure deep inside a long passage (e.g. a table row)
    reaches the model without sending the whole passage.
    """
    if len(text) <= width:
        return text
    terms = {w for w in re.findall(r"[a-z0-9]+", query.lower()) if len(w) >= 3 and w not in _WINDOW_STOP}
    if not terms:
        return text[:width].strip() + "..."
    low = text.lower()
    best_start, best_score = 0, -1
    for start in range(0, max(1, len(text) - width) + 1, 80):
        window = low[start : start + width]
        score = sum(window.count(t) for t in terms)
        if score > best_score:
            best_start, best_score = start, score
    snippet = text[best_start : best_start + width].strip()
    return ("..." if best_start > 0 else "") + snippet + ("..." if best_start + width < len(text) else "")


GENERAL_SYSTEM_PROMPT = """You are FinGraph AI's assistant. FinGraph answers questions from SEC filings, but the
filings it searched did not answer this question, so you are answering from general knowledge instead.

Rules:
1. Be helpful and direct. For greetings or questions about yourself, reply briefly and mention you can answer
   questions about public companies' SEC filings (for example acquisitions, risks, revenue, competitors).
2. For general questions, answer concisely (at most 4 sentences) from widely known facts.
3. Do not state specific recent financial figures (revenue, earnings, prices, dates of recent events) as fact:
   your knowledge may be outdated. Give context instead and suggest checking the company's SEC filings.
4. Never predict stock prices or give investment advice; explain briefly why instead.
5. If you do not know, say so plainly.
6. Return ONLY valid JSON: {"answer": "..."}
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
        max_evidence_items: int = 5,
    ) -> None:
        self.model = model or os.getenv("LLM_MODEL", "qwen3.5:4b")
        self.host = host or os.getenv("OLLAMA_HOST", "http://localhost:11434")
        self.temperature = temperature
        self.validator = validator or AnswerValidator()
        self.llm_client = llm_client
        self.max_evidence_items = max_evidence_items

    def _serialize_evidence(
        self, evidence: list[RetrievalEvidence], snippet_chars: int = 220, query: str | None = None
    ) -> list[dict[str, Any]]:
        """Deterministically serialize evidence items for prompt injection."""
        serialized = []
        for ev in evidence:
            raw_text = ev.text.strip() if ev.text else ""
            if query is not None:
                # Assist mode: the most query-relevant window of the passage. Passages matched to a
                # filing section (e.g. the ownership table) get a longer window centred on that
                # section's own wording, so tables keep their percentages and totals.
                topical = ev.metadata.get("topical_match")
                if isinstance(topical, str) and topical:
                    snippet = _relevant_window(raw_text, f"{query} {topical} %", max(snippet_chars, 1300))
                else:
                    snippet = _relevant_window(raw_text, query, snippet_chars)
            else:
                # Truncate text snippet to avoid blowing context window and triggering unterminated JSON
                snippet = raw_text[:snippet_chars].strip()
                if len(raw_text) > snippet_chars:
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
            if query is not None:
                # Assist mode: a compact record (fits Ollama's default context, leaving room for the
                # answer) that also tells the model whose filing each passage is from.
                compact = {
                    "evidence_id": ev.evidence_id,
                    "source_company": getattr(ev, "source_company", None)
                    or ev.metadata.get("source_company")
                    or ev.metadata.get("company")
                    or "unknown",
                    "text": snippet,
                }
                if ev.metadata.get("document_type"):
                    compact["filing"] = f'{ev.metadata.get("document_type")} {ev.metadata.get("publication_date") or ""}'.strip()
                if ev.metadata.get("fiscal_period"):
                    compact["period_end"] = ev.metadata["fiscal_period"]
                if ev.metadata.get("relationship"):
                    compact["relationship"] = ev.metadata["relationship"]
                if ev.available_time:
                    compact["available_time"] = ev.available_time
                item = compact
            serialized.append(item)
        return serialized

    def generate(
        self,
        package: QAEvidencePackage,
        max_evidence_items: int | None = None,
        assist: bool = False,
    ) -> QAAnswer:
        """
        Generate evidence-grounded structured QAAnswer from QAEvidencePackage.

        Args:
            package: Consolidated, point-in-time filtered QAEvidencePackage.
            max_evidence_items: Optional override for the number of evidence items injected.

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
        # Benchmark mode keeps the Phase 7 settings (5 items x 220 chars). Assist mode lets the
        # model read most of each passage, since figures usually sit beyond the first 220 chars.
        # (Sizes are chosen to stay within Ollama's default context so the model stays on the GPU.)
        evidence_limit = max_evidence_items or (6 if assist else self.max_evidence_items)
        snippet_chars = 700 if assist else 220
        system_prompt = ASSIST_SYSTEM_PROMPT if assist else SYSTEM_PROMPT
        serialized_evidence = self._serialize_evidence(
            package.all_evidence[:evidence_limit], snippet_chars, query=package.query if assist else None
        )

        user_prompt = (
            f"Question:\n{package.query}\n\n"
            f"Reference time:\n{package.reference_time or 'No temporal restriction'}\n\n"
            f"Evidence:\n{json.dumps(serialized_evidence, indent=None if assist else 2)}\n\n"
            "Instructions:\n"
            + ("- Be concise but complete (up to 4 sentences); include the figures and periods stated.\n" if assist else "- Be concise (2-3 sentences max).\n")
            + 
            "- Answer directly using the evidence items.\n"
            "- Output valid JSON strictly adhering to schema:\n"
            "{\n"
            '  "answer": "...",\n'
            + ('  "coverage": "full | partial | none",\n' if assist else "")
            + '  "confidence": 0.0,\n'
            '  "evidence_ids": ["..."],\n'
            '  "reasoning_summary": "..."\n'
            "}"
        )

        # 4. Invoke Ollama model
        raw_response: str
        try:
            if self.llm_client is not None:
                # Custom/mock callable for unit testing
                raw_response = self.llm_client(user_prompt, system_prompt)
            else:
                client = ollama.Client(host=self.host)
                chat_res = client.chat(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    format="json",
                    options={
                        "temperature": self.temperature,
                        # Assist answers are a few sentences; a tighter cap bounds worst-case latency
                        # (truncated JSON is salvaged). Benchmark mode keeps its original limit.
                        "num_predict": 600 if assist else 2048,
                    },
                    think=False,
                    # Keep the model in memory between questions (latency only; answers are unchanged).
                    keep_alive="30m",
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
        if assist and _parse_json_loose(raw_response) is None and self.llm_client is None:
            # One retry for malformed JSON (longer assist answers occasionally break the format).
            try:
                retry = ollama.Client(host=self.host).chat(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt + "\n\nReturn one compact, valid JSON object only."},
                    ],
                    format="json",
                    options={"temperature": self.temperature, "num_predict": 2048},
                    think=False,
                    keep_alive="30m",
                )
                raw_response = retry.message.content
            except Exception:
                pass
        if assist:
            loose = _parse_json_loose(raw_response)
            if loose is not None:
                raw_response = json.dumps(loose)
            else:
                logging.getLogger("fingraph.generator").warning(
                    "Assist generation returned non-JSON output (%d chars): %r", len(raw_response or ""), (raw_response or "")[:600]
                )
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

        insufficient = not val_result.is_valid and len(evidence_ids_cited) == 0
        validation_dict = val_result.to_dict()
        if assist:
            coverage = "none" if not citations else _assist_coverage(parsed_json, answer_text)
            validation_dict["coverage"] = coverage
            if coverage == "none":
                # The model is saying the filings do not answer the question: report a refusal
                # (keeping its explanation), never a supported answer.
                insufficient = True
                citations = []
                evidence_ids_cited = []
                if not answer_text.strip():
                    answer_text = "Insufficient evidence in the retrieved corpus."
            elif coverage == "partial":
                validation_dict.setdefault("warnings", [])
                validation_dict["warnings"] = list(validation_dict["warnings"]) + [
                    "Partial answer: the retrieved filings cover only part of the question."
                ]

        return QAAnswer(
            answer=answer_text,
            confidence=0.0 if (assist and not citations) else confidence_val,
            evidence_ids=evidence_ids_cited,
            citations=citations,
            temporal_reference=package.reference_time,
            insufficient_evidence=insufficient,
            reasoning_summary=reasoning_sum,
            evidence=[ev.to_dict() for ev in package.all_evidence],
            sources=package.sources,
            validation_status=validation_dict,
            evidence_package=package.to_dict(),
        )

    def general_answer(self, query: str, filings_note: str | None = None) -> str | None:
        """Ungrounded fallback answer (live assist mode only). Returns None if the model is unavailable.

        The caller must present this as general knowledge, not as an evidence-backed answer.
        """
        prompt = f"Question:\n{query}\n"
        if filings_note:
            prompt += f"\nWhat the SEC filing search found:\n{filings_note}\n"
        prompt += '\nReturn {"answer": "..."} only.'
        try:
            if self.llm_client is not None:
                raw = self.llm_client(prompt, GENERAL_SYSTEM_PROMPT)
            else:
                res = ollama.Client(host=self.host).chat(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": GENERAL_SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    format="json",
                    options={"temperature": 0.2, "num_predict": 400},
                    think=False,
                    keep_alive="30m",
                )
                raw = res.message.content
        except Exception:
            return None
        parsed = _parse_json_loose(raw or "")
        text = str((parsed or {}).get("answer") or "").strip()
        return text or None

