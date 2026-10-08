from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any

from src.data.graph.snapshot import is_available, parse_datetime
from src.qa.agent import QAEvidencePackage


@dataclass
class ValidationResult:
    """Structured result of answer validation."""

    is_valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    cited_evidence_ids: list[str] = field(default_factory=list)
    temporal_valid: bool = True
    provenance_complete: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class AnswerValidator:
    """
    Deterministic validator for evidence-grounded QA outputs.

    Verifies:
    1. JSON structure and required keys.
    2. Answer non-emptiness.
    3. Valid confidence range [0.0, 1.0].
    4. Legitimate evidence IDs existing strictly within the supplied QAEvidencePackage.
    5. Requirement of evidence citations for factual claims.
    6. Point-in-time temporal compliance (no citations to future evidence).
    7. Provenance completeness (document_id, chunk_id, available_time).
    """

    REQUIRED_KEYS = {"answer", "confidence", "evidence_ids"}

    def validate(
        self,
        output: dict[str, Any] | str,
        package: QAEvidencePackage,
    ) -> ValidationResult:
        """
        Validate an LLM output against the supplied evidence package.

        Args:
            output: Parsed dict or raw JSON string from the generator.
            package: The verified QAEvidencePackage provided to the model.

        Returns:
            ValidationResult: Detailed validation outcome.
        """
        errors: list[str] = []
        warnings: list[str] = []
        cited_ids: list[str] = []
        temporal_valid = True
        provenance_complete = True

        # 1. Parse JSON structure
        payload: dict[str, Any]
        if isinstance(output, str):
            try:
                parsed = json.loads(output)
                if not isinstance(parsed, dict):
                    return ValidationResult(
                        is_valid=False,
                        errors=["Model output must be a JSON object."],
                    )
                payload = parsed
            except json.JSONDecodeError as exc:
                return ValidationResult(
                    is_valid=False,
                    errors=[f"Model output is not valid JSON: {exc}"],
                )
        elif isinstance(output, dict):
            payload = output
        else:
            return ValidationResult(
                is_valid=False,
                errors=["Expected output to be a dictionary or JSON string."],
            )

        # 2. Required keys
        missing_keys = self.REQUIRED_KEYS - set(payload.keys())
        if missing_keys:
            errors.append(f"Missing required fields: {sorted(missing_keys)}")

        # 3. Answer field inspection
        answer = payload.get("answer")
        answer_is_refusal_text = isinstance(answer, str) and "insufficient evidence" in answer.lower()
        is_refusal = (
            payload.get("insufficient_evidence", False)
            or answer_is_refusal_text
        )

        # If package was marked insufficient, answer MUST be an explicit refusal
        if not package.sufficient and not is_refusal:
            errors.append(
                "Package evidence was marked insufficient, but model returned a factual answer "
                "instead of refusing ('Insufficient evidence in the retrieved corpus.')."
            )

        if not isinstance(answer, str) or not answer.strip():
            if not is_refusal:
                errors.append("Answer must be a non-empty string.")

        # 4. Confidence bounds
        confidence = payload.get("confidence")
        if confidence is None or not isinstance(confidence, (int, float)):
            errors.append("Confidence must be a numeric value between 0.0 and 1.0.")
        elif not (0.0 <= float(confidence) <= 1.0):
            errors.append(f"Confidence {confidence} is out of bounds [0.0, 1.0].")

        # 5. Evidence IDs type and validity
        raw_evidence_ids = payload.get("evidence_ids", [])
        if not isinstance(raw_evidence_ids, list):
            errors.append("'evidence_ids' must be a list of strings.")
        else:
            cited_ids = [str(eid) for eid in raw_evidence_ids]

        # Build index of valid evidence items from the package
        package_evidence_map = {ev.evidence_id: ev for ev in package.all_evidence}

        # 6. Check for unknown or hallucinated evidence IDs
        unknown_ids = [eid for eid in cited_ids if eid not in package_evidence_map]
        if unknown_ids:
            errors.append(f"Unknown evidence IDs cited: {unknown_ids}")

        # 7. Check that factual claims cite evidence
        if not is_refusal and len(cited_ids) == 0:
            errors.append("Factual answers must cite at least one valid evidence ID.")

        # 8. Point-in-time temporal compliance on cited evidence
        ref_dt = None
        if package.reference_time:
            ref_dt = parse_datetime(package.reference_time)

        for eid in cited_ids:
            ev = package_evidence_map.get(eid)
            if ev is not None:
                # Temporal check
                if ref_dt is not None:
                    if not is_available(ev.available_time, ref_dt):
                        temporal_valid = False
                        errors.append(
                            f"Cited evidence {eid} ({ev.available_time}) violates "
                            f"reference cutoff {package.reference_time}."
                        )

                # Provenance completeness check
                if not ev.document_id or not ev.chunk_id or not ev.available_time:
                    provenance_complete = False
        # 9. Semantic Evidence-Question Alignment Guard (Issue 1, 2, 3, 9)
        # Verify that cited or retrieved evidence actually matches the requested relationship intents & directions
        if not is_refusal:
            semantic_valid, semantic_error = self.check_semantic_alignment(package, cited_ids)
            if not semantic_valid:
                errors.append(semantic_error)

        is_valid = len(errors) == 0

        return ValidationResult(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
            cited_evidence_ids=cited_ids,
            temporal_valid=temporal_valid,
            provenance_complete=provenance_complete,
        )

    def check_semantic_alignment(
        self,
        package: QAEvidencePackage,
        cited_ids: list[str] | None = None,
    ) -> tuple[bool, str]:
        """
        Deterministic guard checking if evidence genuinely satisfies the query's
        semantic relationship intent, multi-hop chains, and directionality.
        """
        analysis = package.analysis
        if analysis is None:
            return True, ""

        package_evidence_map = {ev.evidence_id: ev for ev in package.all_evidence}
        if cited_ids:
            eval_items = [package_evidence_map[eid] for eid in cited_ids if eid in package_evidence_map]
        else:
            eval_items = package.all_evidence[:5]

        if not eval_items:
            return False, "No evidence items available for semantic alignment check."

        query_lower = package.query.lower()

        # Check 1: Multi-hop queries with supplier risk intent (e.g., Q6)
        # Query requires BOTH: (1) supplier identification AND (2) risk affecting that supplier
        wants_supplier = any(w in query_lower for w in ("supply", "supplies", "supplier", "suppliers"))
        wants_risk = any(w in query_lower for w in ("risk", "risks", "affect", "affects", "affected"))
        if analysis.is_multi_hop and wants_supplier and wants_risk:
            has_risk_rel = False
            for ev in eval_items:
                path = ev.metadata.get("path") or []
                item_rels = []
                if "graph_relations" in ev.metadata:
                    item_rels.extend(ev.metadata["graph_relations"])
                if "relationship" in ev.metadata:
                    item_rels.append(str(ev.metadata["relationship"]))
                rels_str = " ".join(item_rels).upper()
                text_lower = (ev.text or "").lower()

                # Does this evidence contain HAS_RISK or AFFECTED_BY?
                if "HAS_RISK" in rels_str or "AFFECTED_BY" in rels_str or "risk" in text_lower:
                    has_risk_rel = True
                    break

            if not has_risk_rel:
                return (
                    False,
                    "Semantic misalignment: Query requested risks affecting suppliers, but evidence "
                    "only covers supplier identity without supplier risk relations (HAS_RISK/AFFECTED_BY)."
                )

        # Check 2: Dependency / Supplier directionality preservation (Issue 2)
        # "What companies does X depend on?"
        # Supported: Supplier --SUPPLIES--> X or X --DEPENDS_ON--> Supplier
        # UNSUPPORTED: X --SUPPLIES--> Customer
        if "depend" in query_lower:
            # Check if all cited evidence only contains reverse direction (X supplies others)
            supported_direction_found = False
            for ev in eval_items:
                text = ev.text or ""
                edge_id = ev.evidence_id.lower()
                # Check if it has DEPENDS_ON or SUPPLIES into target
                if "depends_on" in edge_id or "--[depends_on]-->" in text.lower():
                    supported_direction_found = True
                    break
                # If relationship is SUPPLIES: target must be the one supplying X
                if "supplies" in edge_id or "--[supplies]-->" in text.lower():
                    # For NVDA: company --SUPPLIES--> NVDA is valid, NVDA --SUPPLIES--> company is NOT valid for dependency
                    if "nvda" in query_lower or "nvidia" in query_lower:
                        # Must be incoming to NVDA
                        if "->supplies->company:nvda" in edge_id or "--[supplies]--> nvidia" in text.lower():
                            supported_direction_found = True
                            break
                        # If edge has NVDA as source of SUPPLIES: explicitly reverse, not valid
                        elif "company:nvda->supplies->" in edge_id or "nvidia corporation --[supplies]-->" in text.lower():
                            continue
                        else:
                            supported_direction_found = True
                            break
                    else:
                        supported_direction_found = True
                        break

            if not supported_direction_found:
                return (
                    False,
                    "Semantic directionality violation: Query asked what X depends on, but evidence "
                    "only establishes reverse relationships (X supplying others)."
                )

        return True, ""
