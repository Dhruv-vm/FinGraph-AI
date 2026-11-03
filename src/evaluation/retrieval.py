from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Sequence


@dataclass
class SingleRetrievalResult:
    """Metrics for an individual question retrieval."""

    question_id: str
    hit_at_k: dict[int, float]
    precision_at_k: dict[int, float]
    recall_at_k: dict[int, float]
    mrr: float
    ndcg_at_k: dict[int, float]
    doc_hit_at_k: dict[int, float]
    doc_recall_at_k: dict[int, float]
    multi_hop_path_accuracy: float = 0.0
    cross_doc_coverage: float = 0.0
    is_answerable: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AggregatedRetrievalMetrics:
    """Aggregated retrieval metrics across a benchmark suite."""

    total_questions: int
    evaluated_questions: int  # Answerable questions with non-empty relevance sets
    mean_hit_at_k: dict[int, float]
    mean_precision_at_k: dict[int, float]
    mean_recall_at_k: dict[int, float]
    mean_mrr: float
    mean_ndcg_at_k: dict[int, float]
    mean_doc_hit_at_k: dict[int, float]
    mean_doc_recall_at_k: dict[int, float]
    multi_hop_path_accuracy: float
    cross_doc_coverage: float
    per_question_results: list[SingleRetrievalResult] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_questions": self.total_questions,
            "evaluated_questions": self.evaluated_questions,
            "mean_hit_at_k": {str(k): round(v, 4) for k, v in self.mean_hit_at_k.items()},
            "mean_precision_at_k": {str(k): round(v, 4) for k, v in self.mean_precision_at_k.items()},
            "mean_recall_at_k": {str(k): round(v, 4) for k, v in self.mean_recall_at_k.items()},
            "mean_mrr": round(self.mean_mrr, 4),
            "mean_ndcg_at_k": {str(k): round(v, 4) for k, v in self.mean_ndcg_at_k.items()},
            "mean_doc_hit_at_k": {str(k): round(v, 4) for k, v in self.mean_doc_hit_at_k.items()},
            "mean_doc_recall_at_k": {str(k): round(v, 4) for k, v in self.mean_doc_recall_at_k.items()},
            "multi_hop_path_accuracy": round(self.multi_hop_path_accuracy, 4),
            "cross_doc_coverage": round(self.cross_doc_coverage, 4),
        }


def _normalize_entity_token(s: Any) -> str:
    """Normalize entity token for robust identifier comparison."""
    cleaned = str(s).strip().lower().replace("_", "").replace(" ", "").replace(":", "").replace("-", "")
    return cleaned


def _edge_matches(ret_e: Any, exp_e: list[Any] | tuple[Any, ...]) -> bool:
    """
    Check if a retrieved directed edge matches an expected directed edge [source, relationship, target].
    Enforces strict directionality (source -> target) and canonical relationship match.
    Rejects reversed edges where source and target are swapped.
    """
    if isinstance(ret_e, dict):
        s = ret_e.get("source", "")
        r = ret_e.get("relationship", "")
        t = ret_e.get("target", "")
    elif isinstance(ret_e, (list, tuple)) and len(ret_e) >= 3:
        s, r, t = ret_e[0], ret_e[1], ret_e[2]
    elif isinstance(ret_e, (list, tuple)) and len(ret_e) == 2:
        s, r, t = ret_e[0], "", ret_e[1]
    else:
        return False

    exp_s = exp_e[0]
    exp_r = exp_e[1] if len(exp_e) >= 2 else ""
    exp_t = exp_e[2] if len(exp_e) >= 3 else None

    # Relationship must match (case-insensitive)
    if exp_r and str(r).upper() != str(exp_r).upper():
        return False

    s_norm = _normalize_entity_token(s)
    exp_s_norm = _normalize_entity_token(exp_s)
    t_norm = _normalize_entity_token(t) if t is not None else ""
    exp_t_norm = _normalize_entity_token(exp_t) if exp_t is not None else ""

    # Check reversed edge: if source matches expected target and target matches expected source
    if exp_t is not None and s_norm and exp_t_norm and t_norm and exp_s_norm:
        if (s_norm == exp_t_norm or exp_t_norm in s_norm) and (t_norm == exp_s_norm or exp_s_norm in t_norm):
            return False  # Strict failure: reversed edge!

    # Source check (directed)
    s_matched = bool(s_norm and exp_s_norm and (s_norm == exp_s_norm or exp_s_norm in s_norm or s_norm in exp_s_norm))
    if not s_matched:
        return False

    # Target check (directed)
    if exp_t is not None:
        t_matched = bool(t_norm and exp_t_norm and (t_norm == exp_t_norm or exp_t_norm in t_norm or t_norm in exp_t_norm))
        if not t_matched:
            return False

    return True


def _candidate_path_matches(ret_path: Any, exp_path: list[list[Any]]) -> bool:
    """
    Evaluate whether a candidate retrieved path satisfies the expected directed multi-hop path.
    For single-edge expected paths: checks if any edge in ret_path matches.
    For multi-hop expected paths: checks that all expected hops are present in connected sequence.
    """
    if not ret_path or not exp_path:
        return False

    # Normalize candidate path into list of edge representations
    candidate_edges: list[Any] = []
    # If ret_path is directly a single edge [source, rel, target]
    if (
        isinstance(ret_path, (list, tuple))
        and len(ret_path) in (2, 3)
        and isinstance(ret_path[0], str)
        and isinstance(ret_path[1], str)
    ):
        candidate_edges = [ret_path]
    elif isinstance(ret_path, list):
        for item in ret_path:
            if isinstance(item, dict) and item.get("source") and item.get("relationship"):
                candidate_edges.append(item)
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                candidate_edges.append(item)

    if not candidate_edges:
        return False

    if len(exp_path) == 1:
        return any(_edge_matches(e, exp_path[0]) for e in candidate_edges)

    # Multi-hop path: each expected edge in exp_path must be matched by an edge in candidate_edges,
    # and consecutive matched edges must be connected (target of hop i connects to source of hop i+1).
    matched_edges = []
    for exp_e in exp_path:
        found_edge = None
        for cand_e in candidate_edges:
            if _edge_matches(cand_e, exp_e):
                found_edge = cand_e
                break
        if not found_edge:
            return False
        matched_edges.append(found_edge)

    # Check connection between consecutive edges in the path
    for i in range(len(matched_edges) - 1):
        e1 = matched_edges[i]
        e2 = matched_edges[i + 1]
        t1 = e1.get("target") if isinstance(e1, dict) else e1[2]
        s2 = e2.get("source") if isinstance(e2, dict) else e2[0]
        t1_norm = _normalize_entity_token(t1)
        s2_norm = _normalize_entity_token(s2)
        if not (t1_norm == s2_norm or t1_norm in s2_norm or s2_norm in t1_norm):
            return False

    return True


class RetrievalEvaluator:
    """
    Evaluates chunk-level and document-level retrieval accuracy, multi-hop path accuracy,
    and cross-document fact coverage.
    """

    DEFAULT_K_VALUES = (1, 3, 5, 10)

    def __init__(self, k_values: Sequence[int] = DEFAULT_K_VALUES) -> None:
        self.k_values = tuple(sorted(k_values))

    def evaluate_single(
        self,
        question_id: str,
        retrieved_chunk_ids: list[str],
        relevant_chunk_ids: list[str],
        retrieved_doc_ids: list[str] | None = None,
        relevant_doc_ids: list[str] | None = None,
        retrieved_graph_paths: list[list[str]] | None = None,
        expected_graph_paths: list[list[str]] | None = None,
        is_answerable: bool = True,
    ) -> SingleRetrievalResult:
        """
        Evaluate retrieval metrics for a single question.

        If relevant_chunk_ids is empty (unanswerable question), retrieval Recall/MRR
        are handled gracefully (marked 0.0 or not counted in recall denominators).
        """
        hit_at_k = {}
        prec_at_k = {}
        rec_at_k = {}
        ndcg_at_k = {}
        doc_hit_at_k = {}
        doc_rec_at_k = {}

        rel_chunk_set = set(relevant_chunk_ids)
        rel_doc_set = set(relevant_doc_ids or [])
        ret_docs = retrieved_doc_ids or []

        # 1. Chunk-level metrics
        if not rel_chunk_set:
            # Unanswerable question
            for k in self.k_values:
                hit_at_k[k] = 0.0
                prec_at_k[k] = 0.0
                rec_at_k[k] = 0.0
                ndcg_at_k[k] = 0.0
            mrr = 0.0
        else:
            # Compute Hit, Precision, Recall, NDCG at each K
            for k in self.k_values:
                top_chunks = retrieved_chunk_ids[:k]
                hits = sum(1 for cid in top_chunks if cid in rel_chunk_set)
                hit_at_k[k] = 1.0 if hits > 0 else 0.0
                prec_at_k[k] = hits / k if k > 0 else 0.0
                rec_at_k[k] = hits / len(rel_chunk_set)

                # DCG@K
                dcg = 0.0
                for rank, cid in enumerate(top_chunks, 1):
                    if cid in rel_chunk_set:
                        dcg += 1.0 / math.log2(rank + 1)
                # IDCG@K
                idcg = sum(1.0 / math.log2(r + 1) for r in range(1, min(k, len(rel_chunk_set)) + 1))
                ndcg_at_k[k] = (dcg / idcg) if idcg > 0 else 0.0

            # MRR
            mrr = 0.0
            for rank, cid in enumerate(retrieved_chunk_ids, 1):
                if cid in rel_chunk_set:
                    mrr = 1.0 / rank
                    break

        # 2. Document-level metrics
        if not rel_doc_set:
            for k in self.k_values:
                doc_hit_at_k[k] = 0.0
                doc_rec_at_k[k] = 0.0
        else:
            for k in self.k_values:
                top_docs = ret_docs[:k]
                doc_hits = sum(1 for d in top_docs if d in rel_doc_set)
                doc_hit_at_k[k] = 1.0 if doc_hits > 0 else 0.0
                doc_rec_at_k[k] = doc_hits / len(rel_doc_set)

        # 3. Multi-hop path accuracy: verify relation type, sequence, and directionality
        path_acc = 0.0
        if expected_graph_paths and is_answerable:
            ret_paths = retrieved_graph_paths or []
            # Check whether ANY candidate retrieved path satisfies the expected directed path
            for cand_path in ret_paths:
                if _candidate_path_matches(cand_path, expected_graph_paths):
                    path_acc = 1.0
                    break

        # 4. Cross-document coverage
        cross_doc_cov = 0.0
        if rel_doc_set and len(rel_doc_set) > 1 and is_answerable:
            ret_doc_set = set(ret_docs)
            covered = len(rel_doc_set.intersection(ret_doc_set))
            cross_doc_cov = covered / len(rel_doc_set)

        return SingleRetrievalResult(
            question_id=question_id,
            hit_at_k=hit_at_k,
            precision_at_k=prec_at_k,
            recall_at_k=rec_at_k,
            mrr=mrr,
            ndcg_at_k=ndcg_at_k,
            doc_hit_at_k=doc_hit_at_k,
            doc_recall_at_k=doc_rec_at_k,
            multi_hop_path_accuracy=path_acc,
            cross_doc_coverage=cross_doc_cov,
            is_answerable=is_answerable,
        )

    def evaluate_suite(
        self,
        results: list[SingleRetrievalResult],
    ) -> AggregatedRetrievalMetrics:
        """
        Aggregate retrieval results across all evaluated questions.
        Evaluates answerable questions for retrieval metrics.
        """
        answerable_results = [r for r in results if r.is_answerable]
        n_eval = len(answerable_results)
        total_q = len(results)

        if n_eval == 0:
            return AggregatedRetrievalMetrics(
                total_questions=total_q,
                evaluated_questions=0,
                mean_hit_at_k={k: 0.0 for k in self.k_values},
                mean_precision_at_k={k: 0.0 for k in self.k_values},
                mean_recall_at_k={k: 0.0 for k in self.k_values},
                mean_mrr=0.0,
                mean_ndcg_at_k={k: 0.0 for k in self.k_values},
                mean_doc_hit_at_k={k: 0.0 for k in self.k_values},
                mean_doc_recall_at_k={k: 0.0 for k in self.k_values},
                multi_hop_path_accuracy=0.0,
                cross_doc_coverage=0.0,
                per_question_results=results,
            )

        mean_hit = {k: sum(r.hit_at_k.get(k, 0.0) for r in answerable_results) / n_eval for k in self.k_values}
        mean_prec = {k: sum(r.precision_at_k.get(k, 0.0) for r in answerable_results) / n_eval for k in self.k_values}
        mean_rec = {k: sum(r.recall_at_k.get(k, 0.0) for r in answerable_results) / n_eval for k in self.k_values}
        mean_ndcg = {k: sum(r.ndcg_at_k.get(k, 0.0) for r in answerable_results) / n_eval for k in self.k_values}
        mean_mrr = sum(r.mrr for r in answerable_results) / n_eval

        mean_doc_hit = {k: sum(r.doc_hit_at_k.get(k, 0.0) for r in answerable_results) / n_eval for k in self.k_values}
        mean_doc_rec = {k: sum(r.doc_recall_at_k.get(k, 0.0) for r in answerable_results) / n_eval for k in self.k_values}

        # Multi-hop path accuracy: denominator is answerable multi-hop questions
        mh_results = [r for r in answerable_results if r.multi_hop_path_accuracy > 0.0 or "RMH" in r.question_id]
        mean_path_acc = (sum(r.multi_hop_path_accuracy for r in mh_results) / len(mh_results)) if mh_results else 0.0

        # Cross-document coverage: denominator is cross-doc questions
        cd_results = [r for r in answerable_results if "CDC" in r.question_id or r.cross_doc_coverage > 0.0]
        mean_cross_doc = (sum(r.cross_doc_coverage for r in cd_results) / len(cd_results)) if cd_results else 0.0

        return AggregatedRetrievalMetrics(
            total_questions=total_q,
            evaluated_questions=n_eval,
            mean_hit_at_k=mean_hit,
            mean_precision_at_k=mean_prec,
            mean_recall_at_k=mean_rec,
            mean_mrr=mean_mrr,
            mean_ndcg_at_k=mean_ndcg,
            mean_doc_hit_at_k=mean_doc_hit,
            mean_doc_recall_at_k=mean_doc_rec,
            multi_hop_path_accuracy=mean_path_acc,
            cross_doc_coverage=mean_cross_doc,
            per_question_results=results,
        )
