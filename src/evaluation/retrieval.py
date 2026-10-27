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

        # 3. Multi-hop path accuracy: verify relation type and directionality
        path_acc = 0.0
        if expected_graph_paths and is_answerable:
            ret_paths = retrieved_graph_paths or []
            # Check if all required edges in expected path exist in retrieved paths
            all_paths_matched = True
            for exp_edge in expected_graph_paths:
                # exp_edge is [source, relation, target]
                matched = False
                for rp in ret_paths:
                    # rp could be a list of node/rel strings or formatted path
                    rp_str = " ".join(rp).upper() if isinstance(rp, list) else str(rp).upper()
                    if len(exp_edge) == 3:
                        s, r, t = exp_edge
                        # Must match relation and entity keywords
                        if r.upper() in rp_str:
                            matched = True
                            break
                    elif len(exp_edge) == 2:
                        s, r = exp_edge
                        if r.upper() in rp_str:
                            matched = True
                            break
                if not matched:
                    all_paths_matched = False
                    break
            path_acc = 1.0 if all_paths_matched else 0.0

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
