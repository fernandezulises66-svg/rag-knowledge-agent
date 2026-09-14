"""Runs retrieval evaluation cases against an already constructed
`SemanticRetriever` and aggregates Hit@1, Hit@3, and MRR@5 over
answerable cases only.

Unsupported diagnostic cases retain their retrieved results for
inspection but are intentionally excluded from every metric, since
there is no correct evidence chunk to score them against.
"""

from dataclasses import dataclass

from evals.retrieval_cases import RetrievalEvalCase
from evals.retrieval_metrics import hit_at_k, reciprocal_rank
from rag.retriever import RetrievalResult, SemanticRetriever


@dataclass(frozen=True)
class RetrievalEvalResult:
    """Per-case retrieval evaluation outcome.

    For unsupported cases, `hit_at_1`, `hit_at_3`, and `reciprocal_rank`
    are `None` rather than a faked value, since there is intentionally
    no correct evidence chunk to score against.
    """

    case: RetrievalEvalCase
    results: tuple[RetrievalResult, ...]
    hit_at_1: float | None
    hit_at_3: float | None
    reciprocal_rank: float | None


@dataclass(frozen=True)
class RetrievalEvalSummary:
    """Aggregate retrieval evaluation metrics over answerable cases only."""

    total_cases: int
    answerable_cases: int
    unsupported_cases: int
    hit_at_1: float
    hit_at_3: float
    mrr_at_5: float


def evaluate_retrieval(
    retriever: SemanticRetriever,
    cases: list[RetrievalEvalCase],
    top_k: int = 5,
) -> tuple[list[RetrievalEvalResult], RetrievalEvalSummary]:
    """Run every case through `retriever.retrieve()` and score answerable ones.

    Args:
        retriever: An already constructed retriever over the embedded
            knowledge base.
        cases: Evaluation cases, evaluated in order.
        top_k: Results requested per case. Must be at least 5, since
            MRR@5 requires 5 ranks.

    Returns:
        Per-case results, in case order, and an aggregate summary
        computed over answerable cases only.

    Raises:
        ValueError: If `top_k` is less than 5.
    """
    if top_k < 5:
        raise ValueError("top_k must be at least 5 to compute MRR@5")

    eval_results: list[RetrievalEvalResult] = []

    for case in cases:
        results = retriever.retrieve(case.question, top_k=top_k)

        if case.answerable:
            case_hit_at_1 = hit_at_k(results, case.relevant_targets, k=1)
            case_hit_at_3 = hit_at_k(results, case.relevant_targets, k=3)
            case_rr = reciprocal_rank(results, case.relevant_targets, max_rank=5)
        else:
            case_hit_at_1 = None
            case_hit_at_3 = None
            case_rr = None

        eval_results.append(
            RetrievalEvalResult(
                case=case,
                results=tuple(results),
                hit_at_1=case_hit_at_1,
                hit_at_3=case_hit_at_3,
                reciprocal_rank=case_rr,
            )
        )

    answerable_results = [result for result in eval_results if result.case.answerable]
    answerable_count = len(answerable_results)

    if answerable_count > 0:
        mean_hit_at_1 = sum(result.hit_at_1 for result in answerable_results) / answerable_count
        mean_hit_at_3 = sum(result.hit_at_3 for result in answerable_results) / answerable_count
        mean_rr = (
            sum(result.reciprocal_rank for result in answerable_results) / answerable_count
        )
    else:
        mean_hit_at_1 = 0.0
        mean_hit_at_3 = 0.0
        mean_rr = 0.0

    summary = RetrievalEvalSummary(
        total_cases=len(cases),
        answerable_cases=answerable_count,
        unsupported_cases=len(cases) - answerable_count,
        hit_at_1=mean_hit_at_1,
        hit_at_3=mean_hit_at_3,
        mrr_at_5=mean_rr,
    )

    return eval_results, summary
