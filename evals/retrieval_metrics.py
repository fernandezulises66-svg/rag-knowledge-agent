"""Pure retrieval evaluation metrics: relevance, Hit@K, and reciprocal rank.

These functions only compare `RetrievalResult` objects against expected
`RetrievalTarget` locations. They never call an embedding service, an
LLM, or the network, and never apply a similarity-score threshold.
"""

from evals.retrieval_cases import RetrievalTarget
from rag.retriever import RetrievalResult


def is_relevant(result: RetrievalResult, targets: tuple[RetrievalTarget, ...]) -> bool:
    """Whether a retrieved result matches one of the expected targets.

    A target with a section requires an exact source and section
    match. A target with `section=None` matches any section from that
    source.
    """
    for target in targets:
        if result.chunk.source != target.source:
            continue
        if target.section is None or result.chunk.section == target.section:
            return True
    return False


def hit_at_k(
    results: list[RetrievalResult],
    targets: tuple[RetrievalTarget, ...],
    k: int,
) -> float:
    """1.0 if a relevant result appears in the first k results, else 0.0.

    Raises:
        ValueError: If `k` is not greater than zero.
    """
    if k <= 0:
        raise ValueError("k must be greater than zero")

    return 1.0 if any(is_relevant(result, targets) for result in results[:k]) else 0.0


def reciprocal_rank(
    results: list[RetrievalResult],
    targets: tuple[RetrievalTarget, ...],
    max_rank: int = 5,
) -> float:
    """1/rank for the first relevant result within max_rank, else 0.0.

    Raises:
        ValueError: If `max_rank` is not greater than zero.
    """
    if max_rank <= 0:
        raise ValueError("max_rank must be greater than zero")

    for rank, result in enumerate(results[:max_rank], start=1):
        if is_relevant(result, targets):
            return 1.0 / rank
    return 0.0
