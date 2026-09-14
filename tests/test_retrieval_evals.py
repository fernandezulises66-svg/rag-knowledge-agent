"""Tests for the evals package (retrieval_cases, retrieval_metrics,
retrieval_runner).

Uses synthetic RetrievalResult objects and a fake retriever exposing
`.retrieve(query, top_k)`. Never uses OpenAI, network access, or real
embeddings.
"""

import dataclasses

import pytest

from evals.retrieval_cases import RETRIEVAL_EVAL_CASES, RetrievalEvalCase, RetrievalTarget
from evals.retrieval_metrics import hit_at_k, is_relevant, reciprocal_rank
from evals.retrieval_runner import RetrievalEvalResult, evaluate_retrieval
from rag.chunker import KnowledgeChunk, chunk_knowledge_base
from rag.document_loader import load_knowledge_base
from rag.retriever import RetrievalResult


# ---------------------------------------------------------------------------
# Fakes and helpers
# ---------------------------------------------------------------------------


class FakeRetriever:
    """Deterministic fake retriever: no embedding service, no network."""

    def __init__(
        self,
        results_by_query: dict[str, list[RetrievalResult]] | None = None,
        default_results: list[RetrievalResult] | None = None,
    ) -> None:
        self._results_by_query = results_by_query or {}
        self._default_results = default_results if default_results is not None else []
        self.calls: list[tuple[str, int]] = []

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        self.calls.append((query, top_k))
        return self._results_by_query.get(query, self._default_results)


def make_chunk(
    chunk_id: str = "doc.md::chunk-001",
    source: str = "doc.md",
    title: str = "Título",
    section: str | None = "Sección",
    text: str = "Texto de ejemplo.",
) -> KnowledgeChunk:
    return KnowledgeChunk(
        chunk_id=chunk_id, source=source, title=title, section=section, text=text
    )


def make_result(score: float, **chunk_kwargs) -> RetrievalResult:
    return RetrievalResult(chunk=make_chunk(**chunk_kwargs), score=score)


# ---------------------------------------------------------------------------
# RetrievalTarget / RetrievalEvalCase
# ---------------------------------------------------------------------------


def test_retrieval_target_has_expected_fields() -> None:
    target = RetrievalTarget(source="doc.md", section="Sección")

    assert target.source == "doc.md"
    assert target.section == "Sección"


def test_retrieval_target_is_immutable() -> None:
    target = RetrievalTarget(source="doc.md", section="Sección")

    with pytest.raises(dataclasses.FrozenInstanceError):
        target.source = "other.md"


def test_retrieval_eval_case_has_expected_fields() -> None:
    targets = (RetrievalTarget(source="doc.md", section=None),)
    case = RetrievalEvalCase(
        case_id="case-1", question="¿Pregunta?", answerable=True, relevant_targets=targets
    )

    assert case.case_id == "case-1"
    assert case.question == "¿Pregunta?"
    assert case.answerable is True
    assert case.relevant_targets == targets


def test_retrieval_eval_case_is_immutable() -> None:
    case = RetrievalEvalCase(case_id="c", question="q", answerable=True, relevant_targets=())

    with pytest.raises(dataclasses.FrozenInstanceError):
        case.answerable = False


def test_real_dataset_case_ids_are_unique() -> None:
    case_ids = [case.case_id for case in RETRIEVAL_EVAL_CASES]

    assert len(case_ids) == len(set(case_ids))


def test_real_dataset_answerable_cases_have_relevant_targets() -> None:
    for case in RETRIEVAL_EVAL_CASES:
        if case.answerable:
            assert len(case.relevant_targets) >= 1, case.case_id


def test_real_dataset_unsupported_cases_have_no_relevant_targets() -> None:
    for case in RETRIEVAL_EVAL_CASES:
        if not case.answerable:
            assert case.relevant_targets == (), case.case_id


def test_real_dataset_has_at_least_15_answerable_cases() -> None:
    answerable = [case for case in RETRIEVAL_EVAL_CASES if case.answerable]

    assert len(answerable) >= 15


def test_real_dataset_has_exactly_4_unsupported_cases() -> None:
    unsupported = [case for case in RETRIEVAL_EVAL_CASES if not case.answerable]

    assert len(unsupported) == 4


def test_real_dataset_includes_at_least_3_english_questions() -> None:
    # Explicit known English case IDs, per dataset design -- no language
    # detector needed.
    english_case_ids = {
        "payment-methods",
        "password-reset-link-validity",
        "support-response-time-professional",
    }
    dataset_ids = {case.case_id for case in RETRIEVAL_EVAL_CASES}

    assert english_case_ids.issubset(dataset_ids)
    assert len(english_case_ids) >= 3


# ---------------------------------------------------------------------------
# is_relevant
# ---------------------------------------------------------------------------


def test_is_relevant_exact_source_and_section_match() -> None:
    result = make_result(0.9, source="doc.md", section="Sección A")
    targets = (RetrievalTarget(source="doc.md", section="Sección A"),)

    assert is_relevant(result, targets) is True


def test_is_relevant_correct_source_wrong_section_is_not_relevant() -> None:
    result = make_result(0.9, source="doc.md", section="Sección B")
    targets = (RetrievalTarget(source="doc.md", section="Sección A"),)

    assert is_relevant(result, targets) is False


def test_is_relevant_wrong_source_is_not_relevant() -> None:
    result = make_result(0.9, source="other.md", section="Sección A")
    targets = (RetrievalTarget(source="doc.md", section="Sección A"),)

    assert is_relevant(result, targets) is False


def test_is_relevant_target_with_none_section_matches_any_section() -> None:
    result = make_result(0.9, source="doc.md", section="Cualquier sección")
    targets = (RetrievalTarget(source="doc.md", section=None),)

    assert is_relevant(result, targets) is True


def test_is_relevant_multiple_targets_any_match_counts() -> None:
    result = make_result(0.9, source="doc.md", section="Sección B")
    targets = (
        RetrievalTarget(source="other.md", section="X"),
        RetrievalTarget(source="doc.md", section="Sección B"),
    )

    assert is_relevant(result, targets) is True


# ---------------------------------------------------------------------------
# hit_at_k
# ---------------------------------------------------------------------------


def test_hit_at_k_relevant_at_rank_1_gives_one() -> None:
    results = [make_result(0.9, source="doc.md", section="A")]
    targets = (RetrievalTarget(source="doc.md", section="A"),)

    assert hit_at_k(results, targets, k=1) == 1.0


def test_hit_at_k_relevant_at_rank_2_gives_hit1_zero_and_hit3_one() -> None:
    results = [
        make_result(0.9, source="other.md", section="X"),
        make_result(0.8, source="doc.md", section="A"),
    ]
    targets = (RetrievalTarget(source="doc.md", section="A"),)

    assert hit_at_k(results, targets, k=1) == 0.0
    assert hit_at_k(results, targets, k=3) == 1.0


def test_hit_at_k_no_relevant_item_gives_zero() -> None:
    results = [make_result(0.9, source="other.md", section="X")]
    targets = (RetrievalTarget(source="doc.md", section="A"),)

    assert hit_at_k(results, targets, k=3) == 0.0


def test_hit_at_k_invalid_k_raises_value_error() -> None:
    with pytest.raises(ValueError):
        hit_at_k([], (), k=0)


# ---------------------------------------------------------------------------
# reciprocal_rank
# ---------------------------------------------------------------------------


def test_reciprocal_rank_rank_1_returns_one() -> None:
    results = [make_result(0.9, source="doc.md", section="A")]
    targets = (RetrievalTarget(source="doc.md", section="A"),)

    assert reciprocal_rank(results, targets) == pytest.approx(1.0)


def test_reciprocal_rank_rank_2_returns_one_half() -> None:
    results = [
        make_result(0.9, source="other.md", section="X"),
        make_result(0.8, source="doc.md", section="A"),
    ]
    targets = (RetrievalTarget(source="doc.md", section="A"),)

    assert reciprocal_rank(results, targets) == pytest.approx(0.5)


def test_reciprocal_rank_rank_3_returns_approximately_one_third() -> None:
    results = [
        make_result(0.9, source="other.md", section="X"),
        make_result(0.85, source="other2.md", section="Y"),
        make_result(0.8, source="doc.md", section="A"),
    ]
    targets = (RetrievalTarget(source="doc.md", section="A"),)

    assert reciprocal_rank(results, targets) == pytest.approx(1 / 3)


def test_reciprocal_rank_no_relevant_item_returns_zero() -> None:
    results = [make_result(0.9, source="other.md", section="X")]
    targets = (RetrievalTarget(source="doc.md", section="A"),)

    assert reciprocal_rank(results, targets) == 0.0


def test_reciprocal_rank_relevant_outside_max_rank_returns_zero() -> None:
    results = [
        make_result(0.9, source="other.md", section="X"),
        make_result(0.85, source="other2.md", section="Y"),
        make_result(0.8, source="doc.md", section="A"),
    ]
    targets = (RetrievalTarget(source="doc.md", section="A"),)

    assert reciprocal_rank(results, targets, max_rank=2) == 0.0


def test_reciprocal_rank_invalid_max_rank_raises_value_error() -> None:
    with pytest.raises(ValueError):
        reciprocal_rank([], (), max_rank=0)


# ---------------------------------------------------------------------------
# evaluate_retrieval
# ---------------------------------------------------------------------------


def test_evaluate_retrieval_preserves_case_order() -> None:
    case_a = RetrievalEvalCase(case_id="a", question="qa", answerable=False, relevant_targets=())
    case_b = RetrievalEvalCase(case_id="b", question="qb", answerable=False, relevant_targets=())

    eval_results, _ = evaluate_retrieval(FakeRetriever(), [case_a, case_b], top_k=5)

    assert [result.case.case_id for result in eval_results] == ["a", "b"]


def test_evaluate_retrieval_calls_retriever_once_per_case() -> None:
    cases = [
        RetrievalEvalCase(case_id="a", question="qa", answerable=False, relevant_targets=()),
        RetrievalEvalCase(case_id="b", question="qb", answerable=False, relevant_targets=()),
    ]
    retriever = FakeRetriever()

    evaluate_retrieval(retriever, cases, top_k=5)

    assert len(retriever.calls) == 2


def test_evaluate_retrieval_requests_at_least_supplied_top_k() -> None:
    cases = [RetrievalEvalCase(case_id="a", question="qa", answerable=False, relevant_targets=())]
    retriever = FakeRetriever()

    evaluate_retrieval(retriever, cases, top_k=7)

    assert retriever.calls[0] == ("qa", 7)


def test_evaluate_retrieval_computes_answerable_metrics_correctly() -> None:
    target = RetrievalTarget(source="doc.md", section="A")
    case = RetrievalEvalCase(case_id="a", question="qa", answerable=True, relevant_targets=(target,))
    result = make_result(0.9, source="doc.md", section="A")
    retriever = FakeRetriever(results_by_query={"qa": [result]})

    eval_results, summary = evaluate_retrieval(retriever, [case], top_k=5)

    assert eval_results[0].hit_at_1 == 1.0
    assert eval_results[0].hit_at_3 == 1.0
    assert eval_results[0].reciprocal_rank == pytest.approx(1.0)
    assert summary.hit_at_1 == pytest.approx(1.0)
    assert summary.hit_at_3 == pytest.approx(1.0)
    assert summary.mrr_at_5 == pytest.approx(1.0)


def test_evaluate_retrieval_unsupported_cases_receive_none_metrics() -> None:
    case = RetrievalEvalCase(case_id="u", question="qu", answerable=False, relevant_targets=())
    result = make_result(0.5, source="doc.md", section="A")
    retriever = FakeRetriever(results_by_query={"qu": [result]})

    eval_results, _ = evaluate_retrieval(retriever, [case], top_k=5)

    assert eval_results[0].hit_at_1 is None
    assert eval_results[0].hit_at_3 is None
    assert eval_results[0].reciprocal_rank is None


def test_evaluate_retrieval_unsupported_cases_retain_retrieved_results() -> None:
    case = RetrievalEvalCase(case_id="u", question="qu", answerable=False, relevant_targets=())
    result = make_result(0.5, source="doc.md", section="A")
    retriever = FakeRetriever(results_by_query={"qu": [result]})

    eval_results, _ = evaluate_retrieval(retriever, [case], top_k=5)

    assert eval_results[0].results == (result,)


def test_evaluate_retrieval_aggregate_excludes_unsupported_cases() -> None:
    answerable_target = RetrievalTarget(source="doc.md", section="A")
    answerable_case = RetrievalEvalCase(
        case_id="a", question="qa", answerable=True, relevant_targets=(answerable_target,)
    )
    unsupported_case = RetrievalEvalCase(
        case_id="u", question="qu", answerable=False, relevant_targets=()
    )
    good_result = make_result(0.9, source="doc.md", section="A")
    bad_result = make_result(0.1, source="other.md", section="Z")
    retriever = FakeRetriever(results_by_query={"qa": [good_result], "qu": [bad_result]})

    _, summary = evaluate_retrieval(retriever, [answerable_case, unsupported_case], top_k=5)

    assert summary.total_cases == 2
    assert summary.answerable_cases == 1
    assert summary.unsupported_cases == 1
    # A "miss" on the unsupported case must not drag the aggregate down,
    # since it is excluded from the denominator entirely.
    assert summary.hit_at_1 == pytest.approx(1.0)


def test_evaluate_retrieval_top_k_below_5_raises_value_error() -> None:
    with pytest.raises(ValueError):
        evaluate_retrieval(FakeRetriever(), [], top_k=4)


# ---------------------------------------------------------------------------
# Real-case consistency check (no OpenAI call)
# ---------------------------------------------------------------------------


def test_real_dataset_targets_match_current_chunker_output() -> None:
    """Every answerable target must refer to a source/section the real
    chunker currently produces, so ground truth cannot silently go
    stale if knowledge-base headings change. Makes no OpenAI call."""
    documents = load_knowledge_base("knowledge")
    chunks = chunk_knowledge_base(documents)

    existing_sources = {chunk.source for chunk in chunks}
    existing_pairs = {(chunk.source, chunk.section) for chunk in chunks}

    for case in RETRIEVAL_EVAL_CASES:
        for target in case.relevant_targets:
            assert target.source in existing_sources, (
                f"{case.case_id}: unknown source {target.source!r}"
            )
            if target.section is not None:
                assert (target.source, target.section) in existing_pairs, (
                    f"{case.case_id}: unknown section {target.section!r} "
                    f"for source {target.source!r}"
                )
