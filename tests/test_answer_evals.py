"""Tests for the evals answer-evaluation package (answer_cases,
answer_checks, answer_runner).

Uses synthetic RAGAnswer/SourceCitation objects and a fake RAGAgent
exposing `.answer(question, top_k)`. Never uses OpenAI, network
access, or real embeddings.
"""

import dataclasses

import pytest

from agent.rag_agent import INSUFFICIENT_INFORMATION_MESSAGE, RAGAnswer, SourceCitation
from evals.answer_cases import ANSWER_EVAL_CASES, AnswerEvalCase
from evals.answer_checks import (
    check_citations,
    check_required_facts,
    citation_matches_target,
    normalize_text,
)
from evals.answer_runner import evaluate_answers
from evals.retrieval_cases import RETRIEVAL_EVAL_CASES, RetrievalTarget
from rag.chunker import chunk_knowledge_base
from rag.document_loader import load_knowledge_base


# ---------------------------------------------------------------------------
# Fakes and helpers
# ---------------------------------------------------------------------------


class FakeAgent:
    """Deterministic fake RAGAgent: no retriever, no generator, no network."""

    def __init__(
        self,
        answers_by_question: dict[str, RAGAnswer] | None = None,
        default_answer: RAGAnswer | None = None,
    ) -> None:
        self._answers_by_question = answers_by_question or {}
        self._default_answer = default_answer
        self.calls: list[tuple[str, int]] = []

    def answer(self, question: str, top_k: int = 5) -> RAGAnswer:
        self.calls.append((question, top_k))
        return self._answers_by_question.get(question, self._default_answer)


def make_citation(
    source: str = "doc.md", title: str = "Título", section: str | None = "Sección"
) -> SourceCitation:
    return SourceCitation(source=source, title=title, section=section)


def make_rag_answer(
    answer: str = "Respuesta.",
    answerable: bool = True,
    citations: tuple[SourceCitation, ...] = (),
) -> RAGAnswer:
    return RAGAnswer(
        answer=answer, answerable=answerable, citations=citations, retrieved_results=()
    )


# ---------------------------------------------------------------------------
# Dataset integrity
# ---------------------------------------------------------------------------


def test_answer_eval_case_has_expected_fields() -> None:
    target = RetrievalTarget(source="doc.md", section="Sección")
    case = AnswerEvalCase(
        case_id="case-1",
        question="¿Pregunta?",
        answerable=True,
        required_fact_groups=(("hecho",),),
        acceptable_citations=(target,),
    )

    assert case.case_id == "case-1"
    assert case.question == "¿Pregunta?"
    assert case.answerable is True
    assert case.required_fact_groups == (("hecho",),)
    assert case.acceptable_citations == (target,)


def test_answer_eval_case_is_immutable() -> None:
    case = AnswerEvalCase(
        case_id="c", question="q", answerable=True, required_fact_groups=(), acceptable_citations=()
    )

    with pytest.raises(dataclasses.FrozenInstanceError):
        case.answerable = False


def test_dataset_has_exactly_16_answerable_cases() -> None:
    answerable = [case for case in ANSWER_EVAL_CASES if case.answerable]

    assert len(answerable) == 16


def test_dataset_has_exactly_4_unsupported_cases() -> None:
    unsupported = [case for case in ANSWER_EVAL_CASES if not case.answerable]

    assert len(unsupported) == 4


def test_case_ids_are_unique() -> None:
    case_ids = [case.case_id for case in ANSWER_EVAL_CASES]

    assert len(case_ids) == len(set(case_ids))


def test_answer_cases_align_with_retrieval_case_ids_and_questions() -> None:
    retrieval_by_id = {case.case_id: case for case in RETRIEVAL_EVAL_CASES}

    assert {case.case_id for case in ANSWER_EVAL_CASES} == set(retrieval_by_id.keys())
    for case in ANSWER_EVAL_CASES:
        assert case.question == retrieval_by_id[case.case_id].question


def test_answerability_labels_align_with_retrieval_dataset() -> None:
    retrieval_by_id = {case.case_id: case for case in RETRIEVAL_EVAL_CASES}

    for case in ANSWER_EVAL_CASES:
        assert case.answerable == retrieval_by_id[case.case_id].answerable


def test_answerable_cases_have_required_fact_groups() -> None:
    for case in ANSWER_EVAL_CASES:
        if case.answerable:
            assert len(case.required_fact_groups) >= 1, case.case_id


def test_answerable_cases_have_acceptable_citations() -> None:
    for case in ANSWER_EVAL_CASES:
        if case.answerable:
            assert len(case.acceptable_citations) >= 1, case.case_id


def test_unsupported_cases_have_no_fact_groups() -> None:
    for case in ANSWER_EVAL_CASES:
        if not case.answerable:
            assert case.required_fact_groups == (), case.case_id


def test_unsupported_cases_have_no_acceptable_citations() -> None:
    for case in ANSWER_EVAL_CASES:
        if not case.answerable:
            assert case.acceptable_citations == (), case.case_id


# ---------------------------------------------------------------------------
# normalize_text
# ---------------------------------------------------------------------------


def test_normalize_text_is_case_insensitive() -> None:
    assert normalize_text("FACTURACIÓN") == normalize_text("facturación")


def test_normalize_text_is_accent_insensitive() -> None:
    assert normalize_text("Facturación ANUAL") == normalize_text("facturacion anual")


def test_normalize_text_collapses_whitespace() -> None:
    assert normalize_text("hola    mundo\n\tcómo estás") == "hola mundo como estas"


def test_normalize_text_preserves_numeric_and_symbol_content() -> None:
    normalized = normalize_text("El precio es $10 USD")

    assert "10" in normalized
    assert "$10" in normalized


# ---------------------------------------------------------------------------
# check_required_facts
# ---------------------------------------------------------------------------


def test_check_required_facts_all_groups_present_passes() -> None:
    groups = (("10 usd",), ("mensual",))

    assert check_required_facts("El plan cuesta 10 USD de forma mensual.", groups) is True


def test_check_required_facts_missing_one_group_fails() -> None:
    groups = (("10 usd",), ("mensual",))

    assert check_required_facts("El plan cuesta 10 USD al año.", groups) is False


def test_check_required_facts_alternative_wording_works() -> None:
    groups = (("10 usd", "usd 10", "$10"),)

    assert check_required_facts("El precio es de $10 por usuario.", groups) is True


def test_check_required_facts_empty_groups_pass() -> None:
    assert check_required_facts("Cualquier respuesta.", ()) is True


def test_check_required_facts_spanish_accents_normalize_correctly() -> None:
    groups = (("facturación anual",),)

    assert check_required_facts("La FACTURACION ANUAL tiene descuento.", groups) is True


# ---------------------------------------------------------------------------
# citation_matches_target / check_citations
# ---------------------------------------------------------------------------


def test_citation_matches_target_exact_source_and_section() -> None:
    citation = make_citation(source="doc.md", section="Sección A")
    target = RetrievalTarget(source="doc.md", section="Sección A")

    assert citation_matches_target(citation, target) is True


def test_citation_matches_target_wrong_source_fails() -> None:
    citation = make_citation(source="other.md", section="Sección A")
    target = RetrievalTarget(source="doc.md", section="Sección A")

    assert citation_matches_target(citation, target) is False


def test_citation_matches_target_wrong_section_fails() -> None:
    citation = make_citation(source="doc.md", section="Sección B")
    target = RetrievalTarget(source="doc.md", section="Sección A")

    assert citation_matches_target(citation, target) is False


def test_citation_matches_target_section_none_source_level() -> None:
    citation = make_citation(source="doc.md", section="Cualquier sección")
    target = RetrievalTarget(source="doc.md", section=None)

    assert citation_matches_target(citation, target) is True


def test_check_citations_requires_at_least_one_citation() -> None:
    targets = (RetrievalTarget(source="doc.md", section="A"),)

    assert check_citations((), targets) is False


def test_check_citations_multiple_valid_citations_pass() -> None:
    targets = (
        RetrievalTarget(source="doc.md", section="A"),
        RetrievalTarget(source="other.md", section="B"),
    )
    citations = (
        make_citation(source="doc.md", section="A"),
        make_citation(source="other.md", section="B"),
    )

    assert check_citations(citations, targets) is True


def test_check_citations_one_valid_one_invalid_fails() -> None:
    targets = (RetrievalTarget(source="doc.md", section="A"),)
    citations = (
        make_citation(source="doc.md", section="A"),
        make_citation(source="unrelated.md", section="Z"),
    )

    assert check_citations(citations, targets) is False


def test_check_citations_empty_fails_for_answerable_case() -> None:
    targets = (RetrievalTarget(source="doc.md", section="A"),)

    assert check_citations((), targets) is False


# ---------------------------------------------------------------------------
# evaluate_answers -- answerable case evaluation
# ---------------------------------------------------------------------------


def test_answerable_case_fully_correct_passes() -> None:
    case = AnswerEvalCase(
        case_id="a",
        question="qa",
        answerable=True,
        required_fact_groups=(("10 usd",),),
        acceptable_citations=(RetrievalTarget(source="doc.md", section="A"),),
    )
    answer = make_rag_answer(
        answer="Cuesta 10 USD.",
        answerable=True,
        citations=(make_citation(source="doc.md", section="A"),),
    )
    agent = FakeAgent(answers_by_question={"qa": answer})

    results, _ = evaluate_answers(agent, [case])

    assert results[0].passed is True


def test_answerable_case_wrong_answerability_fails() -> None:
    case = AnswerEvalCase(
        case_id="a",
        question="qa",
        answerable=True,
        required_fact_groups=(),
        acceptable_citations=(RetrievalTarget(source="doc.md", section="A"),),
    )
    answer = make_rag_answer(answer="", answerable=False, citations=())
    agent = FakeAgent(answers_by_question={"qa": answer})

    results, _ = evaluate_answers(agent, [case])

    assert results[0].answerability_correct is False
    assert results[0].passed is False


def test_answerable_case_missing_required_fact_fails() -> None:
    case = AnswerEvalCase(
        case_id="a",
        question="qa",
        answerable=True,
        required_fact_groups=(("10 usd",),),
        acceptable_citations=(RetrievalTarget(source="doc.md", section="A"),),
    )
    answer = make_rag_answer(
        answer="Cuesta 20 USD.",
        answerable=True,
        citations=(make_citation(source="doc.md", section="A"),),
    )
    agent = FakeAgent(answers_by_question={"qa": answer})

    results, _ = evaluate_answers(agent, [case])

    assert results[0].facts_pass is False
    assert results[0].passed is False


def test_answerable_case_invalid_citation_fails() -> None:
    case = AnswerEvalCase(
        case_id="a",
        question="qa",
        answerable=True,
        required_fact_groups=(("10 usd",),),
        acceptable_citations=(RetrievalTarget(source="doc.md", section="A"),),
    )
    answer = make_rag_answer(
        answer="Cuesta 10 USD.",
        answerable=True,
        citations=(make_citation(source="unrelated.md", section="Z"),),
    )
    agent = FakeAgent(answers_by_question={"qa": answer})

    results, _ = evaluate_answers(agent, [case])

    assert results[0].citations_pass is False
    assert results[0].passed is False


# ---------------------------------------------------------------------------
# evaluate_answers -- unsupported evaluation
# ---------------------------------------------------------------------------


def test_unsupported_exact_fallback_no_citations_passes() -> None:
    case = AnswerEvalCase(
        case_id="u", question="qu", answerable=False, required_fact_groups=(), acceptable_citations=()
    )
    answer = make_rag_answer(answer=INSUFFICIENT_INFORMATION_MESSAGE, answerable=False, citations=())
    agent = FakeAgent(answers_by_question={"qu": answer})

    results, _ = evaluate_answers(agent, [case])

    assert results[0].fallback_pass is True
    assert results[0].passed is True


def test_unsupported_wrong_fallback_text_fails() -> None:
    case = AnswerEvalCase(
        case_id="u", question="qu", answerable=False, required_fact_groups=(), acceptable_citations=()
    )
    answer = make_rag_answer(answer="No lo sé.", answerable=False, citations=())
    agent = FakeAgent(answers_by_question={"qu": answer})

    results, _ = evaluate_answers(agent, [case])

    assert results[0].fallback_pass is False
    assert results[0].passed is False


def test_unsupported_result_with_citation_fails() -> None:
    case = AnswerEvalCase(
        case_id="u", question="qu", answerable=False, required_fact_groups=(), acceptable_citations=()
    )
    answer = make_rag_answer(
        answer=INSUFFICIENT_INFORMATION_MESSAGE, answerable=False, citations=(make_citation(),)
    )
    agent = FakeAgent(answers_by_question={"qu": answer})

    results, _ = evaluate_answers(agent, [case])

    assert results[0].fallback_pass is False
    assert results[0].passed is False


def test_unsupported_incorrect_answerable_true_fails() -> None:
    case = AnswerEvalCase(
        case_id="u", question="qu", answerable=False, required_fact_groups=(), acceptable_citations=()
    )
    answer = make_rag_answer(answer="Sí, es posible.", answerable=True, citations=())
    agent = FakeAgent(answers_by_question={"qu": answer})

    results, _ = evaluate_answers(agent, [case])

    assert results[0].answerability_correct is False
    assert results[0].passed is False


# ---------------------------------------------------------------------------
# evaluate_answers -- runner behavior
# ---------------------------------------------------------------------------


def test_runner_preserves_case_order() -> None:
    case_a = AnswerEvalCase(
        case_id="a", question="qa", answerable=False, required_fact_groups=(), acceptable_citations=()
    )
    case_b = AnswerEvalCase(
        case_id="b", question="qb", answerable=False, required_fact_groups=(), acceptable_citations=()
    )
    refusal = make_rag_answer(answer=INSUFFICIENT_INFORMATION_MESSAGE, answerable=False, citations=())
    agent = FakeAgent(default_answer=refusal)

    results, _ = evaluate_answers(agent, [case_a, case_b])

    assert [result.case.case_id for result in results] == ["a", "b"]


def test_runner_calls_agent_once_per_case() -> None:
    cases = [
        AnswerEvalCase(
            case_id="a", question="qa", answerable=False, required_fact_groups=(), acceptable_citations=()
        ),
        AnswerEvalCase(
            case_id="b", question="qb", answerable=False, required_fact_groups=(), acceptable_citations=()
        ),
    ]
    refusal = make_rag_answer(answer=INSUFFICIENT_INFORMATION_MESSAGE, answerable=False, citations=())
    agent = FakeAgent(default_answer=refusal)

    evaluate_answers(agent, cases)

    assert len(agent.calls) == 2


def test_runner_passes_exact_question() -> None:
    case = AnswerEvalCase(
        case_id="a",
        question="¿Pregunta exacta?",
        answerable=False,
        required_fact_groups=(),
        acceptable_citations=(),
    )
    refusal = make_rag_answer(answer=INSUFFICIENT_INFORMATION_MESSAGE, answerable=False, citations=())
    agent = FakeAgent(default_answer=refusal)

    evaluate_answers(agent, [case])

    assert agent.calls[0][0] == "¿Pregunta exacta?"


def test_runner_passes_top_k() -> None:
    case = AnswerEvalCase(
        case_id="a", question="qa", answerable=False, required_fact_groups=(), acceptable_citations=()
    )
    refusal = make_rag_answer(answer=INSUFFICIENT_INFORMATION_MESSAGE, answerable=False, citations=())
    agent = FakeAgent(default_answer=refusal)

    evaluate_answers(agent, [case], top_k=7)

    assert agent.calls[0][1] == 7


def test_runner_invalid_top_k_raises_value_error() -> None:
    with pytest.raises(ValueError):
        evaluate_answers(FakeAgent(), [], top_k=0)


def test_runner_summary_metrics_calculated_correctly() -> None:
    answerable_case = AnswerEvalCase(
        case_id="a",
        question="qa",
        answerable=True,
        required_fact_groups=(("10 usd",),),
        acceptable_citations=(RetrievalTarget(source="doc.md", section="A"),),
    )
    unsupported_case = AnswerEvalCase(
        case_id="u", question="qu", answerable=False, required_fact_groups=(), acceptable_citations=()
    )
    good_answer = make_rag_answer(
        answer="Cuesta 10 USD.",
        answerable=True,
        citations=(make_citation(source="doc.md", section="A"),),
    )
    refusal_answer = make_rag_answer(
        answer=INSUFFICIENT_INFORMATION_MESSAGE, answerable=False, citations=()
    )
    agent = FakeAgent(answers_by_question={"qa": good_answer, "qu": refusal_answer})

    _, summary = evaluate_answers(agent, [answerable_case, unsupported_case])

    assert summary.total_cases == 2
    assert summary.answerable_cases == 1
    assert summary.unsupported_cases == 1
    assert summary.answerability_accuracy == pytest.approx(1.0)
    assert summary.fact_check_pass_rate == pytest.approx(1.0)
    assert summary.citation_check_pass_rate == pytest.approx(1.0)
    assert summary.unsupported_refusal_rate == pytest.approx(1.0)
    assert summary.overall_pass_rate == pytest.approx(1.0)


def test_runner_unsupported_cases_excluded_from_fact_citation_denominators() -> None:
    answerable_case = AnswerEvalCase(
        case_id="a",
        question="qa",
        answerable=True,
        required_fact_groups=(("10 usd",),),
        acceptable_citations=(RetrievalTarget(source="doc.md", section="A"),),
    )
    unsupported_case = AnswerEvalCase(
        case_id="u", question="qu", answerable=False, required_fact_groups=(), acceptable_citations=()
    )
    good_answer = make_rag_answer(
        answer="Cuesta 10 USD.",
        answerable=True,
        citations=(make_citation(source="doc.md", section="A"),),
    )
    # A wrong fallback text on the unsupported case: if it leaked into
    # the fact/citation denominators, the rates below would drop below 1.
    wrong_answer = make_rag_answer(answer="algo distinto", answerable=False, citations=())
    agent = FakeAgent(answers_by_question={"qa": good_answer, "qu": wrong_answer})

    _, summary = evaluate_answers(agent, [answerable_case, unsupported_case])

    assert summary.fact_check_pass_rate == pytest.approx(1.0)
    assert summary.citation_check_pass_rate == pytest.approx(1.0)


def test_runner_answerable_cases_excluded_from_unsupported_refusal_denominator() -> None:
    answerable_case = AnswerEvalCase(
        case_id="a",
        question="qa",
        answerable=True,
        required_fact_groups=(("10 usd",),),
        acceptable_citations=(RetrievalTarget(source="doc.md", section="A"),),
    )
    unsupported_case = AnswerEvalCase(
        case_id="u", question="qu", answerable=False, required_fact_groups=(), acceptable_citations=()
    )
    # The answerable case is (incorrectly) refused: if it leaked into the
    # unsupported-refusal denominator, the rate below would move off 1.0.
    wrong_answer = make_rag_answer(
        answer=INSUFFICIENT_INFORMATION_MESSAGE, answerable=False, citations=()
    )
    refusal_answer = make_rag_answer(
        answer=INSUFFICIENT_INFORMATION_MESSAGE, answerable=False, citations=()
    )
    agent = FakeAgent(answers_by_question={"qa": wrong_answer, "qu": refusal_answer})

    _, summary = evaluate_answers(agent, [answerable_case, unsupported_case])

    assert summary.unsupported_refusal_rate == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Real knowledge-base consistency check (no OpenAI call)
# ---------------------------------------------------------------------------


def test_real_dataset_citation_targets_match_current_chunker_output() -> None:
    """Every acceptable citation target must refer to a source/section
    the real chunker currently produces, so ground truth cannot silently
    go stale if knowledge-base headings change. Makes no OpenAI call."""
    documents = load_knowledge_base("knowledge")
    chunks = chunk_knowledge_base(documents)

    existing_sources = {chunk.source for chunk in chunks}
    existing_pairs = {(chunk.source, chunk.section) for chunk in chunks}

    for case in ANSWER_EVAL_CASES:
        for target in case.acceptable_citations:
            assert target.source in existing_sources, (
                f"{case.case_id}: unknown source {target.source!r}"
            )
            if target.section is not None:
                assert (target.source, target.section) in existing_pairs, (
                    f"{case.case_id}: unknown section {target.section!r} "
                    f"for source {target.source!r}"
                )
