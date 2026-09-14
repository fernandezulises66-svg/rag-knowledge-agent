"""Runs grounded-answer evaluation cases against a real `RAGAgent` and
aggregates answerability, factual, citation, and refusal metrics.

This evaluates the FINAL grounded RAG pipeline exactly as a user would
call it: only `agent.answer()` is invoked per case. Retrieval and
generation are never called separately here, and these metrics are
distinct from (and not comparable to) the retrieval-only Hit@K/MRR
benchmark in `evals/retrieval_runner.py`.
"""

from dataclasses import dataclass

from agent.rag_agent import INSUFFICIENT_INFORMATION_MESSAGE, RAGAgent, RAGAnswer
from evals.answer_cases import AnswerEvalCase
from evals.answer_checks import check_citations, check_required_facts


@dataclass(frozen=True)
class AnswerEvalResult:
    """Per-case grounded-answer evaluation outcome.

    For answerable cases, `fallback_pass` is `None`. For unsupported
    cases, `facts_pass` and `citations_pass` are `None` -- there is
    nothing to check in cases that were never meant to be answered.
    """

    case: AnswerEvalCase
    rag_answer: RAGAnswer
    answerability_correct: bool
    facts_pass: bool | None
    citations_pass: bool | None
    fallback_pass: bool | None
    passed: bool


@dataclass(frozen=True)
class AnswerEvalSummary:
    """Aggregate grounded-answer evaluation metrics over this curated benchmark.

    These metrics apply only to this hand-curated 20-question
    benchmark, not to general model accuracy.
    """

    total_cases: int
    answerable_cases: int
    unsupported_cases: int
    answerability_accuracy: float
    fact_check_pass_rate: float
    citation_check_pass_rate: float
    unsupported_refusal_rate: float
    overall_pass_rate: float


def evaluate_answers(
    agent: RAGAgent,
    cases: list[AnswerEvalCase],
    top_k: int = 5,
) -> tuple[list[AnswerEvalResult], AnswerEvalSummary]:
    """Run every case through `agent.answer()` and score it.

    Args:
        agent: The full RAG pipeline (retriever + generator) to evaluate.
        cases: Evaluation cases, evaluated in order.
        top_k: Results requested per case. Must be greater than zero.

    Returns:
        Per-case results, in case order, and an aggregate summary.

    Raises:
        ValueError: If `top_k` is not greater than zero.
    """
    if top_k <= 0:
        raise ValueError("top_k must be greater than zero")

    results: list[AnswerEvalResult] = []

    for case in cases:
        rag_answer = agent.answer(case.question, top_k=top_k)
        answerability_correct = rag_answer.answerable == case.answerable

        if case.answerable:
            facts_pass = check_required_facts(rag_answer.answer, case.required_fact_groups)
            citations_pass = check_citations(rag_answer.citations, case.acceptable_citations)
            fallback_pass = None
            passed = answerability_correct and facts_pass and citations_pass
        else:
            facts_pass = None
            citations_pass = None
            fallback_pass = (
                rag_answer.answer == INSUFFICIENT_INFORMATION_MESSAGE
                and rag_answer.citations == ()
            )
            passed = answerability_correct and fallback_pass

        results.append(
            AnswerEvalResult(
                case=case,
                rag_answer=rag_answer,
                answerability_correct=answerability_correct,
                facts_pass=facts_pass,
                citations_pass=citations_pass,
                fallback_pass=fallback_pass,
                passed=passed,
            )
        )

    total_cases = len(results)
    answerable_results = [result for result in results if result.case.answerable]
    unsupported_results = [result for result in results if not result.case.answerable]
    answerable_count = len(answerable_results)
    unsupported_count = len(unsupported_results)

    answerability_accuracy = (
        sum(1 for result in results if result.answerability_correct) / total_cases
        if total_cases > 0
        else 0.0
    )
    fact_check_pass_rate = (
        sum(1 for result in answerable_results if result.facts_pass) / answerable_count
        if answerable_count > 0
        else 0.0
    )
    citation_check_pass_rate = (
        sum(1 for result in answerable_results if result.citations_pass) / answerable_count
        if answerable_count > 0
        else 0.0
    )
    unsupported_refusal_rate = (
        sum(1 for result in unsupported_results if result.fallback_pass) / unsupported_count
        if unsupported_count > 0
        else 0.0
    )
    overall_pass_rate = (
        sum(1 for result in results if result.passed) / total_cases if total_cases > 0 else 0.0
    )

    summary = AnswerEvalSummary(
        total_cases=total_cases,
        answerable_cases=answerable_count,
        unsupported_cases=unsupported_count,
        answerability_accuracy=answerability_accuracy,
        fact_check_pass_rate=fact_check_pass_rate,
        citation_check_pass_rate=citation_check_pass_rate,
        unsupported_refusal_rate=unsupported_refusal_rate,
        overall_pass_rate=overall_pass_rate,
    )

    return results, summary
