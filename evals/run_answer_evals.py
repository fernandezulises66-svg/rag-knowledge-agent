"""CLI: python -m evals.run_answer_evals

Runs the grounded-answer evaluation cases against the real knowledge
base, the real OpenAI embeddings API, and the real generation model,
and prints per-case checks and an aggregate summary.

This evaluates the FINAL RAG answers -- answerability, factual
grounding, citation validity, and controlled refusal -- as a separate
concern from the existing retrieval-only benchmark (see
`evals/run_retrieval_evals.py`, which measures Hit@K/MRR only). This
script makes real OpenAI API calls and network requests. It is never
invoked by pytest and must always be run manually.
"""

from agent.rag_agent import OpenAIGroundedGenerator, RAGAgent
from evals.answer_cases import ANSWER_EVAL_CASES
from evals.answer_runner import AnswerEvalResult, evaluate_answers
from rag.chunker import chunk_knowledge_base
from rag.document_loader import load_knowledge_base
from rag.embeddings import OpenAIEmbeddingService, embed_chunks
from rag.retriever import SemanticRetriever

TOP_K = 5


def _format_check(passed: bool) -> str:
    return "PASS" if passed else "FAIL"


def _failed_checks(result: AnswerEvalResult) -> list[str]:
    failed = []
    if not result.answerability_correct:
        failed.append("answerability")
    if result.case.answerable:
        if not result.facts_pass:
            failed.append("facts")
        if not result.citations_pass:
            failed.append("citations")
    else:
        if not result.fallback_pass:
            failed.append("fallback")
    return failed


def _print_case_result(result: AnswerEvalResult) -> None:
    case = result.case
    answer = result.rag_answer

    print(f"[{'PASS' if result.passed else 'FAIL'}] {case.case_id}")
    print(f"Question: {case.question}")
    print()
    print(f"Answerable expected: {case.answerable}")
    print(f"Answerable actual: {answer.answerable}")
    print()

    if case.answerable:
        print("Answer:")
        print(answer.answer)
        print()
        print("Citations:")
        if answer.citations:
            for citation in answer.citations:
                section = citation.section if citation.section is not None else "(sin sección)"
                print(f"  - {citation.source} | {section}")
        else:
            print("  (none)")
        print()
        print(f"Facts: {_format_check(bool(result.facts_pass))}")
        print(f"Citations: {_format_check(bool(result.citations_pass))}")
    else:
        print(f"Fallback: {_format_check(bool(result.fallback_pass))}")
        print("Citations: none" if answer.citations == () else f"Citations: {answer.citations}")

    if not result.passed:
        print(f"Failed checks: {', '.join(_failed_checks(result))}")

    print()


def main() -> None:
    documents = load_knowledge_base("knowledge")
    chunks = chunk_knowledge_base(documents)

    embedding_service = OpenAIEmbeddingService()
    embedded_chunks = embed_chunks(chunks, embedding_service)

    retriever = SemanticRetriever(
        embedded_chunks=embedded_chunks, embedding_service=embedding_service
    )
    generator = OpenAIGroundedGenerator()
    agent = RAGAgent(retriever=retriever, generator=generator)

    results, summary = evaluate_answers(agent, list(ANSWER_EVAL_CASES), top_k=TOP_K)

    for result in results:
        _print_case_result(result)

    print("Grounded Answer Evaluation")
    print("--------------------------")
    print(f"Answerable cases: {summary.answerable_cases}")
    print(f"Unsupported cases: {summary.unsupported_cases}")
    print(f"Answerability accuracy: {summary.answerability_accuracy:.3f}")
    print(f"Fact check pass rate: {summary.fact_check_pass_rate:.3f}")
    print(f"Citation check pass rate: {summary.citation_check_pass_rate:.3f}")
    print(f"Unsupported refusal rate: {summary.unsupported_refusal_rate:.3f}")
    print(f"Overall pass rate: {summary.overall_pass_rate:.3f}")
    print()
    print(
        "These metrics apply only to this curated 20-question benchmark, "
        "not to general model accuracy."
    )


if __name__ == "__main__":
    main()
