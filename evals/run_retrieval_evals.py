"""CLI: python -m evals.run_retrieval_evals

Runs the retrieval evaluation cases against the real knowledge base
using the real OpenAI embeddings API, and prints per-case and
aggregate Hit@1 / Hit@3 / MRR@5 results.

This script makes real OpenAI API calls and network requests. It is
never invoked by pytest and must always be run manually.
"""

from rag.chunker import chunk_knowledge_base
from rag.document_loader import load_knowledge_base
from rag.embeddings import OpenAIEmbeddingService, embed_chunks
from rag.retriever import SemanticRetriever

from evals.retrieval_cases import RETRIEVAL_EVAL_CASES
from evals.retrieval_runner import RetrievalEvalResult, evaluate_retrieval

TOP_K = 5


def _print_case_result(result: RetrievalEvalResult) -> None:
    case = result.case

    if case.answerable:
        passed = result.hit_at_3 == 1.0
        label = "PASS" if passed else "FAIL"
        print(f"[{label}] {case.case_id}")
        print(f"Question: {case.question}")
        print("Expected:")
        for target in case.relevant_targets:
            section = target.section if target.section is not None else "(any section)"
            print(f"  {target.source} | {section}")
        print("Retrieved:")
        for rank, retrieved in enumerate(result.results, start=1):
            print(
                f"  {rank}. {retrieved.chunk.source} | {retrieved.chunk.section} "
                f"| score={retrieved.score:.4f}"
            )
        print(f"Hit@1: {int(result.hit_at_1)}")
        print(f"Hit@3: {int(result.hit_at_3)}")
        print(f"RR@5: {result.reciprocal_rank:.3f}")
    else:
        print(f"[DIAGNOSTIC] {case.case_id}")
        print(f"Question: {case.question}")
        print("Top retrievals:")
        for rank, retrieved in enumerate(result.results, start=1):
            print(
                f"  {rank}. {retrieved.chunk.source} | {retrieved.chunk.section} "
                f"| score={retrieved.score:.4f}"
            )
        print(
            "No retrieval accuracy metric is assigned because the question "
            "is intentionally unsupported."
        )

    print()


def main() -> None:
    documents = load_knowledge_base("knowledge")
    chunks = chunk_knowledge_base(documents)

    embedding_service = OpenAIEmbeddingService()
    embedded_chunks = embed_chunks(chunks, embedding_service)

    retriever = SemanticRetriever(
        embedded_chunks=embedded_chunks, embedding_service=embedding_service
    )

    eval_results, summary = evaluate_retrieval(
        retriever, list(RETRIEVAL_EVAL_CASES), top_k=TOP_K
    )

    for result in eval_results:
        _print_case_result(result)

    print("Retrieval Evaluation")
    print("--------------------")
    print(f"Answerable cases: {summary.answerable_cases}")
    print(f"Unsupported diagnostic cases: {summary.unsupported_cases}")
    print(f"Hit@1: {summary.hit_at_1:.3f}")
    print(f"Hit@3: {summary.hit_at_3:.3f}")
    print(f"MRR@5: {summary.mrr_at_5:.3f}")


if __name__ == "__main__":
    main()
