"""Tests for rag.retriever.

Uses a small fake embedding service (no OpenAI client, no environment
variables, no network access) so these tests never make real API
calls.
"""

import dataclasses

import pytest

from rag.chunker import KnowledgeChunk, chunk_knowledge_base
from rag.document_loader import load_knowledge_base
from rag.embeddings import EmbeddedChunk
from rag.retriever import (
    RetrievalError,
    RetrievalResult,
    SemanticRetriever,
    cosine_similarity,
    rank_embedded_chunks,
)


# ---------------------------------------------------------------------------
# Fakes and helpers
# ---------------------------------------------------------------------------


class FakeEmbeddingService:
    """Minimal fake embedding service: no client, no env vars, no network."""

    def __init__(self, vectors: list[tuple[float, ...]] | None = None) -> None:
        self._vectors = vectors if vectors is not None else []
        self.calls: list[list[str]] = []

    def embed_texts(self, texts: list[str]) -> list[tuple[float, ...]]:
        self.calls.append(list(texts))
        return list(self._vectors)


def make_chunk(
    chunk_id: str = "doc.md::chunk-001",
    source: str = "doc.md",
    title: str = "Título",
    section: str | None = "Sección",
    text: str = "Contenido de ejemplo.",
) -> KnowledgeChunk:
    return KnowledgeChunk(
        chunk_id=chunk_id, source=source, title=title, section=section, text=text
    )


def make_embedded_chunk(embedding: tuple[float, ...], **chunk_kwargs) -> EmbeddedChunk:
    return EmbeddedChunk(chunk=make_chunk(**chunk_kwargs), embedding=embedding)


# ---------------------------------------------------------------------------
# RetrievalResult
# ---------------------------------------------------------------------------


def test_retrieval_result_has_expected_fields() -> None:
    chunk = make_chunk()
    result = RetrievalResult(chunk=chunk, score=0.87)

    assert result.chunk is chunk
    assert result.score == 0.87


def test_retrieval_result_is_immutable() -> None:
    result = RetrievalResult(chunk=make_chunk(), score=0.5)

    with pytest.raises(dataclasses.FrozenInstanceError):
        result.score = 0.9


# ---------------------------------------------------------------------------
# cosine_similarity
# ---------------------------------------------------------------------------


def test_cosine_similarity_identical_vectors_is_approximately_one() -> None:
    vector = (1.0, 2.0, 3.0)

    assert cosine_similarity(vector, vector) == pytest.approx(1.0)


def test_cosine_similarity_orthogonal_vectors_is_approximately_zero() -> None:
    assert cosine_similarity((1.0, 0.0), (0.0, 1.0)) == pytest.approx(0.0)


def test_cosine_similarity_opposite_vectors_is_approximately_minus_one() -> None:
    assert cosine_similarity((1.0, 2.0), (-1.0, -2.0)) == pytest.approx(-1.0)


def test_cosine_similarity_works_with_non_normalized_vectors() -> None:
    assert cosine_similarity((10.0, 0.0), (5.0, 0.0)) == pytest.approx(1.0)


def test_cosine_similarity_empty_left_vector_raises_error() -> None:
    with pytest.raises(RetrievalError):
        cosine_similarity((), (1.0, 2.0))


def test_cosine_similarity_empty_right_vector_raises_error() -> None:
    with pytest.raises(RetrievalError):
        cosine_similarity((1.0, 2.0), ())


def test_cosine_similarity_dimension_mismatch_raises_error() -> None:
    with pytest.raises(RetrievalError):
        cosine_similarity((1.0, 2.0), (1.0, 2.0, 3.0))


def test_cosine_similarity_zero_magnitude_left_vector_raises_error() -> None:
    with pytest.raises(RetrievalError):
        cosine_similarity((0.0, 0.0), (1.0, 2.0))


def test_cosine_similarity_zero_magnitude_right_vector_raises_error() -> None:
    with pytest.raises(RetrievalError):
        cosine_similarity((1.0, 2.0), (0.0, 0.0))


# ---------------------------------------------------------------------------
# rank_embedded_chunks
# ---------------------------------------------------------------------------


def test_rank_embedded_chunks_ranks_highest_similarity_first() -> None:
    low = make_embedded_chunk((0.0, 1.0), chunk_id="doc.md::chunk-001")
    high = make_embedded_chunk((1.0, 0.0), chunk_id="doc.md::chunk-002")

    results = rank_embedded_chunks((1.0, 0.0), [low, high])

    assert results[0].chunk.chunk_id == "doc.md::chunk-002"
    assert results[1].chunk.chunk_id == "doc.md::chunk-001"


def test_rank_embedded_chunks_returns_retrieval_result_objects() -> None:
    embedded = make_embedded_chunk((1.0, 0.0))

    results = rank_embedded_chunks((1.0, 0.0), [embedded])

    assert all(isinstance(result, RetrievalResult) for result in results)


def test_rank_embedded_chunks_preserves_chunk_metadata() -> None:
    embedded = make_embedded_chunk(
        (1.0, 0.0),
        chunk_id="02_planes_y_precios.md::chunk-004",
        source="02_planes_y_precios.md",
        title="Planes y precios",
        section="Plan Profesional",
        text="El plan Profesional cuesta 20 USD.",
    )

    results = rank_embedded_chunks((1.0, 0.0), [embedded])

    assert results[0].chunk == embedded.chunk


def test_rank_embedded_chunks_top_k_limits_result_count() -> None:
    chunks = [
        make_embedded_chunk((1.0, float(i)), chunk_id=f"doc.md::chunk-{i:03d}")
        for i in range(5)
    ]

    results = rank_embedded_chunks((1.0, 0.0), chunks, top_k=2)

    assert len(results) == 2


def test_rank_embedded_chunks_top_k_larger_than_corpus_returns_all() -> None:
    chunks = [
        make_embedded_chunk((1.0, float(i)), chunk_id=f"doc.md::chunk-{i:03d}")
        for i in range(3)
    ]

    results = rank_embedded_chunks((1.0, 0.0), chunks, top_k=100)

    assert len(results) == 3


def test_rank_embedded_chunks_empty_corpus_returns_empty_list() -> None:
    assert rank_embedded_chunks((1.0, 0.0), []) == []


def test_rank_embedded_chunks_top_k_zero_raises_value_error() -> None:
    chunks = [make_embedded_chunk((1.0, 0.0))]

    with pytest.raises(ValueError):
        rank_embedded_chunks((1.0, 0.0), chunks, top_k=0)


def test_rank_embedded_chunks_negative_top_k_raises_value_error() -> None:
    chunks = [make_embedded_chunk((1.0, 0.0))]

    with pytest.raises(ValueError):
        rank_embedded_chunks((1.0, 0.0), chunks, top_k=-1)


def test_rank_embedded_chunks_exact_ties_preserve_original_order() -> None:
    first = make_embedded_chunk((1.0, 0.0), chunk_id="doc.md::chunk-001")
    second = make_embedded_chunk((1.0, 0.0), chunk_id="doc.md::chunk-002")
    third = make_embedded_chunk((1.0, 0.0), chunk_id="doc.md::chunk-003")

    results = rank_embedded_chunks((1.0, 0.0), [first, second, third])

    assert [result.chunk.chunk_id for result in results] == [
        "doc.md::chunk-001",
        "doc.md::chunk-002",
        "doc.md::chunk-003",
    ]


def test_rank_embedded_chunks_does_not_mutate_input_list() -> None:
    first = make_embedded_chunk((0.0, 1.0), chunk_id="doc.md::chunk-001")
    second = make_embedded_chunk((1.0, 0.0), chunk_id="doc.md::chunk-002")
    chunks = [first, second]

    rank_embedded_chunks((1.0, 0.0), chunks)

    assert chunks == [first, second]


def test_rank_embedded_chunks_invalid_vector_dimension_raises_error() -> None:
    chunks = [make_embedded_chunk((1.0, 0.0, 0.0))]

    with pytest.raises(RetrievalError):
        rank_embedded_chunks((1.0, 0.0), chunks)


# ---------------------------------------------------------------------------
# SemanticRetriever corpus validation
# ---------------------------------------------------------------------------


def test_semantic_retriever_accepts_empty_corpus() -> None:
    retriever = SemanticRetriever(embedded_chunks=[], embedding_service=FakeEmbeddingService())

    assert retriever is not None


def test_semantic_retriever_accepts_valid_consistent_corpus() -> None:
    chunks = [make_embedded_chunk((1.0, 0.0)), make_embedded_chunk((0.0, 1.0))]

    retriever = SemanticRetriever(
        embedded_chunks=chunks, embedding_service=FakeEmbeddingService()
    )

    assert retriever is not None


def test_semantic_retriever_rejects_empty_stored_embedding() -> None:
    chunks = [make_embedded_chunk(())]

    with pytest.raises(RetrievalError):
        SemanticRetriever(embedded_chunks=chunks, embedding_service=FakeEmbeddingService())


def test_semantic_retriever_rejects_inconsistent_stored_dimensions() -> None:
    chunks = [make_embedded_chunk((1.0, 0.0)), make_embedded_chunk((1.0, 0.0, 0.0))]

    with pytest.raises(RetrievalError):
        SemanticRetriever(embedded_chunks=chunks, embedding_service=FakeEmbeddingService())


def test_semantic_retriever_rejects_zero_magnitude_stored_embedding() -> None:
    chunks = [make_embedded_chunk((0.0, 0.0))]

    with pytest.raises(RetrievalError):
        SemanticRetriever(embedded_chunks=chunks, embedding_service=FakeEmbeddingService())


# ---------------------------------------------------------------------------
# SemanticRetriever.retrieve
# ---------------------------------------------------------------------------


def test_retrieve_rejects_empty_query() -> None:
    chunks = [make_embedded_chunk((1.0, 0.0))]
    retriever = SemanticRetriever(
        embedded_chunks=chunks, embedding_service=FakeEmbeddingService(vectors=[(1.0, 0.0)])
    )

    with pytest.raises(RetrievalError):
        retriever.retrieve("")


def test_retrieve_rejects_whitespace_only_query() -> None:
    chunks = [make_embedded_chunk((1.0, 0.0))]
    retriever = SemanticRetriever(
        embedded_chunks=chunks, embedding_service=FakeEmbeddingService(vectors=[(1.0, 0.0)])
    )

    with pytest.raises(RetrievalError):
        retriever.retrieve("   \n\t  ")


def test_retrieve_rejects_invalid_top_k() -> None:
    chunks = [make_embedded_chunk((1.0, 0.0))]
    retriever = SemanticRetriever(
        embedded_chunks=chunks, embedding_service=FakeEmbeddingService(vectors=[(1.0, 0.0)])
    )

    with pytest.raises(ValueError):
        retriever.retrieve("hola", top_k=0)


def test_retrieve_empty_corpus_returns_empty_list_without_calling_service() -> None:
    service = FakeEmbeddingService(vectors=[(1.0, 0.0)])
    retriever = SemanticRetriever(embedded_chunks=[], embedding_service=service)

    results = retriever.retrieve("¿Cuál es el precio?")

    assert results == []
    assert service.calls == []


def test_retrieve_calls_embed_texts_exactly_once() -> None:
    chunks = [make_embedded_chunk((1.0, 0.0))]
    service = FakeEmbeddingService(vectors=[(1.0, 0.0)])
    retriever = SemanticRetriever(embedded_chunks=chunks, embedding_service=service)

    retriever.retrieve("¿Cuál es el precio?")

    assert len(service.calls) == 1


def test_retrieve_passes_exact_original_query_to_embed_texts() -> None:
    chunks = [make_embedded_chunk((1.0, 0.0))]
    service = FakeEmbeddingService(vectors=[(1.0, 0.0)])
    retriever = SemanticRetriever(embedded_chunks=chunks, embedding_service=service)
    query = "¿Puedo pedir un reembolso del plan anual?"

    retriever.retrieve(query)

    assert service.calls[0] == [query]


def test_retrieve_ranks_query_embedding_against_corpus() -> None:
    chunks = [
        make_embedded_chunk((0.0, 1.0), chunk_id="doc.md::chunk-001"),
        make_embedded_chunk((1.0, 0.0), chunk_id="doc.md::chunk-002"),
    ]
    service = FakeEmbeddingService(vectors=[(1.0, 0.0)])
    retriever = SemanticRetriever(embedded_chunks=chunks, embedding_service=service)

    results = retriever.retrieve("consulta")

    assert results[0].chunk.chunk_id == "doc.md::chunk-002"


def test_retrieve_result_order_reflects_similarity() -> None:
    chunks = [
        make_embedded_chunk((0.0, 1.0), chunk_id="low"),
        make_embedded_chunk((0.9, 0.1), chunk_id="medium"),
        make_embedded_chunk((1.0, 0.0), chunk_id="high"),
    ]
    service = FakeEmbeddingService(vectors=[(1.0, 0.0)])
    retriever = SemanticRetriever(embedded_chunks=chunks, embedding_service=service)

    results = retriever.retrieve("consulta", top_k=3)

    assert [result.chunk.chunk_id for result in results] == ["high", "medium", "low"]


def test_retrieve_respects_top_k() -> None:
    chunks = [
        make_embedded_chunk((1.0, float(i)), chunk_id=f"doc.md::chunk-{i:03d}")
        for i in range(5)
    ]
    service = FakeEmbeddingService(vectors=[(1.0, 0.0)])
    retriever = SemanticRetriever(embedded_chunks=chunks, embedding_service=service)

    results = retriever.retrieve("consulta", top_k=2)

    assert len(results) == 2


def test_retrieve_zero_returned_query_vectors_raises_error() -> None:
    chunks = [make_embedded_chunk((1.0, 0.0))]
    service = FakeEmbeddingService(vectors=[])
    retriever = SemanticRetriever(embedded_chunks=chunks, embedding_service=service)

    with pytest.raises(RetrievalError):
        retriever.retrieve("consulta")


def test_retrieve_multiple_returned_query_vectors_raises_error() -> None:
    chunks = [make_embedded_chunk((1.0, 0.0))]
    service = FakeEmbeddingService(vectors=[(1.0, 0.0), (0.0, 1.0)])
    retriever = SemanticRetriever(embedded_chunks=chunks, embedding_service=service)

    with pytest.raises(RetrievalError):
        retriever.retrieve("consulta")


def test_retrieve_query_vector_dimension_mismatch_fails_clearly() -> None:
    chunks = [make_embedded_chunk((1.0, 0.0, 0.0))]
    service = FakeEmbeddingService(vectors=[(1.0, 0.0)])
    retriever = SemanticRetriever(embedded_chunks=chunks, embedding_service=service)

    with pytest.raises(RetrievalError):
        retriever.retrieve("consulta")


# ---------------------------------------------------------------------------
# Real knowledge-base integration check (no real OpenAI API)
# ---------------------------------------------------------------------------


def test_real_knowledge_base_retrieval_integration() -> None:
    documents = load_knowledge_base("knowledge")
    chunks = chunk_knowledge_base(documents)
    assert chunks

    # Deterministic synthetic vectors, only to prove component
    # interoperability -- not meant to reflect real semantic quality.
    embedded_chunks = [
        EmbeddedChunk(
            chunk=chunk,
            embedding=(
                float(i % 7) + 1.0,
                float((i + 1) % 5) + 1.0,
                float((i + 2) % 3) + 1.0,
            ),
        )
        for i, chunk in enumerate(chunks)
    ]

    service = FakeEmbeddingService(vectors=[(1.0, 2.0, 3.0)])
    retriever = SemanticRetriever(embedded_chunks=embedded_chunks, embedding_service=service)

    results = retriever.retrieve("¿Cuál es la política de reembolsos?", top_k=5)

    assert len(results) == 5
    assert all(isinstance(result, RetrievalResult) for result in results)
