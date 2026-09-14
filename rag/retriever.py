"""Deterministic cosine-similarity semantic retrieval over embedded chunks.

Given a user query, embeds it with the same embedding service used for
the knowledge base and ranks the already-embedded corpus by cosine
similarity. Loading documents, chunking them, and embedding the
knowledge base are separate pipeline stages (see `rag.document_loader`,
`rag.chunker`, and `rag.embeddings`) and are not this module's
responsibility.
"""

import math
from dataclasses import dataclass

from rag.chunker import KnowledgeChunk
from rag.embeddings import EmbeddedChunk, OpenAIEmbeddingService


class RetrievalError(Exception):
    """Raised for invalid retrieval/vector conditions."""


@dataclass(frozen=True)
class RetrievalResult:
    """A knowledge-base chunk ranked for a query, with its similarity score."""

    chunk: KnowledgeChunk
    score: float


def cosine_similarity(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    """Compute the cosine similarity between two vectors.

    Args:
        left: The first vector.
        right: The second vector.

    Returns:
        `dot(left, right) / (norm(left) * norm(right))`, without rounding.

    Raises:
        RetrievalError: If either vector is empty, the vectors have
            different dimensions, or either vector has zero magnitude.
    """
    if not left or not right:
        raise RetrievalError("Cannot compute cosine similarity for an empty vector.")
    if len(left) != len(right):
        raise RetrievalError(f"Vector dimension mismatch: {len(left)} vs {len(right)}.")

    dot_product = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))

    if left_norm == 0.0 or right_norm == 0.0:
        raise RetrievalError("Cannot compute cosine similarity for a zero-magnitude vector.")

    return dot_product / (left_norm * right_norm)


def rank_embedded_chunks(
    query_embedding: tuple[float, ...],
    embedded_chunks: list[EmbeddedChunk],
    top_k: int = 5,
) -> list[RetrievalResult]:
    """Rank embedded chunks by cosine similarity to a query embedding.

    Args:
        query_embedding: The query's embedding vector.
        embedded_chunks: The knowledge-base corpus to rank.
        top_k: Maximum number of results to return. Must be greater than zero.

    Returns:
        Up to `top_k` `RetrievalResult` objects sorted by descending
        score. Exact-score ties preserve original corpus order. An
        empty corpus returns an empty list.

    Raises:
        ValueError: If `top_k` is not greater than zero.
        RetrievalError: If a vector is invalid (see `cosine_similarity`).
    """
    if top_k <= 0:
        raise ValueError("top_k must be greater than zero")

    if not embedded_chunks:
        return []

    scored = [
        RetrievalResult(
            chunk=embedded.chunk,
            score=cosine_similarity(query_embedding, embedded.embedding),
        )
        for embedded in embedded_chunks
    ]

    ranked = sorted(scored, key=lambda result: result.score, reverse=True)

    return ranked[:top_k]


def _validate_corpus(embedded_chunks: list[EmbeddedChunk]) -> None:
    """Validate a stored embedded corpus, when non-empty.

    Raises:
        RetrievalError: If any embedding is empty, dimensions are
            inconsistent across the corpus, or any embedding has zero
            magnitude.
    """
    if not embedded_chunks:
        return

    for embedded in embedded_chunks:
        if not embedded.embedding:
            raise RetrievalError(
                f"Embedded chunk {embedded.chunk.chunk_id} has an empty embedding vector."
            )

    dimensions = {len(embedded.embedding) for embedded in embedded_chunks}
    if len(dimensions) > 1:
        raise RetrievalError(
            f"Embedded corpus has inconsistent vector dimensions: {sorted(dimensions)}."
        )

    for embedded in embedded_chunks:
        magnitude = math.sqrt(sum(value * value for value in embedded.embedding))
        if magnitude == 0.0:
            raise RetrievalError(
                f"Embedded chunk {embedded.chunk.chunk_id} has a zero-magnitude embedding vector."
            )


class SemanticRetriever:
    """Ranks an already-embedded knowledge-base corpus against user queries.

    Holds embedded chunks produced elsewhere (see `embed_chunks()`) and
    uses the embedding service only to embed incoming queries. It never
    re-embeds the knowledge-base corpus itself.
    """

    def __init__(
        self,
        embedded_chunks: list[EmbeddedChunk],
        embedding_service: OpenAIEmbeddingService,
    ) -> None:
        """Create the retriever.

        Args:
            embedded_chunks: The knowledge-base corpus to search, e.g.
                from `embed_chunks()`. May be empty.
            embedding_service: Used only to embed queries passed to
                `retrieve()`.

        Raises:
            RetrievalError: If the non-empty corpus contains an empty,
                zero-magnitude, or dimensionally inconsistent embedding.
        """
        _validate_corpus(embedded_chunks)
        self._embedded_chunks = list(embedded_chunks)
        self._embedding_service = embedding_service

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        """Retrieve the most relevant chunks for a natural-language query.

        Args:
            query: The user's question, used exactly as given (no
                prepended instructions, filenames, or metadata).
            top_k: Maximum number of results to return. Must be greater
                than zero.

        Returns:
            Up to `top_k` `RetrievalResult` objects sorted by descending
            similarity score. An empty corpus returns `[]` without
            calling the embedding service.

        Raises:
            RetrievalError: If the query is empty/whitespace-only, or
                the embedding service does not return exactly one vector.
            ValueError: If `top_k` is not greater than zero.
        """
        if not query or not query.strip():
            raise RetrievalError("Query must not be empty or whitespace-only.")
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")

        if not self._embedded_chunks:
            return []

        query_embeddings = self._embedding_service.embed_texts([query])

        if len(query_embeddings) != 1:
            raise RetrievalError(
                f"Expected exactly one query embedding, got {len(query_embeddings)}."
            )

        return rank_embedded_chunks(query_embeddings[0], self._embedded_chunks, top_k=top_k)
