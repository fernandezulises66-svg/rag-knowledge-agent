"""OpenAI embedding generation for the RAG knowledge base.

Isolates all direct calls to the OpenAI embeddings API behind
`OpenAIEmbeddingService`, and provides a deterministic way to turn a
`KnowledgeChunk` into embedding input text and a list of chunks into
`EmbeddedChunk` objects. Vector search and retrieval are out of scope
for this module.
"""

import os
from dataclasses import dataclass
from typing import Any

from rag.chunker import KnowledgeChunk

DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"


class EmbeddingError(Exception):
    """Raised for invalid embedding-service responses or embedding-layer errors."""


@dataclass(frozen=True)
class EmbeddedChunk:
    """A knowledge-base chunk paired with its embedding vector."""

    chunk: KnowledgeChunk
    embedding: tuple[float, ...]


def build_embedding_text(chunk: KnowledgeChunk) -> str:
    """Build deterministic embedding input text for a chunk.

    Combines the document title and, when present, the section with
    the chunk's own text so retrieval can use this extra semantic
    context. Does not mutate `chunk` or include the source file name
    or chunk ID.
    """
    lines = [f"Título: {chunk.title}"]
    if chunk.section is not None:
        lines.append(f"Sección: {chunk.section}")

    header = "\n".join(lines)
    return f"{header}\n\n{chunk.text}"


def _default_model() -> str:
    return os.environ.get("OPENAI_EMBEDDING_MODEL") or DEFAULT_EMBEDDING_MODEL


def _load_dotenv_once() -> None:
    """Load variables from a local `.env` file into the environment, if present."""
    from dotenv import load_dotenv

    load_dotenv()


def _build_default_client() -> Any:
    from openai import OpenAI

    return OpenAI()


def _reconstruct_embeddings(response: Any, expected_count: int) -> list[tuple[float, ...]]:
    """Validate an embeddings API response and restore input order.

    Raises:
        EmbeddingError: If the response item count, indexes, or vector
            shapes are inconsistent with a valid embeddings response.
    """
    data = getattr(response, "data", None)
    if data is None:
        raise EmbeddingError("Embedding response is missing 'data'.")

    if len(data) != expected_count:
        raise EmbeddingError(
            f"Embedding response item count ({len(data)}) does not match "
            f"input count ({expected_count})."
        )

    ordered: list[tuple[float, ...] | None] = [None] * expected_count
    seen_indexes: set[int] = set()

    for item in data:
        index = getattr(item, "index", None)
        if index is None or not (0 <= index < expected_count):
            raise EmbeddingError(f"Embedding response contains an invalid index: {index!r}.")
        if index in seen_indexes:
            raise EmbeddingError(f"Embedding response contains a duplicate index: {index}.")
        seen_indexes.add(index)

        embedding = getattr(item, "embedding", None)
        if not embedding:
            raise EmbeddingError(f"Embedding response item at index {index} has an empty vector.")

        ordered[index] = tuple(float(value) for value in embedding)

    if len(seen_indexes) != expected_count:
        missing = sorted(set(range(expected_count)) - seen_indexes)
        raise EmbeddingError(f"Embedding response is missing indexes: {missing}.")

    dimensions = {len(vector) for vector in ordered if vector is not None}
    if len(dimensions) > 1:
        raise EmbeddingError(
            f"Embedding response contains inconsistent vector dimensions: {sorted(dimensions)}."
        )

    return [vector for vector in ordered if vector is not None]


class OpenAIEmbeddingService:
    """Isolates direct calls to the OpenAI embeddings API."""

    def __init__(self, client: Any | None = None, model: str | None = None) -> None:
        """Create the service.

        Args:
            client: An OpenAI-compatible client exposing
                `client.embeddings.create(...)`. Inject a fake client
                in tests. If omitted, a local `.env` file (if any) is
                loaded first, then a real client is built from
                environment configuration (requires `OPENAI_API_KEY`).
            model: Embedding model name. If omitted, uses
                `OPENAI_EMBEDDING_MODEL` from the environment (including
                a local `.env` file when no client is injected), falling
                back to `text-embedding-3-small`.
        """
        if client is None:
            # Load .env before building the client or resolving the
            # model, since both may depend on values it defines
            # (OPENAI_API_KEY and OPENAI_EMBEDDING_MODEL).
            _load_dotenv_once()
            client = _build_default_client()

        self._client = client
        self._model = model or _default_model()

    @property
    def model(self) -> str:
        return self._model

    def embed_texts(self, texts: list[str]) -> list[tuple[float, ...]]:
        """Embed a list of texts, preserving input order.

        Args:
            texts: Texts to embed. Must not be empty or whitespace-only.

        Returns:
            One embedding tuple per input text, in the same order. An
            empty input list returns an empty list without calling the API.

        Raises:
            EmbeddingError: If any text is empty/whitespace-only, the
                API call fails, or the response is invalid.
        """
        if not texts:
            return []

        for text in texts:
            if not text or not text.strip():
                raise EmbeddingError("Cannot embed an empty or whitespace-only text.")

        try:
            response = self._client.embeddings.create(model=self._model, input=texts)
        except Exception as exc:
            raise EmbeddingError(
                f"OpenAI embeddings request failed: {exc.__class__.__name__}"
            ) from exc

        return _reconstruct_embeddings(response, expected_count=len(texts))


def embed_chunks(
    chunks: list[KnowledgeChunk],
    service: OpenAIEmbeddingService,
) -> list[EmbeddedChunk]:
    """Embed a list of chunks with a single embeddings API call.

    Args:
        chunks: Chunks to embed, e.g. from `chunk_knowledge_base()`.
        service: The embedding service to use.

    Returns:
        One `EmbeddedChunk` per input chunk, in the same order. An
        empty input list returns an empty list.

    Raises:
        EmbeddingError: If the service returns a different number of
            vectors than chunks.
    """
    if not chunks:
        return []

    texts = [build_embedding_text(chunk) for chunk in chunks]
    embeddings = service.embed_texts(texts)

    if len(embeddings) != len(chunks):
        raise EmbeddingError(
            f"Embedding service returned {len(embeddings)} vectors for {len(chunks)} chunks."
        )

    return [
        EmbeddedChunk(chunk=chunk, embedding=embedding)
        for chunk, embedding in zip(chunks, embeddings)
    ]
