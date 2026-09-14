"""Tests for rag.embeddings.

Uses a small fake OpenAI client (plain objects exposing
`.embeddings.create(...)`) so these tests never make real API calls,
never require real credentials, and never touch the network.
"""

import dataclasses

import pytest

import rag.embeddings as embeddings_module
from rag.chunker import KnowledgeChunk
from rag.embeddings import (
    DEFAULT_EMBEDDING_MODEL,
    EmbeddedChunk,
    EmbeddingError,
    OpenAIEmbeddingService,
    build_embedding_text,
    embed_chunks,
)


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class FakeEmbeddingItem:
    def __init__(self, index, embedding):
        self.index = index
        self.embedding = embedding


class FakeEmbeddingResponse:
    def __init__(self, data):
        self.data = data


class FakeEmbeddingsResource:
    def __init__(self, response=None, exception=None):
        self._response = response
        self._exception = exception
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self._exception is not None:
            raise self._exception
        return self._response


class FakeOpenAIClient:
    def __init__(self, response=None, exception=None):
        self.embeddings = FakeEmbeddingsResource(response=response, exception=exception)


def make_chunk(
    text: str = "Contenido de ejemplo.",
    chunk_id: str = "doc.md::chunk-001",
    source: str = "doc.md",
    title: str = "Título",
    section: str | None = "Sección",
) -> KnowledgeChunk:
    return KnowledgeChunk(
        chunk_id=chunk_id, source=source, title=title, section=section, text=text
    )


def service_with_response(data, model: str | None = None):
    client = FakeOpenAIClient(response=FakeEmbeddingResponse(data))
    service = OpenAIEmbeddingService(client=client, model=model)
    return service, client


# ---------------------------------------------------------------------------
# EmbeddedChunk model
# ---------------------------------------------------------------------------


def test_embedded_chunk_has_expected_fields() -> None:
    chunk = make_chunk()
    embedded = EmbeddedChunk(chunk=chunk, embedding=(0.1, 0.2, 0.3))

    assert embedded.chunk is chunk
    assert embedded.embedding == (0.1, 0.2, 0.3)


def test_embedded_chunk_is_immutable() -> None:
    embedded = EmbeddedChunk(chunk=make_chunk(), embedding=(0.1,))

    with pytest.raises(dataclasses.FrozenInstanceError):
        embedded.embedding = (0.2,)


def test_embedded_chunk_embedding_is_a_tuple() -> None:
    embedded = EmbeddedChunk(chunk=make_chunk(), embedding=(0.1, 0.2))

    assert isinstance(embedded.embedding, tuple)


# ---------------------------------------------------------------------------
# Embedding input
# ---------------------------------------------------------------------------


def test_embedding_input_includes_title() -> None:
    chunk = make_chunk(title="Planes y precios")

    text = build_embedding_text(chunk)

    assert "Título: Planes y precios" in text


def test_embedding_input_includes_section_when_present() -> None:
    chunk = make_chunk(section="Plan Profesional")

    text = build_embedding_text(chunk)

    assert "Sección: Plan Profesional" in text


def test_embedding_input_omits_section_line_when_none() -> None:
    chunk = make_chunk(section=None)

    text = build_embedding_text(chunk)

    assert "Sección" not in text


def test_embedding_input_includes_chunk_text() -> None:
    chunk = make_chunk(text="El plan Profesional cuesta 20 USD.")

    text = build_embedding_text(chunk)

    assert "El plan Profesional cuesta 20 USD." in text


def test_embedding_input_does_not_include_source_filename() -> None:
    chunk = make_chunk(source="02_planes_y_precios.md")

    text = build_embedding_text(chunk)

    assert "02_planes_y_precios.md" not in text


def test_embedding_input_does_not_include_chunk_id() -> None:
    chunk = make_chunk(chunk_id="02_planes_y_precios.md::chunk-004")

    text = build_embedding_text(chunk)

    assert "chunk-004" not in text
    assert "::" not in text


def test_embedding_input_preserves_spanish_characters() -> None:
    chunk = make_chunk(
        title="Título",
        section="Facturación > Métodos de pago",
        text="áéíóúñ ¿Cómo estás? ¡Hola!",
    )

    text = build_embedding_text(chunk)

    assert "Título" in text
    assert "Facturación > Métodos de pago" in text
    assert "áéíóúñ ¿Cómo estás? ¡Hola!" in text


def test_embedding_input_is_deterministic() -> None:
    chunk = make_chunk()

    assert build_embedding_text(chunk) == build_embedding_text(chunk)


def test_embedding_input_does_not_mutate_chunk() -> None:
    chunk = make_chunk(text="Texto original.")

    build_embedding_text(chunk)

    assert chunk.text == "Texto original."


# ---------------------------------------------------------------------------
# Service configuration
# ---------------------------------------------------------------------------


def test_explicit_model_overrides_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large")
    client = FakeOpenAIClient(response=FakeEmbeddingResponse([]))

    service = OpenAIEmbeddingService(client=client, model="custom-model")

    assert service.model == "custom-model"


def test_environment_model_used_when_explicit_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large")
    client = FakeOpenAIClient(response=FakeEmbeddingResponse([]))

    service = OpenAIEmbeddingService(client=client)

    assert service.model == "text-embedding-3-large"


def test_default_model_when_no_override_exists(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_EMBEDDING_MODEL", raising=False)
    client = FakeOpenAIClient(response=FakeEmbeddingResponse([]))

    service = OpenAIEmbeddingService(client=client)

    assert service.model == DEFAULT_EMBEDDING_MODEL
    assert service.model == "text-embedding-3-small"


def test_default_client_path_loads_dotenv_before_building_client_and_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When no client is injected, a local .env must be loaded before the
    real client is built and before OPENAI_EMBEDDING_MODEL is resolved,
    since both may depend on values it defines."""
    monkeypatch.delenv("OPENAI_EMBEDDING_MODEL", raising=False)
    call_order: list[str] = []
    fake_client = FakeOpenAIClient(response=FakeEmbeddingResponse([]))

    def fake_load_dotenv_once() -> None:
        call_order.append("load_dotenv")
        # Simulate a .env file defining OPENAI_EMBEDDING_MODEL.
        monkeypatch.setenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large")

    def fake_build_default_client():
        call_order.append("build_client")
        return fake_client

    monkeypatch.setattr(embeddings_module, "_load_dotenv_once", fake_load_dotenv_once)
    monkeypatch.setattr(embeddings_module, "_build_default_client", fake_build_default_client)

    service = OpenAIEmbeddingService()  # client omitted: exercises the default-client path

    assert call_order == ["load_dotenv", "build_client"]
    assert service.model == "text-embedding-3-large"


def test_injected_client_never_loads_dotenv(monkeypatch: pytest.MonkeyPatch) -> None:
    call_order: list[str] = []

    def fake_load_dotenv_once() -> None:
        call_order.append("load_dotenv")

    monkeypatch.setattr(embeddings_module, "_load_dotenv_once", fake_load_dotenv_once)

    client = FakeOpenAIClient(response=FakeEmbeddingResponse([]))
    OpenAIEmbeddingService(client=client)

    assert call_order == []


# ---------------------------------------------------------------------------
# embed_texts
# ---------------------------------------------------------------------------


def test_embed_texts_empty_list_returns_empty_without_calling_client() -> None:
    service, client = service_with_response([FakeEmbeddingItem(0, [0.1])])

    result = service.embed_texts([])

    assert result == []
    assert client.embeddings.calls == []


def test_embed_texts_single_text_returns_one_embedding() -> None:
    service, client = service_with_response([FakeEmbeddingItem(0, [0.1, 0.2])])

    result = service.embed_texts(["hola"])

    assert result == [(0.1, 0.2)]
    assert client.embeddings.calls[0]["input"] == ["hola"]


def test_embed_texts_preserves_input_order() -> None:
    data = [
        FakeEmbeddingItem(0, [0.1]),
        FakeEmbeddingItem(1, [0.2]),
        FakeEmbeddingItem(2, [0.3]),
    ]
    service, _ = service_with_response(data)

    result = service.embed_texts(["a", "b", "c"])

    assert result == [(0.1,), (0.2,), (0.3,)]


def test_embed_texts_reorders_out_of_order_response_items() -> None:
    data = [
        FakeEmbeddingItem(2, [0.3]),
        FakeEmbeddingItem(0, [0.1]),
        FakeEmbeddingItem(1, [0.2]),
    ]
    service, _ = service_with_response(data)

    result = service.embed_texts(["a", "b", "c"])

    assert result == [(0.1,), (0.2,), (0.3,)]


def test_embed_texts_converts_embeddings_to_tuples() -> None:
    service, _ = service_with_response([FakeEmbeddingItem(0, [0.1, 0.2, 0.3])])

    result = service.embed_texts(["hola"])

    assert isinstance(result[0], tuple)


def test_embed_texts_rejects_empty_string() -> None:
    service, client = service_with_response([])

    with pytest.raises(EmbeddingError):
        service.embed_texts([""])

    assert client.embeddings.calls == []


def test_embed_texts_rejects_whitespace_only_string() -> None:
    service, client = service_with_response([])

    with pytest.raises(EmbeddingError):
        service.embed_texts(["   \n\t  "])

    assert client.embeddings.calls == []


def test_embed_texts_missing_response_item_raises_error() -> None:
    service, _ = service_with_response([FakeEmbeddingItem(0, [0.1])])

    with pytest.raises(EmbeddingError):
        service.embed_texts(["a", "b"])


def test_embed_texts_duplicate_response_index_raises_error() -> None:
    data = [FakeEmbeddingItem(0, [0.1]), FakeEmbeddingItem(0, [0.2])]
    service, _ = service_with_response(data)

    with pytest.raises(EmbeddingError):
        service.embed_texts(["a", "b"])


def test_embed_texts_out_of_range_index_raises_error() -> None:
    data = [FakeEmbeddingItem(0, [0.1]), FakeEmbeddingItem(5, [0.2])]
    service, _ = service_with_response(data)

    with pytest.raises(EmbeddingError):
        service.embed_texts(["a", "b"])


def test_embed_texts_empty_embedding_vector_raises_error() -> None:
    data = [FakeEmbeddingItem(0, []), FakeEmbeddingItem(1, [0.2])]
    service, _ = service_with_response(data)

    with pytest.raises(EmbeddingError):
        service.embed_texts(["a", "b"])


def test_embed_texts_inconsistent_dimensions_raise_error() -> None:
    data = [FakeEmbeddingItem(0, [0.1, 0.2]), FakeEmbeddingItem(1, [0.3])]
    service, _ = service_with_response(data)

    with pytest.raises(EmbeddingError):
        service.embed_texts(["a", "b"])


def test_embed_texts_wraps_client_exception_in_embedding_error() -> None:
    client = FakeOpenAIClient(exception=RuntimeError("boom"))
    service = OpenAIEmbeddingService(client=client)

    with pytest.raises(EmbeddingError) as exc_info:
        service.embed_texts(["a"])

    assert isinstance(exc_info.value.__cause__, RuntimeError)


def test_embed_texts_wrapped_error_does_not_expose_api_key() -> None:
    secret = "sk-test-super-secret-key-value"
    client = FakeOpenAIClient(exception=RuntimeError(f"Unauthorized: {secret}"))
    service = OpenAIEmbeddingService(client=client)

    with pytest.raises(EmbeddingError) as exc_info:
        service.embed_texts(["a"])

    assert secret not in str(exc_info.value)


# ---------------------------------------------------------------------------
# embed_chunks
# ---------------------------------------------------------------------------


def test_embed_chunks_empty_list_returns_empty_without_calling_client() -> None:
    service, client = service_with_response([])

    result = embed_chunks([], service)

    assert result == []
    assert client.embeddings.calls == []


def test_embed_chunks_uses_a_single_embed_texts_call() -> None:
    chunks = [
        make_chunk(chunk_id=f"doc.md::chunk-{i:03d}", text=f"Texto {i}") for i in range(3)
    ]
    data = [FakeEmbeddingItem(i, [float(i)]) for i in range(3)]
    service, client = service_with_response(data)

    embed_chunks(chunks, service)

    assert len(client.embeddings.calls) == 1
    assert len(client.embeddings.calls[0]["input"]) == 3


def test_embed_chunks_preserves_chunk_order() -> None:
    chunks = [
        make_chunk(chunk_id="doc.md::chunk-001", text="Primero"),
        make_chunk(chunk_id="doc.md::chunk-002", text="Segundo"),
        make_chunk(chunk_id="doc.md::chunk-003", text="Tercero"),
    ]
    data = [FakeEmbeddingItem(i, [float(i)]) for i in range(3)]
    service, _ = service_with_response(data)

    result = embed_chunks(chunks, service)

    assert [item.chunk.chunk_id for item in result] == [
        "doc.md::chunk-001",
        "doc.md::chunk-002",
        "doc.md::chunk-003",
    ]


def test_embed_chunks_preserves_original_chunk_metadata() -> None:
    chunk = make_chunk(
        chunk_id="02_planes_y_precios.md::chunk-004",
        source="02_planes_y_precios.md",
        title="Planes y precios",
        section="Plan Profesional",
        text="El plan Profesional cuesta 20 USD.",
    )
    service, _ = service_with_response([FakeEmbeddingItem(0, [0.1])])

    result = embed_chunks([chunk], service)

    assert result[0].chunk == chunk


def test_embed_chunks_pairs_correct_vector_with_each_chunk() -> None:
    chunks = [
        make_chunk(chunk_id="doc.md::chunk-001"),
        make_chunk(chunk_id="doc.md::chunk-002"),
    ]
    # Response items arrive out of order to prove pairing follows the
    # reconstructed order, not the raw response order.
    data = [FakeEmbeddingItem(1, [0.2]), FakeEmbeddingItem(0, [0.1])]
    service, _ = service_with_response(data)

    result = embed_chunks(chunks, service)

    assert result[0].chunk.chunk_id == "doc.md::chunk-001"
    assert result[0].embedding == (0.1,)
    assert result[1].chunk.chunk_id == "doc.md::chunk-002"
    assert result[1].embedding == (0.2,)


def test_embed_chunks_mismatched_vector_count_raises_error() -> None:
    chunks = [
        make_chunk(chunk_id="doc.md::chunk-001"),
        make_chunk(chunk_id="doc.md::chunk-002"),
    ]

    class ShortService:
        """Minimal fake that violates the embed_texts count contract,
        to exercise embed_chunks' own defensive guard."""

        model = "fake-model"

        def embed_texts(self, texts: list[str]) -> list[tuple[float, ...]]:
            return [(0.1,)]

    with pytest.raises(EmbeddingError):
        embed_chunks(chunks, ShortService())
