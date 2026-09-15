"""Tests for streamlit_app.py.

Importing `streamlit_app` must never build the real RAG pipeline or
call OpenAI -- pipeline construction lives behind `main()` /
`build_rag_agent()`, never at module import time. These tests only
exercise the pure presentation helpers; they never call
`build_rag_agent()`, never touch Streamlit's runtime/session machinery,
and never make network or OpenAI calls.
"""

from rag.chunker import KnowledgeChunk
from rag.retriever import RetrievalResult
from streamlit_app import (
    citation_view,
    is_question_valid,
    map_error_to_user_message,
    recent_history_entries,
    retrieval_detail_view,
    truncate_preview,
)
from agent.rag_agent import SourceCitation


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


def make_result(score: float = 0.9, **chunk_kwargs) -> RetrievalResult:
    return RetrievalResult(chunk=make_chunk(**chunk_kwargs), score=score)


# ---------------------------------------------------------------------------
# Module import safety
# ---------------------------------------------------------------------------


def test_import_does_not_build_pipeline_or_call_openai() -> None:
    import streamlit_app

    # If import had built the pipeline, it would have required a real
    # OpenAI client/API key and this import would already have failed
    # or made a network call. Reaching this point with the functions
    # present (but not invoked) is the actual safety property tested.
    assert callable(streamlit_app.build_rag_agent)
    assert callable(streamlit_app.main)


# ---------------------------------------------------------------------------
# is_question_valid
# ---------------------------------------------------------------------------


def test_is_question_valid_accepts_normal_question() -> None:
    assert is_question_valid("¿Cuánto cuesta el plan Inicial?") is True


def test_is_question_valid_rejects_empty_string() -> None:
    assert is_question_valid("") is False


def test_is_question_valid_rejects_whitespace_only() -> None:
    assert is_question_valid("   \n\t  ") is False


# ---------------------------------------------------------------------------
# citation_view
# ---------------------------------------------------------------------------


def test_citation_view_preserves_title_section_source() -> None:
    citation = SourceCitation(
        source="04_cancelaciones_y_reembolsos.md",
        title="Cancelaciones y reembolsos en Nubira",
        section="Elegibilidad de reembolso",
    )

    view = citation_view(citation)

    assert view["title"] == "Cancelaciones y reembolsos en Nubira"
    assert view["section"] == "Elegibilidad de reembolso"
    assert view["source"] == "04_cancelaciones_y_reembolsos.md"


def test_citation_view_handles_section_none() -> None:
    citation = SourceCitation(source="doc.md", title="Título", section=None)

    view = citation_view(citation)

    assert view["section"] is None


def test_citation_view_does_not_expose_evidence_ids() -> None:
    citation = SourceCitation(source="doc.md", title="Título", section="Sección")

    view = citation_view(citation)

    assert set(view.keys()) == {"title", "section", "source"}
    assert "E1" not in view.values()
    assert not any(key.lower() in ("evidence_id", "evidence_ids") for key in view)


def test_citation_view_preserves_spanish_characters() -> None:
    citation = SourceCitation(
        source="doc.md", title="Facturación y pagos", section="Métodos de pago aceptados"
    )

    view = citation_view(citation)

    assert view["title"] == "Facturación y pagos"
    assert view["section"] == "Métodos de pago aceptados"


# ---------------------------------------------------------------------------
# truncate_preview
# ---------------------------------------------------------------------------


def test_truncate_preview_truncates_long_text() -> None:
    long_text = "palabra " * 100

    preview = truncate_preview(long_text, max_chars=50)

    assert len(preview) <= 51  # 50 chars + ellipsis
    assert preview.endswith("…")


def test_truncate_preview_leaves_short_text_unchanged() -> None:
    short_text = "El plan Inicial cuesta 10 USD por usuario al mes."

    preview = truncate_preview(short_text, max_chars=200)

    assert preview == short_text


def test_truncate_preview_preserves_spanish_characters() -> None:
    text = "áéíóúñ ¿Cómo estás? ¡Hola!"

    preview = truncate_preview(text, max_chars=200)

    assert preview == text


def test_truncate_preview_is_deterministic() -> None:
    text = "Texto de prueba " * 20

    first = truncate_preview(text, max_chars=80)
    second = truncate_preview(text, max_chars=80)

    assert first == second


# ---------------------------------------------------------------------------
# retrieval_detail_view
# ---------------------------------------------------------------------------


def test_retrieval_detail_view_preserves_expected_fields() -> None:
    result = make_result(
        score=0.6789,
        source="04_cancelaciones_y_reembolsos.md",
        title="Cancelaciones y reembolsos en Nubira",
        section="Elegibilidad de reembolso",
        text="Nubira ofrece una garantía de reembolso completo...",
    )

    view = retrieval_detail_view([result])

    assert view[0]["rank"] == 1
    assert view[0]["source"] == "04_cancelaciones_y_reembolsos.md"
    assert view[0]["title"] == "Cancelaciones y reembolsos en Nubira"
    assert view[0]["section"] == "Elegibilidad de reembolso"
    assert view[0]["score"] == 0.679
    assert "Nubira ofrece" in view[0]["preview"]


def test_retrieval_detail_view_ranks_in_order() -> None:
    results = [make_result(chunk_id=f"doc.md::chunk-{i:03d}", score=1.0 - i / 10) for i in range(3)]

    view = retrieval_detail_view(results)

    assert [item["rank"] for item in view] == [1, 2, 3]


def test_retrieval_detail_view_rounds_score_without_mutating_original() -> None:
    result = make_result(score=0.123456789)

    view = retrieval_detail_view([result])

    assert view[0]["score"] == 0.123
    assert result.score == 0.123456789  # RetrievalResult itself is untouched


def test_retrieval_detail_view_does_not_expose_embeddings() -> None:
    result = make_result()

    view = retrieval_detail_view([result])

    assert "embedding" not in view[0]
    assert not any("embedding" in key.lower() for key in view[0])


def test_retrieval_detail_view_does_not_expose_evidence_ids() -> None:
    result = make_result()

    view = retrieval_detail_view([result])

    assert set(view[0].keys()) == {"rank", "source", "title", "section", "score", "preview"}


def test_retrieval_detail_view_has_no_chain_of_thought_field() -> None:
    result = make_result()

    view = retrieval_detail_view([result])

    forbidden = {"reasoning", "chain_of_thought", "thought", "rationale"}
    assert not (set(view[0].keys()) & forbidden)


def test_retrieval_detail_view_is_deterministic() -> None:
    results = [make_result(score=0.5), make_result(chunk_id="doc.md::chunk-002", score=0.4)]

    first = retrieval_detail_view(results)
    second = retrieval_detail_view(results)

    assert first == second


# ---------------------------------------------------------------------------
# Answerable vs. unsupported citation/retrieval availability
# ---------------------------------------------------------------------------


def test_unsupported_answer_has_no_citation_data_to_render() -> None:
    from agent.rag_agent import INSUFFICIENT_INFORMATION_MESSAGE, RAGAnswer

    result = make_result()
    rag_answer = RAGAnswer(
        answer=INSUFFICIENT_INFORMATION_MESSAGE,
        answerable=False,
        citations=(),
        retrieved_results=(result,),
    )

    citation_views = [citation_view(c) for c in rag_answer.citations]

    assert citation_views == []


def test_supported_answer_produces_citation_data() -> None:
    from agent.rag_agent import RAGAnswer

    citation = SourceCitation(source="doc.md", title="Título", section="Sección")
    rag_answer = RAGAnswer(
        answer="Respuesta.", answerable=True, citations=(citation,), retrieved_results=()
    )

    citation_views = [citation_view(c) for c in rag_answer.citations]

    assert citation_views == [{"title": "Título", "section": "Sección", "source": "doc.md"}]


def test_retrieval_results_remain_available_when_unsupported() -> None:
    from agent.rag_agent import INSUFFICIENT_INFORMATION_MESSAGE, RAGAnswer

    result = make_result(source="03_facturacion_y_pagos.md")
    rag_answer = RAGAnswer(
        answer=INSUFFICIENT_INFORMATION_MESSAGE,
        answerable=False,
        citations=(),
        retrieved_results=(result,),
    )

    view = retrieval_detail_view(rag_answer.retrieved_results)

    assert len(view) == 1
    assert view[0]["source"] == "03_facturacion_y_pagos.md"


# ---------------------------------------------------------------------------
# map_error_to_user_message
# ---------------------------------------------------------------------------


def test_error_mapping_returns_clean_spanish_message() -> None:
    message = map_error_to_user_message(RuntimeError("boom"))

    assert message.strip() != ""
    assert "asistente" in message.lower()
    assert "configuraci" in message.lower()


def test_error_mapping_does_not_expose_secret_from_exception() -> None:
    secret = "sk-test-super-secret-key-value"
    exc = RuntimeError(f"Unauthorized: {secret}")

    message = map_error_to_user_message(exc)

    assert secret not in message


def test_error_mapping_is_deterministic() -> None:
    exc = ValueError("some internal detail")

    first = map_error_to_user_message(exc)
    second = map_error_to_user_message(exc)

    assert first == second


# ---------------------------------------------------------------------------
# recent_history_entries
# ---------------------------------------------------------------------------


def _history_entry(label: str) -> dict:
    return {"question": label, "rag_answer": None}


def test_recent_history_entries_preserves_newest_first_order() -> None:
    history = [_history_entry("q1"), _history_entry("q2"), _history_entry("q3")]

    entries = recent_history_entries(history)

    assert [entry["question"] for entry in entries] == ["q3", "q2", "q1"]


def test_recent_history_entries_limits_to_default_max() -> None:
    history = [_history_entry(f"q{i}") for i in range(10)]

    entries = recent_history_entries(history)

    assert len(entries) == 6
    assert [entry["question"] for entry in entries] == ["q9", "q8", "q7", "q6", "q5", "q4"]


def test_recent_history_entries_respects_custom_limit() -> None:
    history = [_history_entry(f"q{i}") for i in range(10)]

    entries = recent_history_entries(history, limit=2)

    assert [entry["question"] for entry in entries] == ["q9", "q8"]


def test_recent_history_entries_handles_fewer_than_limit() -> None:
    history = [_history_entry("q1"), _history_entry("q2")]

    entries = recent_history_entries(history)

    assert [entry["question"] for entry in entries] == ["q2", "q1"]


def test_recent_history_entries_handles_empty_history() -> None:
    assert recent_history_entries([]) == []


def test_recent_history_entries_does_not_mutate_original_list() -> None:
    history = [_history_entry("q1"), _history_entry("q2"), _history_entry("q3")]
    original_order = list(history)

    recent_history_entries(history)

    assert history == original_order


def test_recent_history_entries_is_deterministic() -> None:
    history = [_history_entry(f"q{i}") for i in range(8)]

    first = recent_history_entries(history)
    second = recent_history_entries(history)

    assert first == second
