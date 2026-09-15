"""Spanish-first Streamlit interface for the RAG Knowledge Agent.

Wraps the existing `RAGAgent` (the `rag.*` pipeline plus
`agent.rag_agent`) in a simple, observable UI: ask a question, see the
grounded answer, its citations, and the retrieved evidence behind it.
This module adds no new retrieval, chunking, embedding, or grounding
behavior -- it only presents the existing public `RAGAnswer`.

Importing this module must never build the RAG pipeline or call
OpenAI; that only happens inside `main()` (via `build_rag_agent()`),
never at import time. This keeps `pytest` (and any other importer)
completely free of network access or real API calls.
"""

import streamlit as st

from agent.rag_agent import OpenAIGroundedGenerator, RAGAgent, RAGAnswer, SourceCitation
from rag.chunker import chunk_knowledge_base
from rag.document_loader import load_knowledge_base
from rag.embeddings import OpenAIEmbeddingService, embed_chunks
from rag.retriever import RetrievalResult, SemanticRetriever

PAGE_TITLE = "Asistente de Conocimiento con RAG"
TOP_K = 5
MAX_HISTORY_ENTRIES = 6

EXAMPLE_QUESTIONS: tuple[str, ...] = (
    "¿Cuánto cuesta el plan Inicial?",
    "¿Qué métodos de pago acepta Nubira?",
    "¿Puedo pedir un reembolso del plan anual?",
    "¿Cuánto dura un enlace para restablecer la contraseña?",
    "¿Nubira acepta pagos con criptomonedas?",
    "Respondeme en inglés: ¿qué métodos de pago acepta Nubira?",
)

SIMILARITY_NOTE = (
    "Una similitud alta indica cercanía semántica, no garantiza que la "
    "documentación contenga una respuesta suficiente."
)

_GENERIC_ERROR_MESSAGE = (
    "No se pudo inicializar el asistente en este momento. Verifica la "
    "configuración (por ejemplo, la clave de la API de OpenAI) e inténtalo "
    "de nuevo."
)


# ---------------------------------------------------------------------------
# Pure presentation helpers (unit-testable without Streamlit or OpenAI)
# ---------------------------------------------------------------------------


def is_question_valid(question: str) -> bool:
    """Whether a submitted question is non-empty/non-whitespace-only."""
    return bool(question and question.strip())


def citation_view(citation: SourceCitation) -> dict:
    """Convert a `SourceCitation` into a plain presentation dict.

    Exposes only `title`, `section`, and `source` -- never an evidence
    ID (E1/E2/...), which is internal to a single generation request
    and is not part of `SourceCitation` in the first place.
    """
    return {
        "title": citation.title,
        "section": citation.section,
        "source": citation.source,
    }


def truncate_preview(text: str, max_chars: int = 200) -> str:
    """Truncate chunk text into a short, deterministic UI preview.

    Collapses internal whitespace (so a multi-line chunk body, e.g. a
    bullet list, reads as one short line) and appends an ellipsis only
    when truncation actually occurred.
    """
    collapsed = " ".join(text.split())
    if len(collapsed) <= max_chars:
        return collapsed
    return collapsed[:max_chars].rstrip() + "…"


def retrieval_detail_view(results: list[RetrievalResult] | tuple[RetrievalResult, ...]) -> list[dict]:
    """Convert retrieved results into plain, rank-ordered presentation dicts.

    Rounds the similarity score for display only -- `RetrievalResult`
    is an immutable dataclass and its `score` is never mutated. Never
    includes the embedding vector.
    """
    return [
        {
            "rank": rank,
            "source": result.chunk.source,
            "title": result.chunk.title,
            "section": result.chunk.section,
            "score": round(result.score, 3),
            "preview": truncate_preview(result.chunk.text),
        }
        for rank, result in enumerate(results, start=1)
    ]


def map_error_to_user_message(exc: Exception) -> str:
    """Map any pipeline-construction exception to a safe Spanish message.

    Never includes the exception's own text, which could otherwise
    echo an API key, header, or other sensitive configuration detail.
    """
    return _GENERIC_ERROR_MESSAGE


def recent_history_entries(history: list[dict], limit: int = MAX_HISTORY_ENTRIES) -> list[dict]:
    """Return up to the most recent `limit` history entries, newest first.

    `history` is expected in the order entries were appended (oldest
    first); this only slices and reorders it for display -- it never
    mutates `history` and never feeds anything back into the RAGAgent
    (this history is presentation-only, not conversational memory).
    """
    return list(reversed(history[-limit:] if limit > 0 else []))


# ---------------------------------------------------------------------------
# RAG pipeline construction (never called at import time)
# ---------------------------------------------------------------------------


@st.cache_resource(show_spinner="Preparando la base de conocimiento...")
def build_rag_agent() -> RAGAgent:
    """Build the full RAG pipeline once per Streamlit app process.

    Loads and chunks the knowledge base, embeds it (one real OpenAI
    embeddings request for the whole corpus), and wires up the
    retriever and grounded generator. Cached via `st.cache_resource` so
    this expensive setup runs once per process rather than on every
    Streamlit rerun. Only the pipeline/corpus is cached -- individual
    user answers are never cached here.
    """
    documents = load_knowledge_base("knowledge")
    chunks = chunk_knowledge_base(documents)

    embedding_service = OpenAIEmbeddingService()
    embedded_chunks = embed_chunks(chunks, embedding_service)

    retriever = SemanticRetriever(
        embedded_chunks=embedded_chunks, embedding_service=embedding_service
    )
    generator = OpenAIGroundedGenerator()

    return RAGAgent(retriever=retriever, generator=generator)


# ---------------------------------------------------------------------------
# UI rendering
# ---------------------------------------------------------------------------


def _render_sidebar() -> None:
    with st.sidebar:
        st.header("Sobre este proyecto")
        st.markdown(
            "**Nubira** es una empresa SaaS ficticia (solo para este "
            "portafolio). Este asistente demuestra:\n"
            "- RAG con recuperación semántica\n"
            "- Generación fundamentada en evidencia\n"
            "- Validación de citas\n"
            "- Manejo de preguntas sin soporte documental\n"
            "- Evaluación con benchmarks propios"
        )

        st.header("Tecnologías")
        st.markdown(
            "- Python\n"
            "- API de OpenAI\n"
            "- Embeddings de OpenAI\n"
            "- Responses API (salida estructurada)\n"
            "- Streamlit\n"
            "- pytest"
        )

        st.header("Arquitectura")
        st.markdown(
            "Documentos\n"
            "→ Fragmentación (chunking)\n"
            "→ Embeddings\n"
            "→ Recuperación (retrieval)\n"
            "→ Generación fundamentada\n"
            "→ Citas"
        )


def _render_examples() -> None:
    st.caption("Ejemplos de preguntas:")
    for example in EXAMPLE_QUESTIONS:
        st.markdown(f"- {example}")


def _render_citations(citations: tuple[SourceCitation, ...]) -> None:
    st.markdown("### Fuentes")
    for citation in citations:
        view = citation_view(citation)
        with st.container(border=True):
            if view["section"] is not None:
                st.markdown(f"**{view['title']}** — {view['section']}")
            else:
                st.markdown(f"**{view['title']}**")
            st.caption(f"`{view['source']}`")


def _render_retrieval_details(results: tuple[RetrievalResult, ...]) -> None:
    with st.expander("Ver detalles de recuperación"):
        st.caption(SIMILARITY_NOTE)
        for item in retrieval_detail_view(results):
            st.markdown(f"**{item['rank']}. `{item['source']}`**")
            if item["section"] is not None:
                st.markdown(f"- Sección: {item['section']}")
            st.markdown(f"- Similitud (semántica): {item['score']}")
            st.markdown(f"- Vista previa: {item['preview']}")


def _render_answer(rag_answer: RAGAnswer) -> None:
    if rag_answer.answerable:
        st.markdown("### Respuesta")
        st.write(rag_answer.answer)
        if rag_answer.citations:
            _render_citations(rag_answer.citations)
    else:
        st.warning(rag_answer.answer)

    if rag_answer.retrieved_results:
        _render_retrieval_details(rag_answer.retrieved_results)


def _render_history() -> None:
    history = st.session_state.get("history", [])
    if not history:
        return

    st.markdown("### Consultas anteriores")
    if st.button("Limpiar historial"):
        st.session_state["history"] = []
        st.rerun()

    for entry in recent_history_entries(history):
        with st.expander(entry["question"], expanded=False):
            _render_answer(entry["rag_answer"])


def main() -> None:
    st.set_page_config(
        page_title=PAGE_TITLE,
        page_icon="📚",
        layout="centered",
    )

    st.title(PAGE_TITLE)
    st.markdown(
        "**Nubira** es una empresa SaaS ficticia creada únicamente para "
        "este proyecto de portafolio. Este asistente responde preguntas "
        "usando documentación interna de Nubira: las respuestas están "
        "fundamentadas en evidencia recuperada, y las preguntas sin "
        "soporte documental reciben una respuesta explícita indicando "
        "que no hay información suficiente."
    )
    st.caption("Todos los datos y la documentación son ficticios.")

    _render_sidebar()

    if "history" not in st.session_state:
        st.session_state["history"] = []

    try:
        agent = build_rag_agent()
    except Exception as exc:  # noqa: BLE001 - mapped to a safe message below
        st.error(map_error_to_user_message(exc))
        with st.expander("Detalles técnicos (avanzado)"):
            st.caption(exc.__class__.__name__)
        st.stop()
        return

    _render_examples()

    with st.form("question_form"):
        question = st.text_input(
            "Escribe tu pregunta",
            placeholder="¿Puedo pedir un reembolso de mi plan anual?",
        )
        submitted = st.form_submit_button("Consultar")

    if submitted:
        if not is_question_valid(question):
            st.warning("Escribe una pregunta antes de consultar.")
        else:
            with st.spinner("Consultando la documentación..."):
                rag_answer = agent.answer(question, top_k=TOP_K)

            _render_answer(rag_answer)

            st.session_state["history"].append({"question": question, "rag_answer": rag_answer})

    _render_history()


if __name__ == "__main__":
    main()
