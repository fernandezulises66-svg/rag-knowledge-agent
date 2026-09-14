"""Tests for agent.rag_agent.

Uses fake retrievers, fake grounded generators, and a fake OpenAI
Responses client (plain objects exposing `.responses.parse(...)`) so
these tests never make real API calls, never require real credentials,
and never touch the network.
"""

import dataclasses

import pytest

from agent.rag_agent import (
    DEFAULT_GENERATION_MODEL,
    INSUFFICIENT_INFORMATION_MESSAGE,
    GroundingDecision,
    OpenAIGroundedGenerator,
    RAGAgent,
    RAGAnswer,
    RAGGenerationError,
    SourceCitation,
    build_citations,
    build_evidence_map,
    build_grounding_input,
    validate_grounding_decision,
)
from rag.chunker import KnowledgeChunk, chunk_knowledge_base
from rag.document_loader import load_knowledge_base
from rag.embeddings import EmbeddedChunk
from rag.retriever import RetrievalResult, SemanticRetriever


# ---------------------------------------------------------------------------
# Fakes and helpers
# ---------------------------------------------------------------------------


class FakeParsedResponse:
    def __init__(self, output_parsed):
        self.output_parsed = output_parsed


class FakeResponsesResource:
    def __init__(self, parsed=None, exception=None):
        self._parsed = parsed
        self._exception = exception
        self.calls: list[dict] = []

    def parse(self, **kwargs):
        self.calls.append(kwargs)
        if self._exception is not None:
            raise self._exception
        return FakeParsedResponse(self._parsed)


class FakeOpenAIClient:
    def __init__(self, parsed=None, exception=None):
        self.responses = FakeResponsesResource(parsed=parsed, exception=exception)


class FakeRetriever:
    def __init__(self, results: list[RetrievalResult] | None = None) -> None:
        self._results = results if results is not None else []
        self.calls: list[tuple[str, int]] = []

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        self.calls.append((query, top_k))
        return list(self._results)


class FakeGenerator:
    def __init__(self, decision: GroundingDecision | None = None) -> None:
        self._decision = decision
        self.calls: list[tuple[str, list[RetrievalResult]]] = []

    def generate(self, question: str, results: list[RetrievalResult]) -> GroundingDecision:
        self.calls.append((question, list(results)))
        return self._decision


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
# SourceCitation / RAGAnswer
# ---------------------------------------------------------------------------


def test_source_citation_has_expected_fields() -> None:
    citation = SourceCitation(source="doc.md", title="Título", section="Sección")

    assert citation.source == "doc.md"
    assert citation.title == "Título"
    assert citation.section == "Sección"


def test_source_citation_is_immutable() -> None:
    citation = SourceCitation(source="doc.md", title="Título", section=None)

    with pytest.raises(dataclasses.FrozenInstanceError):
        citation.source = "other.md"


def test_rag_answer_has_expected_fields() -> None:
    citation = SourceCitation(source="doc.md", title="Título", section="Sección")
    result = make_result()
    answer = RAGAnswer(
        answer="Respuesta.",
        answerable=True,
        citations=(citation,),
        retrieved_results=(result,),
    )

    assert answer.answer == "Respuesta."
    assert answer.answerable is True
    assert answer.citations == (citation,)
    assert answer.retrieved_results == (result,)


def test_rag_answer_is_immutable() -> None:
    answer = RAGAnswer(answer="Respuesta.", answerable=True, citations=(), retrieved_results=())

    with pytest.raises(dataclasses.FrozenInstanceError):
        answer.answerable = False


def test_rag_answer_has_no_chain_of_thought_field() -> None:
    field_names = {field.name for field in dataclasses.fields(RAGAnswer)}

    assert field_names == {"answer", "answerable", "citations", "retrieved_results"}


# ---------------------------------------------------------------------------
# Prompt / evidence construction
# ---------------------------------------------------------------------------


def test_grounding_input_includes_exact_question() -> None:
    question = "¿Puedo pedir un reembolso del plan anual?"
    text = build_grounding_input(question, [make_result()])

    assert question in text


def test_grounding_input_labels_chunks_in_order() -> None:
    results = [
        make_result(chunk_id="a.md::chunk-001", source="a.md"),
        make_result(chunk_id="b.md::chunk-001", source="b.md"),
    ]
    text = build_grounding_input("¿Pregunta?", results)

    assert text.index("[E1]") < text.index("[E2]")
    assert text.index("[E1]") < text.index("a.md")
    assert text.index("[E2]") < text.index("b.md")


def test_grounding_input_includes_source_filename() -> None:
    text = build_grounding_input("¿Pregunta?", [make_result(source="02_planes_y_precios.md")])

    assert "02_planes_y_precios.md" in text


def test_grounding_input_includes_title() -> None:
    text = build_grounding_input("¿Pregunta?", [make_result(title="Planes y precios")])

    assert "Planes y precios" in text


def test_grounding_input_includes_section_when_present() -> None:
    text = build_grounding_input("¿Pregunta?", [make_result(section="Plan Profesional")])

    assert "Sección: Plan Profesional" in text


def test_grounding_input_handles_section_none_clearly() -> None:
    text = build_grounding_input("¿Pregunta?", [make_result(section=None)])

    assert "Sección" not in text
    assert "None" not in text


def test_grounding_input_includes_chunk_body_text() -> None:
    text = build_grounding_input(
        "¿Pregunta?", [make_result(text="El plan Profesional cuesta 20 USD.")]
    )

    assert "El plan Profesional cuesta 20 USD." in text


def test_grounding_input_does_not_include_embedding_values() -> None:
    result = make_result()
    text = build_grounding_input("¿Pregunta?", [result])

    # No numeric embedding-like content is injected; only score-free
    # RetrievalResult/KnowledgeChunk fields are used.
    assert "embedding" not in text.lower()
    assert str(result.score) not in text


def test_grounding_input_does_not_include_absolute_paths() -> None:
    text = build_grounding_input("¿Pregunta?", [make_result(source="02_planes_y_precios.md")])

    assert ":\\" not in text
    assert "/home/" not in text
    assert "C:" not in text


def test_grounding_input_preserves_spanish_characters() -> None:
    text = build_grounding_input(
        "¿Cuál es la política de reembolsos?",
        [
            make_result(
                title="Título",
                section="Facturación > Métodos de pago",
                text="áéíóúñ ¿Cómo estás? ¡Hola!",
            )
        ],
    )

    assert "¿Cuál es la política de reembolsos?" in text
    assert "Facturación > Métodos de pago" in text
    assert "áéíóúñ ¿Cómo estás? ¡Hola!" in text


def test_grounding_input_construction_is_deterministic() -> None:
    results = [make_result(chunk_id="a.md::chunk-001"), make_result(chunk_id="b.md::chunk-001")]

    first = build_grounding_input("¿Pregunta?", results)
    second = build_grounding_input("¿Pregunta?", results)

    assert first == second


# ---------------------------------------------------------------------------
# Generator configuration
# ---------------------------------------------------------------------------


def test_explicit_model_overrides_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_MODEL", "gpt-other")
    client = FakeOpenAIClient(parsed=None)

    generator = OpenAIGroundedGenerator(client=client, model="custom-model")

    assert generator.model == "custom-model"


def test_environment_model_used_when_explicit_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_MODEL", "gpt-other")
    client = FakeOpenAIClient(parsed=None)

    generator = OpenAIGroundedGenerator(client=client)

    assert generator.model == "gpt-other"


def test_default_model_is_gpt_5_6_luna(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    client = FakeOpenAIClient(parsed=None)

    generator = OpenAIGroundedGenerator(client=client)

    assert generator.model == DEFAULT_GENERATION_MODEL
    assert generator.model == "gpt-5.6-luna"


def test_injected_client_does_not_load_dotenv(monkeypatch: pytest.MonkeyPatch) -> None:
    import agent.rag_agent as rag_agent_module

    call_order: list[str] = []
    monkeypatch.setattr(
        rag_agent_module, "_load_dotenv_once", lambda: call_order.append("load_dotenv")
    )

    OpenAIGroundedGenerator(client=FakeOpenAIClient(parsed=None))

    assert call_order == []


def test_default_client_path_loads_dotenv_before_client_and_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import agent.rag_agent as rag_agent_module

    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    call_order: list[str] = []
    fake_client = FakeOpenAIClient(parsed=None)

    def fake_load_dotenv_once() -> None:
        call_order.append("load_dotenv")
        monkeypatch.setenv("OPENAI_MODEL", "gpt-from-dotenv")

    def fake_build_default_client():
        call_order.append("build_client")
        return fake_client

    monkeypatch.setattr(rag_agent_module, "_load_dotenv_once", fake_load_dotenv_once)
    monkeypatch.setattr(rag_agent_module, "_build_default_client", fake_build_default_client)

    generator = OpenAIGroundedGenerator()  # client omitted: default-client path

    assert call_order == ["load_dotenv", "build_client"]
    assert generator.model == "gpt-from-dotenv"


# ---------------------------------------------------------------------------
# Structured generation
# ---------------------------------------------------------------------------


def test_generate_uses_configured_model() -> None:
    decision = GroundingDecision(answerable=True, answer="Respuesta.", evidence_ids=["E1"])
    client = FakeOpenAIClient(parsed=decision)
    generator = OpenAIGroundedGenerator(client=client, model="custom-model")

    generator.generate("¿Pregunta?", [make_result()])

    assert client.responses.calls[0]["model"] == "custom-model"


def test_generate_calls_responses_parse_exactly_once() -> None:
    decision = GroundingDecision(answerable=True, answer="Respuesta.", evidence_ids=["E1"])
    client = FakeOpenAIClient(parsed=decision)
    generator = OpenAIGroundedGenerator(client=client)

    generator.generate("¿Pregunta?", [make_result()])

    assert len(client.responses.calls) == 1


def test_generate_supplies_structured_output_model() -> None:
    decision = GroundingDecision(answerable=True, answer="Respuesta.", evidence_ids=["E1"])
    client = FakeOpenAIClient(parsed=decision)
    generator = OpenAIGroundedGenerator(client=client)

    generator.generate("¿Pregunta?", [make_result()])

    assert client.responses.calls[0]["text_format"] is GroundingDecision


def test_generate_missing_parsed_output_raises_error() -> None:
    client = FakeOpenAIClient(parsed=None)
    generator = OpenAIGroundedGenerator(client=client)

    with pytest.raises(RAGGenerationError):
        generator.generate("¿Pregunta?", [make_result()])


def test_generate_wraps_client_exception() -> None:
    client = FakeOpenAIClient(exception=RuntimeError("boom"))
    generator = OpenAIGroundedGenerator(client=client)

    with pytest.raises(RAGGenerationError) as exc_info:
        generator.generate("¿Pregunta?", [make_result()])

    assert isinstance(exc_info.value.__cause__, RuntimeError)


def test_generate_wrapped_error_does_not_expose_api_key() -> None:
    secret = "sk-test-super-secret-key-value"
    client = FakeOpenAIClient(exception=RuntimeError(f"Unauthorized: {secret}"))
    generator = OpenAIGroundedGenerator(client=client)

    with pytest.raises(RAGGenerationError) as exc_info:
        generator.generate("¿Pregunta?", [make_result()])

    assert secret not in str(exc_info.value)


def test_generate_rejects_empty_question() -> None:
    generator = OpenAIGroundedGenerator(client=FakeOpenAIClient(parsed=None))

    with pytest.raises(RAGGenerationError):
        generator.generate("", [make_result()])


def test_generate_rejects_empty_results() -> None:
    generator = OpenAIGroundedGenerator(client=FakeOpenAIClient(parsed=None))

    with pytest.raises(RAGGenerationError):
        generator.generate("¿Pregunta?", [])


# ---------------------------------------------------------------------------
# Minimal-sufficient-evidence prompt policy (Iteration 10)
#
# These check for focused key phrases, not one large exact paragraph, so
# harmless future wording tweaks won't make them brittle -- each targets
# one specific policy statement.
# ---------------------------------------------------------------------------


def _sent_instructions(client: FakeOpenAIClient) -> str:
    """The instructions sent to the model, with whitespace collapsed so
    assertions are not sensitive to incidental line-wrapping in the
    source text."""
    return " ".join(client.responses.calls[0]["instructions"].split())


def test_instructions_require_direct_concise_answers_without_extra_context() -> None:
    client = FakeOpenAIClient(
        parsed=GroundingDecision(answerable=True, answer="Respuesta.", evidence_ids=["E1"])
    )
    OpenAIGroundedGenerator(client=client).generate("¿Pregunta?", [make_result()])

    instructions = _sent_instructions(client)

    assert "responde de forma directa y concisa" in instructions.lower()
    assert "información de fondo opcional" in instructions.lower()


def test_instructions_require_minimum_sufficient_evidence_set() -> None:
    client = FakeOpenAIClient(
        parsed=GroundingDecision(answerable=True, answer="Respuesta.", evidence_ids=["E1"])
    )
    OpenAIGroundedGenerator(client=client).generate("¿Pregunta?", [make_result()])

    instructions = _sent_instructions(client)

    assert "evidencia mínima suficiente" in instructions.lower()
    assert "conjunto de evidencia más pequeño que sea suficiente" in instructions.lower()


def test_instructions_forbid_citing_merely_related_or_background_evidence() -> None:
    client = FakeOpenAIClient(
        parsed=GroundingDecision(answerable=True, answer="Respuesta.", evidence_ids=["E1"])
    )
    OpenAIGroundedGenerator(client=client).generate("¿Pregunta?", [make_result()])

    instructions = _sent_instructions(client).lower()

    assert "está relacionado con el tema de la pregunta" in instructions
    assert "contexto de fondo opcional" in instructions
    assert "describe el producto de forma general" in instructions
    assert "posición alta en la recuperación" in instructions


def test_instructions_prefer_single_citation_when_sufficient() -> None:
    client = FakeOpenAIClient(
        parsed=GroundingDecision(answerable=True, answer="Respuesta.", evidence_ids=["E1"])
    )
    OpenAIGroundedGenerator(client=client).generate("¿Pregunta?", [make_result()])

    instructions = _sent_instructions(client).lower()

    assert "único elemento de evidencia respalda por completo la respuesta" in instructions
    assert "cita únicamente ese elemento" in instructions


def test_instructions_allow_multiple_evidence_ids_only_when_necessary() -> None:
    client = FakeOpenAIClient(
        parsed=GroundingDecision(answerable=True, answer="Respuesta.", evidence_ids=["E1"])
    )
    OpenAIGroundedGenerator(client=client).generate("¿Pregunta?", [make_result()])

    instructions = _sent_instructions(client).lower()

    assert "varios evidence_ids solo cuando" in instructions
    assert "realmente necesarios" in instructions


def test_instructions_still_forbid_external_knowledge_and_inference() -> None:
    """Preserve pre-existing safeguards: this iteration must not weaken them."""
    client = FakeOpenAIClient(
        parsed=GroundingDecision(answerable=True, answer="Respuesta.", evidence_ids=["E1"])
    )
    OpenAIGroundedGenerator(client=client).generate("¿Pregunta?", [make_result()])

    instructions = _sent_instructions(client).lower()

    assert "no utilices conocimiento externo" in instructions
    assert "nunca infieras" in instructions
    assert "ausencia de evidencia en una respuesta factual" in instructions
    assert "responde en español de forma predeterminada" in instructions
    assert "respuesta en inglés" in instructions


# ---------------------------------------------------------------------------
# Grounding validation
# ---------------------------------------------------------------------------


def test_validate_answerable_requires_non_empty_answer() -> None:
    evidence_map = build_evidence_map([make_result()])
    decision = GroundingDecision(answerable=True, answer="   ", evidence_ids=["E1"])

    with pytest.raises(RAGGenerationError):
        validate_grounding_decision(decision, evidence_map)


def test_validate_answerable_requires_at_least_one_evidence_id() -> None:
    evidence_map = build_evidence_map([make_result()])
    decision = GroundingDecision(answerable=True, answer="Respuesta.", evidence_ids=[])

    with pytest.raises(RAGGenerationError):
        validate_grounding_decision(decision, evidence_map)


def test_validate_unknown_evidence_id_raises_error() -> None:
    evidence_map = build_evidence_map([make_result()])
    decision = GroundingDecision(answerable=True, answer="Respuesta.", evidence_ids=["E99"])

    with pytest.raises(RAGGenerationError):
        validate_grounding_decision(decision, evidence_map)


def test_validate_duplicate_evidence_id_is_rejected() -> None:
    evidence_map = build_evidence_map([make_result(), make_result(chunk_id="doc.md::chunk-002")])
    decision = GroundingDecision(answerable=True, answer="Respuesta.", evidence_ids=["E1", "E1"])

    with pytest.raises(RAGGenerationError):
        validate_grounding_decision(decision, evidence_map)


def test_validate_unsupported_with_no_evidence_ids_is_valid() -> None:
    evidence_map = build_evidence_map([make_result()])
    decision = GroundingDecision(answerable=False, answer="", evidence_ids=[])

    validate_grounding_decision(decision, evidence_map)  # must not raise


def test_validate_unsupported_with_evidence_ids_raises_error() -> None:
    evidence_map = build_evidence_map([make_result()])
    decision = GroundingDecision(answerable=False, answer="", evidence_ids=["E1"])

    with pytest.raises(RAGGenerationError):
        validate_grounding_decision(decision, evidence_map)


def test_unsupported_free_form_answer_is_not_exposed_publicly() -> None:
    result = make_result()
    decision = GroundingDecision(
        answerable=False, answer="Nubira no acepta criptomonedas.", evidence_ids=[]
    )
    retriever = FakeRetriever(results=[result])
    generator = FakeGenerator(decision=decision)
    agent = RAGAgent(retriever=retriever, generator=generator)

    answer = agent.answer("¿Nubira acepta pagos con criptomonedas?")

    assert answer.answer == INSUFFICIENT_INFORMATION_MESSAGE
    assert "criptomonedas" not in answer.answer


def test_build_citations_maps_evidence_ids_to_real_chunk_metadata() -> None:
    result = make_result(
        chunk_id="04_cancelaciones_y_reembolsos.md::chunk-004",
        source="04_cancelaciones_y_reembolsos.md",
        title="Cancelaciones y reembolsos",
        section="Elegibilidad de reembolso",
    )
    evidence_map = build_evidence_map([result])

    citations = build_citations(["E1"], evidence_map)

    assert citations == (
        SourceCitation(
            source="04_cancelaciones_y_reembolsos.md",
            title="Cancelaciones y reembolsos",
            section="Elegibilidad de reembolso",
        ),
    )


def test_build_citations_preserves_evidence_id_order() -> None:
    first = make_result(chunk_id="a.md::chunk-001", source="a.md", title="A", section="Sec A")
    second = make_result(chunk_id="b.md::chunk-001", source="b.md", title="B", section="Sec B")
    evidence_map = build_evidence_map([first, second])

    citations = build_citations(["E2", "E1"], evidence_map)

    assert [citation.source for citation in citations] == ["b.md", "a.md"]


def test_build_citations_deduplicates_preserving_first_occurrence() -> None:
    first = make_result(chunk_id="a.md::chunk-001", source="a.md", title="A", section="Sec A")
    duplicate = make_result(chunk_id="a.md::chunk-001", source="a.md", title="A", section="Sec A")
    third = make_result(chunk_id="b.md::chunk-001", source="b.md", title="B", section="Sec B")
    evidence_map = build_evidence_map([first, duplicate, third])

    citations = build_citations(["E1", "E2", "E3"], evidence_map)

    assert [citation.source for citation in citations] == ["a.md", "b.md"]


# ---------------------------------------------------------------------------
# RAGAgent orchestration
# ---------------------------------------------------------------------------


def _make_agent(results=None, decision=None):
    retriever = FakeRetriever(results=results)
    generator = FakeGenerator(decision=decision)
    return RAGAgent(retriever=retriever, generator=generator), retriever, generator


def test_agent_rejects_empty_question() -> None:
    agent, _, _ = _make_agent(results=[make_result()])

    with pytest.raises(RAGGenerationError):
        agent.answer("")


def test_agent_rejects_whitespace_only_question() -> None:
    agent, _, _ = _make_agent(results=[make_result()])

    with pytest.raises(RAGGenerationError):
        agent.answer("   \n\t  ")


def test_agent_rejects_invalid_top_k() -> None:
    agent, _, _ = _make_agent(results=[make_result()])

    with pytest.raises(ValueError):
        agent.answer("¿Pregunta?", top_k=0)


def test_agent_calls_retriever_exactly_once() -> None:
    decision = GroundingDecision(answerable=True, answer="Respuesta.", evidence_ids=["E1"])
    agent, retriever, _ = _make_agent(results=[make_result()], decision=decision)

    agent.answer("¿Pregunta?")

    assert len(retriever.calls) == 1


def test_agent_passes_exact_question_to_retriever() -> None:
    decision = GroundingDecision(answerable=True, answer="Respuesta.", evidence_ids=["E1"])
    agent, retriever, _ = _make_agent(results=[make_result()], decision=decision)
    question = "¿Puedo pedir un reembolso del plan anual?"

    agent.answer(question)

    assert retriever.calls[0][0] == question


def test_agent_passes_requested_top_k_to_retriever() -> None:
    decision = GroundingDecision(answerable=True, answer="Respuesta.", evidence_ids=["E1"])
    agent, retriever, _ = _make_agent(results=[make_result()], decision=decision)

    agent.answer("¿Pregunta?", top_k=7)

    assert retriever.calls[0][1] == 7


def test_agent_empty_retrieval_returns_fallback_without_calling_generator() -> None:
    agent, retriever, generator = _make_agent(results=[])

    answer = agent.answer("¿Pregunta?")

    assert answer.answer == INSUFFICIENT_INFORMATION_MESSAGE
    assert answer.answerable is False
    assert answer.citations == ()
    assert answer.retrieved_results == ()
    assert generator.calls == []


def test_agent_non_empty_retrieval_calls_generator_exactly_once() -> None:
    decision = GroundingDecision(answerable=True, answer="Respuesta.", evidence_ids=["E1"])
    agent, _, generator = _make_agent(results=[make_result()], decision=decision)

    agent.answer("¿Pregunta?")

    assert len(generator.calls) == 1


def test_agent_passes_retrieved_results_unchanged_to_generator() -> None:
    decision = GroundingDecision(answerable=True, answer="Respuesta.", evidence_ids=["E1"])
    results = [make_result(chunk_id="a.md::chunk-001"), make_result(chunk_id="b.md::chunk-001")]
    agent, _, generator = _make_agent(results=results, decision=decision)

    agent.answer("¿Pregunta?")

    assert generator.calls[0][1] == results


def test_agent_supported_decision_returns_model_answer() -> None:
    decision = GroundingDecision(
        answerable=True, answer="El reembolso aplica dentro de 14 días.", evidence_ids=["E1"]
    )
    agent, _, _ = _make_agent(results=[make_result()], decision=decision)

    answer = agent.answer("¿Pregunta?")

    assert answer.answer == "El reembolso aplica dentro de 14 días."
    assert answer.answerable is True


def test_agent_supported_decision_returns_deterministic_citations() -> None:
    result = make_result(
        chunk_id="04_cancelaciones_y_reembolsos.md::chunk-004",
        source="04_cancelaciones_y_reembolsos.md",
        title="Cancelaciones y reembolsos",
        section="Elegibilidad de reembolso",
    )
    decision = GroundingDecision(answerable=True, answer="Respuesta.", evidence_ids=["E1"])
    agent, _, _ = _make_agent(results=[result], decision=decision)

    answer = agent.answer("¿Pregunta?")

    assert answer.citations == (
        SourceCitation(
            source="04_cancelaciones_y_reembolsos.md",
            title="Cancelaciones y reembolsos",
            section="Elegibilidad de reembolso",
        ),
    )


def test_agent_unsupported_decision_returns_controlled_fallback() -> None:
    decision = GroundingDecision(answerable=False, answer="", evidence_ids=[])
    agent, _, _ = _make_agent(results=[make_result()], decision=decision)

    answer = agent.answer("¿Pregunta?")

    assert answer.answer == INSUFFICIENT_INFORMATION_MESSAGE
    assert answer.answerable is False


def test_agent_unsupported_decision_returns_no_citations() -> None:
    decision = GroundingDecision(answerable=False, answer="", evidence_ids=[])
    agent, _, _ = _make_agent(results=[make_result()], decision=decision)

    answer = agent.answer("¿Pregunta?")

    assert answer.citations == ()


def test_agent_preserves_retrieved_results_in_public_result() -> None:
    result = make_result()
    decision = GroundingDecision(answerable=True, answer="Respuesta.", evidence_ids=["E1"])
    agent, _, _ = _make_agent(results=[result], decision=decision)

    answer = agent.answer("¿Pregunta?")

    assert answer.retrieved_results == (result,)


def test_agent_inconsistent_decision_raises_error_via_validation() -> None:
    decision = GroundingDecision(answerable=False, answer="", evidence_ids=["E1"])
    agent, _, _ = _make_agent(results=[make_result()], decision=decision)

    with pytest.raises(RAGGenerationError):
        agent.answer("¿Pregunta?")


def test_minimal_evidence_scenario_maps_to_single_citation() -> None:
    """Not a proof of model quality: only verifies that when a decision
    correctly cites just the one evidence item that actually supports the
    fact (E1) and ignores generic background evidence (E2), the pipeline
    maps that decision to exactly one SourceCitation."""
    specific_result = make_result(
        chunk_id="05_cuentas_y_acceso.md::chunk-005",
        source="05_cuentas_y_acceso.md",
        title="Cuentas y acceso",
        section="Recuperación de contraseña",
        text="El enlace de restablecimiento es válido durante 1 hora.",
    )
    background_result = make_result(
        chunk_id="09_preguntas_frecuentes.md::chunk-002",
        source="09_preguntas_frecuentes.md",
        title="Preguntas frecuentes",
        section="¿Qué es Nubira?",
        text="Nubira es una plataforma de gestión de proyectos.",
    )
    decision = GroundingDecision(
        answerable=True,
        answer="El enlace de restablecimiento es válido durante 1 hora.",
        evidence_ids=["E1"],
    )
    agent, _, generator = _make_agent(
        results=[specific_result, background_result], decision=decision
    )

    answer = agent.answer("¿Cuánto tiempo dura el enlace de restablecimiento de contraseña?")

    assert answer.citations == (
        SourceCitation(
            source="05_cuentas_y_acceso.md",
            title="Cuentas y acceso",
            section="Recuperación de contraseña",
        ),
    )
    # Both results were still passed to the generator -- it is the
    # (faked) decision, not any Python pruning, that limits citations.
    assert generator.calls[0][1] == [specific_result, background_result]


# ---------------------------------------------------------------------------
# Real knowledge-base integration check (no OpenAI)
# ---------------------------------------------------------------------------


class FakeEmbeddingServiceForQuery:
    """No OpenAI client, no env vars, no network -- deterministic only."""

    def __init__(self, vectors: list[tuple[float, ...]]) -> None:
        self._vectors = vectors
        self.calls: list[list[str]] = []

    def embed_texts(self, texts: list[str]) -> list[tuple[float, ...]]:
        self.calls.append(list(texts))
        return list(self._vectors)


def test_real_knowledge_base_agent_orchestration_integration() -> None:
    documents = load_knowledge_base("knowledge")
    chunks = chunk_knowledge_base(documents)
    assert chunks

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

    embedding_service = FakeEmbeddingServiceForQuery(vectors=[(1.0, 2.0, 3.0)])
    retriever = SemanticRetriever(
        embedded_chunks=embedded_chunks, embedding_service=embedding_service
    )

    # The retriever is deterministic for a fixed query vector, so the
    # expected top result can be computed independently for assertion.
    expected_top_chunk = retriever.retrieve("¿Cuál es la política de reembolsos?", top_k=5)[0].chunk

    decision = GroundingDecision(
        answerable=True,
        answer="Respuesta de prueba basada en la evidencia recuperada.",
        evidence_ids=["E1"],
    )
    generator = FakeGenerator(decision=decision)

    agent = RAGAgent(retriever=retriever, generator=generator)

    answer = agent.answer("¿Cuál es la política de reembolsos?", top_k=5)

    assert isinstance(answer, RAGAnswer)
    assert answer.answerable is True
    assert answer.answer == "Respuesta de prueba basada en la evidencia recuperada."
    assert len(answer.retrieved_results) == 5
    assert answer.citations == (
        SourceCitation(
            source=expected_top_chunk.source,
            title=expected_top_chunk.title,
            section=expected_top_chunk.section,
        ),
    )
