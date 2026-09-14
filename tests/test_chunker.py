"""Tests for rag.chunker.

Synthetic KnowledgeDocument objects are constructed directly for
chunking-behavior tests; only the final integration test touches the
real repository knowledge base.
"""

import dataclasses

import pytest

from rag.chunker import KnowledgeChunk, chunk_document, chunk_knowledge_base
from rag.document_loader import KnowledgeDocument, load_knowledge_base


def make_document(text: str, source: str = "doc.md", title: str = "Titulo") -> KnowledgeDocument:
    return KnowledgeDocument(source=source, title=title, text=text)


# ---------------------------------------------------------------------------
# KnowledgeChunk model
# ---------------------------------------------------------------------------


def test_knowledge_chunk_has_expected_fields() -> None:
    chunk = KnowledgeChunk(
        chunk_id="doc.md::chunk-001",
        source="doc.md",
        title="Título",
        section="Sección",
        text="Contenido.",
    )

    assert chunk.chunk_id == "doc.md::chunk-001"
    assert chunk.source == "doc.md"
    assert chunk.title == "Título"
    assert chunk.section == "Sección"
    assert chunk.text == "Contenido."


def test_knowledge_chunk_is_immutable() -> None:
    chunk = KnowledgeChunk(
        chunk_id="doc.md::chunk-001",
        source="doc.md",
        title="Título",
        section=None,
        text="Contenido.",
    )

    with pytest.raises(dataclasses.FrozenInstanceError):
        chunk.text = "Otro contenido."


# ---------------------------------------------------------------------------
# Basic chunking
# ---------------------------------------------------------------------------


def test_short_document_body_produces_one_chunk() -> None:
    document = make_document("# Título\n\nUn párrafo corto.\n")

    chunks = chunk_document(document)

    assert len(chunks) == 1
    assert chunks[0].text == "Un párrafo corto."


def test_metadata_preserves_source_and_title() -> None:
    document = make_document(
        "# Planes y precios\n\nTexto.\n",
        source="02_planes_y_precios.md",
        title="Planes y precios",
    )

    chunks = chunk_document(document)

    assert chunks[0].source == "02_planes_y_precios.md"
    assert chunks[0].title == "Planes y precios"


def test_chunk_id_format_is_deterministic() -> None:
    document = make_document(
        "# Título\n\nParte uno.\n\n## Sección\n\nParte dos.\n",
        source="02_planes_y_precios.md",
    )

    chunks = chunk_document(document)

    assert chunks[0].chunk_id == "02_planes_y_precios.md::chunk-001"
    assert chunks[1].chunk_id == "02_planes_y_precios.md::chunk-002"


def test_repeated_chunking_is_identical() -> None:
    document = make_document("# Título\n\nParte uno.\n\n## Sección\n\nParte dos.\n")

    first = chunk_document(document)
    second = chunk_document(document)

    assert first == second


def test_spanish_characters_are_preserved() -> None:
    document = make_document("# Título\n\náéíóúñ ¿Cómo estás? ¡Hola!\n")

    chunks = chunk_document(document)

    assert chunks[0].text == "áéíóúñ ¿Cómo estás? ¡Hola!"


# ---------------------------------------------------------------------------
# Section handling
# ---------------------------------------------------------------------------


def test_content_before_first_h2_has_no_section() -> None:
    document = make_document("# Título\n\nIntroducción sin sección.\n\n## Primera\n\nTexto.\n")

    chunks = chunk_document(document)

    assert chunks[0].section is None
    assert chunks[0].text == "Introducción sin sección."


def test_h2_content_gets_correct_section_name() -> None:
    document = make_document("# Título\n\n## Facturación\n\nTexto de facturación.\n")

    chunks = chunk_document(document)

    assert chunks[0].section == "Facturación"


def test_h3_content_uses_hierarchical_section_name() -> None:
    document = make_document(
        "# Título\n\n## Facturación\n\n### Medios de pago\n\nTarjeta y PayPal.\n"
    )

    chunks = chunk_document(document)

    assert chunks[0].section == "Facturación > Medios de pago"


def test_different_sections_are_never_merged() -> None:
    document = make_document(
        "# Título\n\n## Facturación\n\nTexto corto.\n\n## Reembolsos\n\nOtro texto corto.\n"
    )

    chunks = chunk_document(document, max_chars=1200)

    assert len(chunks) == 2
    assert chunks[0].section == "Facturación"
    assert chunks[1].section == "Reembolsos"


def test_empty_sections_produce_no_chunks() -> None:
    document = make_document("# Título\n\n## Vacía\n\n## Con contenido\n\nTexto real.\n")

    chunks = chunk_document(document)

    assert len(chunks) == 1
    assert chunks[0].section == "Con contenido"


def test_h1_is_not_included_as_body_content() -> None:
    document = make_document("# Título del documento\n\nCuerpo real.\n")

    chunks = chunk_document(document)

    assert "Título del documento" not in chunks[0].text


# ---------------------------------------------------------------------------
# Paragraph packing
# ---------------------------------------------------------------------------


def test_multiple_short_paragraphs_are_packed_into_one_chunk() -> None:
    document = make_document("# Título\n\nPárrafo uno.\n\nPárrafo dos.\n\nPárrafo tres.\n")

    chunks = chunk_document(document, max_chars=1200)

    assert len(chunks) == 1
    assert chunks[0].text == "Párrafo uno.\n\nPárrafo dos.\n\nPárrafo tres."


def test_packed_blocks_separated_by_single_blank_line() -> None:
    document = make_document("# Título\n\nUno.\n\nDos.\n")

    chunks = chunk_document(document, max_chars=1200)

    assert "\n\n" in chunks[0].text
    assert "\n\n\n" not in chunks[0].text


def test_blocks_preserve_original_order() -> None:
    document = make_document("# Título\n\nPrimero.\n\nSegundo.\n\nTercero.\n")

    chunks = chunk_document(document, max_chars=1200)

    text = chunks[0].text
    assert text.index("Primero") < text.index("Segundo") < text.index("Tercero")


def test_content_exceeding_max_chars_starts_new_chunk() -> None:
    block_a = "A" * 50
    block_b = "B" * 50
    document = make_document(f"# Título\n\n{block_a}\n\n{block_b}\n")

    chunks = chunk_document(document, max_chars=60)

    assert len(chunks) == 2
    assert chunks[0].text == block_a
    assert chunks[1].text == block_b


# ---------------------------------------------------------------------------
# Oversized content
# ---------------------------------------------------------------------------


def test_long_block_is_split_deterministically() -> None:
    long_paragraph = " ".join(f"palabra{i}" for i in range(200))
    document = make_document(f"# Título\n\n{long_paragraph}\n")

    first = chunk_document(document, max_chars=100)
    second = chunk_document(document, max_chars=100)

    assert len(first) > 1
    assert first == second


def test_split_fragments_preserve_original_order() -> None:
    long_paragraph = " ".join(f"palabra{i:03d}" for i in range(200))
    document = make_document(f"# Título\n\n{long_paragraph}\n")

    chunks = chunk_document(document, max_chars=100)

    combined = " ".join(chunk.text for chunk in chunks)
    assert combined.split() == long_paragraph.split()


def test_no_chunk_exceeds_max_chars() -> None:
    long_paragraph = " ".join(f"palabra{i}" for i in range(300))
    document = make_document(f"# Título\n\n{long_paragraph}\n")

    chunks = chunk_document(document, max_chars=80)

    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk.text) <= 80


def test_no_empty_chunks_are_produced() -> None:
    long_paragraph = " ".join(f"palabra{i}" for i in range(300))
    document = make_document(f"# Título\n\n{long_paragraph}\n")

    chunks = chunk_document(document, max_chars=80)

    for chunk in chunks:
        assert chunk.text.strip() != ""


def test_oversized_bullet_list_preserves_lines_and_order() -> None:
    items = [f"- Elemento número {i} de la lista de ejemplo." for i in range(30)]
    bullet_list = "\n".join(items)
    document = make_document(f"# Título\n\n{bullet_list}\n")

    chunks = chunk_document(document, max_chars=150)

    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk.text) <= 150

    reconstructed = "\n".join(chunk.text for chunk in chunks)
    reconstructed_lines = [line for line in reconstructed.split("\n") if line != ""]
    assert reconstructed_lines == items


def test_oversized_numbered_list_preserves_lines_and_order() -> None:
    items = [f"{i + 1}. Paso número {i + 1} del procedimiento de ejemplo." for i in range(25)]
    numbered_list = "\n".join(items)
    document = make_document(f"# Título\n\n{numbered_list}\n")

    chunks = chunk_document(document, max_chars=150)

    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk.text) <= 150

    reconstructed = "\n".join(chunk.text for chunk in chunks)
    reconstructed_lines = [line for line in reconstructed.split("\n") if line != ""]
    assert reconstructed_lines == items


def test_uninterrupted_long_token_is_split_safely() -> None:
    document = make_document("# Título\n\n" + "x" * 500 + "\n")

    chunks = chunk_document(document, max_chars=100)

    assert len(chunks) >= 5
    for chunk in chunks:
        assert len(chunk.text) <= 100
        assert chunk.text != ""
    assert "".join(chunk.text for chunk in chunks) == "x" * 500


# ---------------------------------------------------------------------------
# Validation / collection behavior
# ---------------------------------------------------------------------------


def test_max_chars_zero_raises_value_error() -> None:
    document = make_document("# Título\n\nTexto.\n")

    with pytest.raises(ValueError):
        chunk_document(document, max_chars=0)


def test_max_chars_negative_raises_value_error() -> None:
    document = make_document("# Título\n\nTexto.\n")

    with pytest.raises(ValueError):
        chunk_document(document, max_chars=-10)


def test_document_with_no_body_content_returns_empty_list() -> None:
    document = make_document("# Título\n")

    chunks = chunk_document(document)

    assert chunks == []


def test_chunk_knowledge_base_empty_list_returns_empty_list() -> None:
    assert chunk_knowledge_base([]) == []


def test_chunk_knowledge_base_preserves_document_order() -> None:
    doc_a = make_document("# A\n\nContenido A.\n", source="a.md", title="A")
    doc_b = make_document("# B\n\nContenido B.\n", source="b.md", title="B")

    chunks = chunk_knowledge_base([doc_a, doc_b])

    assert [chunk.source for chunk in chunks] == ["a.md", "b.md"]


def test_chunk_ids_restart_per_document() -> None:
    doc_a = make_document("# A\n\nContenido A.\n", source="a.md", title="A")
    doc_b = make_document("# B\n\nContenido B.\n", source="b.md", title="B")

    chunks = chunk_knowledge_base([doc_a, doc_b])

    assert chunks[0].chunk_id == "a.md::chunk-001"
    assert chunks[1].chunk_id == "b.md::chunk-001"


# ---------------------------------------------------------------------------
# Real knowledge-base integration check
# ---------------------------------------------------------------------------


def test_real_knowledge_base_chunks_successfully() -> None:
    documents = load_knowledge_base("knowledge")
    assert len(documents) == 9

    chunks = chunk_knowledge_base(documents)
    assert chunks

    chunk_counts_by_source: dict[str, int] = {}
    for chunk in chunks:
        chunk_counts_by_source[chunk.source] = chunk_counts_by_source.get(chunk.source, 0) + 1

    document_sources = [document.source for document in documents]
    for source in document_sources:
        assert chunk_counts_by_source.get(source, 0) >= 1

    chunk_ids = [chunk.chunk_id for chunk in chunks]
    assert len(chunk_ids) == len(set(chunk_ids))

    for chunk in chunks:
        assert chunk.chunk_id
        assert chunk.source
        assert chunk.title
        assert chunk.text
        assert len(chunk.text) <= 1200

    sources_in_first_seen_order: list[str] = []
    for chunk in chunks:
        if chunk.source not in sources_in_first_seen_order:
            sources_in_first_seen_order.append(chunk.source)
    assert sources_in_first_seen_order == document_sources

    combined_text = " ".join(chunk.text for chunk in chunks)
    assert any(character in combined_text for character in "áéíóúñ¿¡")
