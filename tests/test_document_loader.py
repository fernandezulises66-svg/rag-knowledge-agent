"""Tests for rag.document_loader.

Malformed and edge-case documents are created under pytest's tmp_path
fixture so the real knowledge base is never modified by these tests.
"""

from pathlib import Path

import pytest

from rag.document_loader import (
    DocumentLoadError,
    KnowledgeDocument,
    load_knowledge_base,
    load_markdown_document,
)


# ---------------------------------------------------------------------------
# Single-document loading
# ---------------------------------------------------------------------------


def test_loads_valid_utf8_markdown_document(tmp_path: Path) -> None:
    doc_path = tmp_path / "doc.md"
    doc_path.write_text("# Título de prueba\n\nContenido de ejemplo.\n", encoding="utf-8")

    document = load_markdown_document(doc_path)

    assert isinstance(document, KnowledgeDocument)


def test_extracts_h1_title(tmp_path: Path) -> None:
    doc_path = tmp_path / "doc.md"
    doc_path.write_text("# Planes y precios\n\nTexto.\n", encoding="utf-8")

    document = load_markdown_document(doc_path)

    assert document.title == "Planes y precios"


def test_preserves_spanish_characters(tmp_path: Path) -> None:
    content = "# Título con acentos\n\náéíóúñ ¿Cómo estás? ¡Hola!\n"
    doc_path = tmp_path / "doc.md"
    doc_path.write_text(content, encoding="utf-8")

    document = load_markdown_document(doc_path)

    assert "áéíóúñ" in document.text
    assert "¿Cómo estás?" in document.text
    assert "¡Hola!" in document.text
    assert "Título con acentos" in document.text


def test_preserves_full_markdown_text(tmp_path: Path) -> None:
    content = "# Título\n\n## Sección\n\n- item uno\n- item dos\n\n> una cita\n"
    doc_path = tmp_path / "doc.md"
    doc_path.write_text(content, encoding="utf-8")

    document = load_markdown_document(doc_path)

    assert document.text == content


def test_source_is_filename_only_not_absolute_path(tmp_path: Path) -> None:
    doc_path = tmp_path / "02_planes_y_precios.md"
    doc_path.write_text("# Planes y precios\n", encoding="utf-8")

    document = load_markdown_document(doc_path)

    assert document.source == "02_planes_y_precios.md"
    assert str(tmp_path) not in document.source


def test_allows_blank_lines_before_h1(tmp_path: Path) -> None:
    doc_path = tmp_path / "doc.md"
    doc_path.write_text("\n\n   \n# Título tras líneas en blanco\n\nTexto.\n", encoding="utf-8")

    document = load_markdown_document(doc_path)

    assert document.title == "Título tras líneas en blanco"


def test_accepts_str_path_and_path_object(tmp_path: Path) -> None:
    doc_path = tmp_path / "doc.md"
    doc_path.write_text("# Título\n", encoding="utf-8")

    from_str = load_markdown_document(str(doc_path))
    from_path = load_markdown_document(doc_path)

    assert from_str.title == "Título"
    assert from_path.title == "Título"


# ---------------------------------------------------------------------------
# Validation errors - load_markdown_document
# ---------------------------------------------------------------------------


def test_missing_file_raises_error(tmp_path: Path) -> None:
    missing_path = tmp_path / "does_not_exist.md"

    with pytest.raises(DocumentLoadError):
        load_markdown_document(missing_path)


def test_directory_path_raises_error(tmp_path: Path) -> None:
    with pytest.raises(DocumentLoadError):
        load_markdown_document(tmp_path)


def test_non_markdown_file_raises_error(tmp_path: Path) -> None:
    doc_path = tmp_path / "doc.txt"
    doc_path.write_text("# Título\n", encoding="utf-8")

    with pytest.raises(DocumentLoadError):
        load_markdown_document(doc_path)


def test_missing_h1_raises_error(tmp_path: Path) -> None:
    doc_path = tmp_path / "doc.md"
    doc_path.write_text("Solo texto, sin encabezado alguno.\n", encoding="utf-8")

    with pytest.raises(DocumentLoadError):
        load_markdown_document(doc_path)


def test_only_h2_headings_do_not_count_as_h1(tmp_path: Path) -> None:
    doc_path = tmp_path / "doc.md"
    doc_path.write_text("## No es un H1\n\n### Tampoco esto\n", encoding="utf-8")

    with pytest.raises(DocumentLoadError):
        load_markdown_document(doc_path)


def test_empty_h1_is_rejected(tmp_path: Path) -> None:
    doc_path = tmp_path / "doc.md"
    doc_path.write_text("# \n\nTexto después de un H1 vacío.\n", encoding="utf-8")

    with pytest.raises(DocumentLoadError):
        load_markdown_document(doc_path)


def test_invalid_utf8_raises_error(tmp_path: Path) -> None:
    doc_path = tmp_path / "doc.md"
    doc_path.write_bytes(b"# T\xfftulo inv\xe1lido\n")

    with pytest.raises(DocumentLoadError):
        load_markdown_document(doc_path)


# ---------------------------------------------------------------------------
# Knowledge-base loading
# ---------------------------------------------------------------------------


def test_loads_multiple_markdown_documents(tmp_path: Path) -> None:
    (tmp_path / "a.md").write_text("# A\n", encoding="utf-8")
    (tmp_path / "b.md").write_text("# B\n", encoding="utf-8")

    documents = load_knowledge_base(tmp_path)

    assert len(documents) == 2


def test_results_sorted_deterministically_by_filename(tmp_path: Path) -> None:
    (tmp_path / "b.md").write_text("# B\n", encoding="utf-8")
    (tmp_path / "a.md").write_text("# A\n", encoding="utf-8")
    (tmp_path / "c.md").write_text("# C\n", encoding="utf-8")

    documents = load_knowledge_base(tmp_path)

    assert [doc.source for doc in documents] == ["a.md", "b.md", "c.md"]


def test_ignores_non_markdown_files(tmp_path: Path) -> None:
    (tmp_path / "a.md").write_text("# A\n", encoding="utf-8")
    (tmp_path / "notes.txt").write_text("no es markdown", encoding="utf-8")

    documents = load_knowledge_base(tmp_path)

    assert [doc.source for doc in documents] == ["a.md"]


def test_ignores_hidden_files(tmp_path: Path) -> None:
    (tmp_path / "a.md").write_text("# A\n", encoding="utf-8")
    (tmp_path / ".gitkeep").write_text("", encoding="utf-8")
    (tmp_path / ".hidden.md").write_text("# Oculto\n", encoding="utf-8")

    documents = load_knowledge_base(tmp_path)

    assert [doc.source for doc in documents] == ["a.md"]


def test_ignores_subdirectories_without_recursing(tmp_path: Path) -> None:
    (tmp_path / "a.md").write_text("# A\n", encoding="utf-8")
    subdir = tmp_path / "subdir"
    subdir.mkdir()
    (subdir / "nested.md").write_text("# Nested\n", encoding="utf-8")

    documents = load_knowledge_base(tmp_path)

    assert [doc.source for doc in documents] == ["a.md"]


def test_empty_directory_returns_empty_list(tmp_path: Path) -> None:
    documents = load_knowledge_base(tmp_path)

    assert documents == []


def test_missing_knowledge_directory_raises_error(tmp_path: Path) -> None:
    missing_dir = tmp_path / "does_not_exist"

    with pytest.raises(DocumentLoadError):
        load_knowledge_base(missing_dir)


def test_directory_argument_that_is_a_file_raises_error(tmp_path: Path) -> None:
    file_path = tmp_path / "not_a_directory.md"
    file_path.write_text("# Título\n", encoding="utf-8")

    with pytest.raises(DocumentLoadError):
        load_knowledge_base(file_path)


# ---------------------------------------------------------------------------
# Repository knowledge-base integration check
# ---------------------------------------------------------------------------


def test_real_knowledge_base_loads_successfully() -> None:
    documents = load_knowledge_base("knowledge")

    assert len(documents) == 9

    for document in documents:
        assert document.source
        assert document.title
        assert document.text

    sources = [document.source for document in documents]
    assert sources == sorted(sources)
    assert sources[0] == "01_descripcion_producto.md"
    assert sources[-1] == "09_preguntas_frecuentes.md"
