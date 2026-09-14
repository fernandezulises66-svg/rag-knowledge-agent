"""Document loading for the RAG knowledge base.

Loads Markdown source documents from the `knowledge/` directory into
structured `KnowledgeDocument` objects, extracting the H1 title while
preserving the full document text and a portable source file name for
later chunking, retrieval, and citation.
"""

from dataclasses import dataclass
from pathlib import Path


class DocumentLoadError(Exception):
    """Raised when a knowledge-base document or directory cannot be loaded."""


@dataclass(frozen=True)
class KnowledgeDocument:
    """A single loaded knowledge-base document."""

    source: str
    title: str
    text: str


def _extract_title(text: str, source_name: str) -> str:
    """Return the title from the first valid Markdown H1 (`# Title`) in text."""
    for line in text.splitlines():
        if line.startswith("# "):
            title = line[2:].strip()
            if not title:
                raise DocumentLoadError(f"H1 heading is empty in document: {source_name}")
            return title
    raise DocumentLoadError(f"No H1 heading ('# Title') found in document: {source_name}")


def load_markdown_document(path: str | Path) -> KnowledgeDocument:
    """Load a single Markdown knowledge-base document.

    Args:
        path: Path to a `.md` file, as a string or `Path`.

    Returns:
        The loaded document with its source file name, extracted H1
        title, and full text.

    Raises:
        DocumentLoadError: If the path is missing, is not a regular
            `.md` file, is not valid UTF-8, or has no valid H1 title.
    """
    path = Path(path)

    if not path.exists():
        raise DocumentLoadError(f"Path does not exist: {path}")
    if not path.is_file():
        raise DocumentLoadError(f"Path is not a regular file: {path}")
    if path.suffix.lower() != ".md":
        raise DocumentLoadError(f"Not a Markdown (.md) file: {path}")

    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise DocumentLoadError(f"File is not valid UTF-8: {path}") from exc

    title = _extract_title(text, path.name)

    return KnowledgeDocument(source=path.name, title=title, text=text)


def load_knowledge_base(directory: str | Path = "knowledge") -> list[KnowledgeDocument]:
    """Load every Markdown document directly inside a knowledge-base directory.

    Subdirectories, hidden files, and non-`.md` files are ignored. This
    does not recurse into subdirectories.

    Args:
        directory: Path to the knowledge-base directory.

    Returns:
        Documents sorted by file name, for deterministic ordering. An
        empty directory returns an empty list.

    Raises:
        DocumentLoadError: If the directory does not exist or is not a
            directory.
    """
    directory = Path(directory)

    if not directory.exists():
        raise DocumentLoadError(f"Knowledge base directory does not exist: {directory}")
    if not directory.is_dir():
        raise DocumentLoadError(f"Knowledge base path is not a directory: {directory}")

    markdown_paths = sorted(
        (
            entry
            for entry in directory.iterdir()
            if entry.is_file()
            and not entry.name.startswith(".")
            and entry.suffix.lower() == ".md"
        ),
        key=lambda entry: entry.name,
    )

    return [load_markdown_document(entry) for entry in markdown_paths]
