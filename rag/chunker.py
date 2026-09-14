"""Deterministic Markdown-aware chunking for the RAG knowledge base.

Splits a `KnowledgeDocument` into `KnowledgeChunk` objects along H2/H3
section boundaries, packing consecutive blank-line-separated Markdown
blocks (paragraphs, lists, etc.) into chunks up to a maximum character
budget. Chunking is deterministic: the same document and `max_chars`
always produce identical chunks and chunk IDs.
"""

from dataclasses import dataclass

from rag.document_loader import KnowledgeDocument


@dataclass(frozen=True)
class KnowledgeChunk:
    """A single chunk of a knowledge-base document, ready for embedding."""

    chunk_id: str
    source: str
    title: str
    section: str | None
    text: str


def _validate_max_chars(max_chars: int) -> None:
    if max_chars <= 0:
        raise ValueError("max_chars must be greater than zero")


def _iter_sections(text: str) -> list[tuple[str | None, list[str]]]:
    """Group document lines by H2/H3 section, skipping the H1 title line.

    Returns (section_label, body_lines) pairs in document order. Body
    content before the first H2 is labeled `None`. Content under an H3
    nested in an H2 is labeled "H2 > H3".
    """
    sections: list[tuple[str | None, list[str]]] = []
    current_label: str | None = None
    current_lines: list[str] = []
    current_h2: str | None = None
    h1_skipped = False

    def flush() -> None:
        sections.append((current_label, current_lines))

    for line in text.splitlines():
        if not h1_skipped and line.startswith("# "):
            h1_skipped = True
            continue

        if line.startswith("### "):
            flush()
            heading = line[4:].strip()
            current_label = f"{current_h2} > {heading}" if current_h2 else heading
            current_lines = []
            continue

        if line.startswith("## "):
            flush()
            current_h2 = line[3:].strip()
            current_label = current_h2
            current_lines = []
            continue

        current_lines.append(line)

    flush()
    return sections


def _split_blocks(lines: list[str]) -> list[str]:
    """Group lines into blank-line-separated Markdown blocks, in order."""
    blocks: list[str] = []
    current: list[str] = []
    for line in lines:
        if line.strip() == "":
            if current:
                blocks.append("\n".join(current))
                current = []
        else:
            current.append(line)
    if current:
        blocks.append("\n".join(current))
    return blocks


def _split_by_character(token: str, max_chars: int) -> list[str]:
    """Hard-split a single uninterrupted token into max_chars-sized fragments.

    This is the last-resort boundary, used only for a token that by
    itself exceeds max_chars and therefore cannot be kept whole.
    """
    return [token[i : i + max_chars] for i in range(0, len(token), max_chars)]


def _split_long_line(line: str, max_chars: int) -> list[str]:
    """Split a single line exceeding max_chars at whitespace/word boundaries.

    Falls back to a hard character split only for an individual word
    that by itself is longer than max_chars.
    """
    pieces: list[str] = []
    current = ""

    for word in line.split(" "):
        word_fragments = (
            _split_by_character(word, max_chars) if len(word) > max_chars else [word]
        )
        for fragment in word_fragments:
            candidate = f"{current} {fragment}" if current else fragment
            if len(candidate) <= max_chars:
                current = candidate
            else:
                if current:
                    pieces.append(current)
                current = fragment

    if current:
        pieces.append(current)

    return pieces


def _split_oversized_block(block: str, max_chars: int) -> list[str]:
    """Split a block longer than max_chars into ordered, size-bounded pieces.

    Prefers existing newline boundaries first (so Markdown list/paragraph
    line breaks are kept whenever the resulting piece still fits), then
    whitespace/word boundaries for any individual line that is still too
    long, and finally a hard character split for a single uninterrupted
    token that by itself exceeds max_chars.
    """
    line_pieces: list[str] = []
    for line in block.split("\n"):
        if len(line) <= max_chars:
            line_pieces.append(line)
        else:
            line_pieces.extend(_split_long_line(line, max_chars))

    pieces: list[str] = []
    current = ""
    for line_piece in line_pieces:
        candidate = f"{current}\n{line_piece}" if current else line_piece
        if len(candidate) <= max_chars:
            current = candidate
        else:
            if current:
                pieces.append(current)
            current = line_piece

    if current:
        pieces.append(current)

    return pieces if pieces else [block[:max_chars]]


def _pack_blocks(blocks: list[str], max_chars: int) -> list[str]:
    """Pack blocks into chunks of at most max_chars, splitting oversized blocks."""
    pieces: list[str] = []
    for block in blocks:
        if len(block) <= max_chars:
            pieces.append(block)
        else:
            pieces.extend(_split_oversized_block(block, max_chars))

    packed: list[str] = []
    current: list[str] = []
    for piece in pieces:
        candidate = current + [piece]
        candidate_text = "\n\n".join(candidate)
        if current and len(candidate_text) > max_chars:
            packed.append("\n\n".join(current))
            current = [piece]
        else:
            current = candidate

    if current:
        packed.append("\n\n".join(current))

    return packed


def chunk_document(
    document: KnowledgeDocument,
    max_chars: int = 1200,
) -> list[KnowledgeChunk]:
    """Split a knowledge-base document into deterministic, section-aware chunks.

    Args:
        document: The loaded document to chunk.
        max_chars: Maximum characters per chunk. Must be greater than zero.

    Returns:
        Chunks in document order, numbered from 1 for this document. A
        document with no meaningful body content returns an empty list.

    Raises:
        ValueError: If `max_chars` is not greater than zero.
    """
    _validate_max_chars(max_chars)

    chunks: list[KnowledgeChunk] = []
    index = 1

    for section, lines in _iter_sections(document.text):
        for chunk_text in _pack_blocks(_split_blocks(lines), max_chars):
            chunks.append(
                KnowledgeChunk(
                    chunk_id=f"{document.source}::chunk-{index:03d}",
                    source=document.source,
                    title=document.title,
                    section=section,
                    text=chunk_text,
                )
            )
            index += 1

    return chunks


def chunk_knowledge_base(
    documents: list[KnowledgeDocument],
    max_chars: int = 1200,
) -> list[KnowledgeChunk]:
    """Chunk every document in order, restarting chunk numbering per document.

    Args:
        documents: Loaded documents, e.g. from `load_knowledge_base()`.
        max_chars: Maximum characters per chunk, passed to `chunk_document()`.

    Returns:
        All chunks, grouped by document in input order. An empty input
        list returns an empty list.

    Raises:
        ValueError: If `max_chars` is not greater than zero.
    """
    _validate_max_chars(max_chars)

    chunks: list[KnowledgeChunk] = []
    for document in documents:
        chunks.extend(chunk_document(document, max_chars=max_chars))
    return chunks
