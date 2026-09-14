"""Deterministic text chunking for loaded documents.

This module will split document text into chunks that are small enough
to embed usefully but large enough to preserve context, while retaining
source metadata on each chunk.

Not implemented yet.
"""


def chunk_text(text: str, source_metadata: dict) -> list:
    """Split text into deterministic chunks with preserved metadata.

    Args:
        text: The raw document text to split.
        source_metadata: Metadata identifying the source document.

    Returns:
        A list of chunks, each retaining a reference to its source.

    Raises:
        NotImplementedError: Chunking is not implemented yet.
    """
    raise NotImplementedError("Text chunking is not implemented yet.")
