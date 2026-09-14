"""Document loading for the RAG knowledge base.

This module will be responsible for reading source documents from the
`knowledge/` directory and extracting the metadata needed downstream
(e.g. source file name, title) to support grounding and citations.

Not implemented yet.
"""

from pathlib import Path


def load_documents(knowledge_dir: Path) -> list:
    """Load documents from the knowledge base directory.

    Args:
        knowledge_dir: Path to the directory containing source documents.

    Returns:
        A list of loaded documents with their metadata.

    Raises:
        NotImplementedError: Document loading is not implemented yet.
    """
    raise NotImplementedError("Document loading is not implemented yet.")
