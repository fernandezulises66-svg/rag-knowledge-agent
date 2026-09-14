"""Semantic retrieval over embedded document chunks.

This module will perform vector search to find the chunks most
relevant to a user's question, to be used as grounding context for the
RAG agent.

Not implemented yet.
"""


def retrieve_relevant_chunks(query: str, top_k: int = 5) -> list:
    """Retrieve the most relevant chunks for a query.

    Args:
        query: The user's natural-language question.
        top_k: The maximum number of chunks to return.

    Returns:
        A list of the most relevant chunks with their source metadata.

    Raises:
        NotImplementedError: Retrieval is not implemented yet.
    """
    raise NotImplementedError("Retrieval is not implemented yet.")
