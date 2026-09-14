"""Embedding generation abstraction.

This module will isolate embedding generation behind a simple
interface so that OpenAI embedding calls are not spread throughout the
codebase, and so retrieval logic can be tested without real API calls.

Not implemented yet.
"""


def embed_text(text: str) -> list:
    """Generate an embedding vector for a piece of text.

    Args:
        text: The text to embed.

    Returns:
        A list of floats representing the embedding vector.

    Raises:
        NotImplementedError: Embedding generation is not implemented yet.
    """
    raise NotImplementedError("Embedding generation is not implemented yet.")
