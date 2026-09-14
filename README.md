# RAG Knowledge Agent

## Problem

Internal company knowledge is often scattered across multiple documents —
pricing, billing, policies, onboarding guides, FAQs — making it hard for
users to find accurate answers quickly. Manually searching through
documentation is slow and error-prone. This project explores how a
Retrieval-Augmented Generation (RAG) assistant can let users ask
natural-language questions and receive answers grounded in a controlled,
internal knowledge base, with visible sources.

## Planned RAG Flow

```
User question
  → document retrieval
  → relevant chunks
  → grounded LLM answer
  → source citations
```

This is the **planned** architecture. As of this iteration, none of these
steps are implemented yet — only the project scaffold exists.

## Planned Tech Stack

- Python 3.13+
- OpenAI API
- Embeddings
- Vector search
- Streamlit
- pytest

## Language

- The user-facing application and the fictional knowledge base will be
  **Spanish-first**.
- The assistant is intended to understand both Spanish and English
  questions.
- All code, comments, and technical documentation remain in **English**.

## Current Status

The knowledge base, document loading, chunking, and OpenAI embedding
integration are implemented: `rag/document_loader.py` loads the
Markdown files in `knowledge/` into `KnowledgeDocument` objects,
`rag/chunker.py` splits them into section-aware `KnowledgeChunk`
objects, and `rag/embeddings.py` turns chunks into `EmbeddedChunk`
objects via the OpenAI embeddings API. Vector search, retrieval, and
RAG answer generation are not implemented yet — embeddings are not
yet stored or searched.

## Project Structure

```
rag-knowledge-agent/
├── app.py                   # Placeholder CLI entry point
├── rag/
│   ├── __init__.py
│   ├── document_loader.py   # Loads Markdown files from knowledge/ into KnowledgeDocument objects
│   ├── chunker.py           # Splits KnowledgeDocuments into section-aware KnowledgeChunk objects
│   ├── embeddings.py        # Turns KnowledgeChunks into EmbeddedChunk objects via the OpenAI API
│   └── retriever.py         # Future semantic retrieval
├── agent/
│   └── __init__.py          # Future grounded RAG agent
├── knowledge/
│   └── .gitkeep              # Will hold the fictional Spanish knowledge base
├── tests/
│   └── __init__.py          # pytest suite (no real OpenAI calls)
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

## Local Setup

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Copy `.env.example` to `.env` when environment variables are needed in a
future iteration. The OpenAI API is not used yet, so no API key is required
at this stage.
