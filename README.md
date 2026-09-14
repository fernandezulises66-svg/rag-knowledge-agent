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

Document loading, deterministic Markdown-aware chunking, OpenAI
multilingual embeddings, cosine-similarity semantic retrieval, and a
transparent retrieval-evaluation framework (`evals/`) are implemented.
The real retrieval benchmark reports **Hit@1 = 100%, Hit@3 = 100%,
MRR@5 = 1.00** over 16 curated answerable queries, with 4 unsupported
queries tracked separately as diagnostics. These metrics evaluate
**retrieval only** — whether the right evidence is found — not final
answer quality.

Grounded answer generation is also implemented: `agent/rag_agent.py`
retrieves evidence with `SemanticRetriever`, asks the generation model
(via the OpenAI Responses API, structured Pydantic output) whether
that evidence is sufficient, independently validates the model's
decision in Python, and returns deterministic citations built only
from real chunk metadata. Unsupported questions — including ones where
retrieval returns topically related, high-scoring but insufficient
evidence — receive an explicit, controlled fallback message instead of
a model-generated guess.

A second, separate evaluation framework (also under `evals/`) checks
the FINAL grounded answers — not just retrieval — using transparent,
rule-based checks: is the question correctly marked answerable, does
the answer mention the actually-documented facts (via accent/case-
insensitive alternative-text matching), is every returned citation a
genuinely valid evidence location, and do unsupported questions get
exactly the controlled refusal with zero citations. No LLM judge and
no semantic quality score are used. The real answer evaluation has not
been run yet, so no answer-quality scores are reported here, and no
claim of zero hallucinations or perfect RAG accuracy is made.

## Project Structure

```
rag-knowledge-agent/
├── app.py                   # Placeholder CLI entry point
├── rag/
│   ├── __init__.py
│   ├── document_loader.py   # Loads Markdown files from knowledge/ into KnowledgeDocument objects
│   ├── chunker.py           # Splits KnowledgeDocuments into section-aware KnowledgeChunk objects
│   ├── embeddings.py        # Turns KnowledgeChunks into EmbeddedChunk objects via the OpenAI API
│   └── retriever.py         # Ranks EmbeddedChunks against a query via SemanticRetriever
├── agent/
│   ├── __init__.py
│   └── rag_agent.py         # RAGAgent: grounded answer generation via OpenAIGroundedGenerator
├── knowledge/
│   └── .gitkeep              # Will hold the fictional Spanish knowledge base
├── evals/
│   ├── __init__.py
│   ├── retrieval_cases.py   # Hand-curated retrieval evaluation dataset
│   ├── retrieval_metrics.py # Hit@K / reciprocal-rank pure functions
│   ├── retrieval_runner.py  # Runs cases against a SemanticRetriever, aggregates metrics
│   ├── run_retrieval_evals.py # CLI: python -m evals.run_retrieval_evals (real OpenAI calls)
│   ├── answer_cases.py      # Same 20 questions, with required facts + acceptable citations
│   ├── answer_checks.py     # normalize_text / fact / citation rule-based checks
│   ├── answer_runner.py     # Runs cases against a RAGAgent, aggregates answer-quality metrics
│   └── run_answer_evals.py  # CLI: python -m evals.run_answer_evals (real OpenAI calls)
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
