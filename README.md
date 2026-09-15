# RAG Knowledge Agent

## Problem

Internal company knowledge is often scattered across multiple documents —
pricing, billing, policies, onboarding guides, FAQs — making it hard for
users to find accurate answers quickly. Manually searching through
documentation is slow and error-prone. This project explores how a
Retrieval-Augmented Generation (RAG) assistant can let users ask
natural-language questions and receive answers grounded in a controlled,
internal knowledge base, with visible sources.

## RAG Flow

```
User question
  → document retrieval
  → relevant chunks
  → grounded LLM answer
  → source citations
```

This flow is implemented end to end, from the Markdown knowledge base through
a Streamlit UI. See **Current Status** and **Evaluation Results** below for
what has actually been built and measured.

## Tech Stack

- Python 3.13+
- OpenAI API (embeddings + Responses API structured output)
- Retrieval-Augmented Generation (RAG)
- Cosine-similarity vector search
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
multilingual embeddings, cosine-similarity semantic retrieval, grounded
structured answer generation, two transparent evaluation frameworks, and a
Streamlit UI are all implemented.

`agent/rag_agent.py` retrieves evidence with `SemanticRetriever`, asks the
generation model (via the OpenAI Responses API, structured Pydantic output)
whether that evidence is sufficient, independently validates the model's
decision in Python, and returns deterministic citations built only from
real chunk metadata. Unsupported questions — including ones where
retrieval returns topically related, high-scoring but insufficient
evidence — receive an explicit, controlled fallback message instead of a
model-generated guess.

Two separate evaluation frameworks live under `evals/`: one measures
**retrieval** quality (Hit@K, MRR), the other measures the **final grounded
answer** (answerability, factual grounding via accent/case-insensitive
alternative-text matching, citation validity, controlled refusal) using
transparent, rule-based checks only — no LLM judge, no semantic quality
score. See **Evaluation Results** below; these are curated-benchmark
results, not claims of general model accuracy, zero hallucinations, or
perfect RAG accuracy.

`streamlit_app.py` provides a Spanish-first UI around the same public
`RAGAgent`: it shows the grounded answer, its source citations, and an
expandable retrieval-observability panel (retrieved chunks and their
semantic-similarity scores), plus lightweight per-session query history.
It adds no new retrieval, chunking, embedding, or grounding behavior.

## Evaluation Results

These results apply only to this project's own small, hand-curated
benchmark of 20 questions against its fictional knowledge base — they are
**not** a measure of general model accuracy.

### Retrieval evaluation

- 16 answerable queries
- Hit@1 = 100%
- Hit@3 = 100%
- MRR@5 = 1.00
- 4 unsupported queries tracked separately as diagnostics

### Grounded answer evaluation

- 20 curated cases total (16 answerable + 4 unsupported)
- Answerability accuracy = 100%
- Fact check pass rate = 100%
- Citation check pass rate = 100%
- Unsupported refusal rate = 100%
- Overall pass rate = 100%

Run `python -m evals.run_retrieval_evals` or `python -m evals.run_answer_evals`
to reproduce these (real OpenAI API calls; never run by `pytest`).

## Project Structure

```
rag-knowledge-agent/
├── app.py                   # Placeholder CLI entry point (unrelated to the Streamlit UI)
├── streamlit_app.py         # Spanish-first Streamlit UI around the public RAGAgent
├── rag/
│   ├── __init__.py
│   ├── document_loader.py   # Loads Markdown files from knowledge/ into KnowledgeDocument objects
│   ├── chunker.py           # Splits KnowledgeDocuments into section-aware KnowledgeChunk objects
│   ├── embeddings.py        # Turns KnowledgeChunks into EmbeddedChunk objects via the OpenAI API
│   └── retriever.py         # Ranks EmbeddedChunks against a query via SemanticRetriever
├── agent/
│   ├── __init__.py
│   └── rag_agent.py         # RAGAgent: grounded answer generation via OpenAIGroundedGenerator
├── knowledge/                # Fictional Spanish knowledge base (9 Markdown documents)
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
├── tests/                    # pytest suite (no real OpenAI calls)
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

Copy `.env.example` to `.env` and set `OPENAI_API_KEY` before running the
Streamlit app or either evaluation CLI — they make real OpenAI API calls
(embeddings and/or the Responses API). `pytest` never requires a real key,
since every test uses a fake/injected client.

## Running the UI

```powershell
streamlit run streamlit_app.py
```

On first run, `build_rag_agent()` loads and embeds the knowledge base once
(one real OpenAI embeddings request for the whole corpus) and caches the
resulting pipeline via `st.cache_resource`, so it is not rebuilt on every
rerun. No deployment is set up yet — this runs locally only.
