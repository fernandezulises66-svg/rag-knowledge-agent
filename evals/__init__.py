"""Transparent evaluation frameworks for the RAG Knowledge Agent.

Separate from pytest unit tests, and separate from each other:

- `retrieval_cases.py` / `retrieval_metrics.py` / `retrieval_runner.py`
  measure whether semantic RETRIEVAL returns the expected evidence
  chunks for known questions (Hit@1, Hit@3, MRR@5).
- `answer_cases.py` / `answer_checks.py` / `answer_runner.py` measure
  the FINAL grounded RAG answer (answerability, factual grounding,
  citation validity, controlled refusal) using the same 20 curated
  questions.

Both frameworks track intentionally unsupported questions as
diagnostics rather than scoring them as failures. Neither uses an LLM
judge or a similarity-score threshold, and neither implements answer
generation itself -- that lives in `agent.rag_agent`.
"""
