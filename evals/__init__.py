"""Transparent retrieval-evaluation framework for the RAG Knowledge Agent.

Separate from pytest unit tests: this package measures whether semantic
retrieval returns the expected evidence chunks for known questions,
using Hit@1, Hit@3, and MRR@5 over answerable cases, while tracking
intentionally unsupported questions as diagnostics only (see
`evals/retrieval_cases.py`). It does not implement answer generation,
grounding decisions, similarity thresholds, or an LLM judge.
"""
