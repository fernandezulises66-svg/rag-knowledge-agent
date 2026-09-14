"""Pure rule-based checks for grounded-answer evaluation.

Text normalization, factual-content checks, and citation checks used
to evaluate final `RAGAnswer` objects against `AnswerEvalCase` ground
truth. No LLM judge, no semantic scoring, no similarity thresholds --
every check here is a transparent, explainable substring/equality
check over normalized text. Standard library only.
"""

import unicodedata

from agent.rag_agent import SourceCitation
from evals.retrieval_cases import RetrievalTarget


def normalize_text(text: str) -> str:
    """Normalize text for tolerant, transparent substring matching.

    Unicode-normalizes (NFKD), strips accents/diacritics, casefolds,
    and collapses repeated whitespace into single spaces.
    """
    decomposed = unicodedata.normalize("NFKD", text)
    without_accents = "".join(
        character for character in decomposed if not unicodedata.combining(character)
    )
    return " ".join(without_accents.casefold().split())


def check_required_facts(
    answer: str,
    required_fact_groups: tuple[tuple[str, ...], ...],
) -> bool:
    """Whether the answer mentions at least one alternative from every group.

    Each inner tuple is one required factual concept; any listed
    alternative satisfies it. Empty `required_fact_groups` passes
    trivially. This is deliberately simple substring matching -- it
    does not infer semantic equivalence beyond the listed alternatives.
    """
    normalized_answer = normalize_text(answer)

    for group in required_fact_groups:
        if not any(normalize_text(alternative) in normalized_answer for alternative in group):
            return False

    return True


def citation_matches_target(citation: SourceCitation, target: RetrievalTarget) -> bool:
    """Whether a citation matches one acceptable evidence target.

    A target with a section requires an exact source and section
    match. A target with `section=None` matches any section from that
    source.
    """
    if citation.source != target.source:
        return False
    if target.section is None:
        return True
    return citation.section == target.section


def check_citations(
    citations: tuple[SourceCitation, ...],
    acceptable_targets: tuple[RetrievalTarget, ...],
) -> bool:
    """Whether at least one citation exists and every citation is valid.

    A response does not pass merely because it includes one good
    citation alongside unrelated evidence -- EVERY returned citation
    must match at least one acceptable target.
    """
    if not citations:
        return False

    return all(
        any(citation_matches_target(citation, target) for target in acceptable_targets)
        for citation in citations
    )
