"""Validate numbered citations before an answer can be cached or accepted."""

import re
from dataclasses import asdict, dataclass

from app.generation.prompt import PromptSource
from app.retrieval.gate import REFUSAL_MESSAGE

CITATION = re.compile(r"\[(\d+)\]")


@dataclass(frozen=True)
class CitationCheck:
    valid: bool
    refusal: bool
    citations: tuple[dict[str, object], ...]
    reason: str | None


def check_citations(answer: str, sources: tuple[PromptSource, ...]) -> CitationCheck:
    stripped = answer.strip()
    if not stripped:
        return CitationCheck(False, False, (), "empty_answer")
    if stripped == REFUSAL_MESSAGE:
        return CitationCheck(True, True, (), None)

    by_marker = {source.marker: source for source in sources}
    markers = [int(match.group(1)) for match in CITATION.finditer(stripped)]
    if not markers:
        return CitationCheck(False, False, (), "missing_citations")
    if any(marker not in by_marker for marker in markers):
        return CitationCheck(False, False, (), "unknown_citation")
    ordered = tuple(dict.fromkeys(markers))
    return CitationCheck(
        True,
        False,
        tuple(asdict(by_marker[marker]) for marker in ordered),
        None,
    )
