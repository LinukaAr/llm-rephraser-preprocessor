import re
from typing import Optional

from .types import Constraint


NEGATION_CUES = [
    r"\bdon't\b",
    r"\bdont\b",
    r"\bdo not\b",
    r"\bnever\b",
    r"\bwithout\b",
    r"\bother than\b",
    r"\bavoid\b",
    r"\brefrain from\b",
    r"\bexclude\b",
    r"\bexcluding\b",
    r"\bno\b",
    r"\bnot\b",
]

CUES_RE = re.compile("|".join(NEGATION_CUES), flags=re.IGNORECASE)
SENTENCE_SPLIT_RE = re.compile(r"(?<=[\.?\!\n])\s+")

# ---------------------------------------------------------------------------
# Provisional constraint-type classifier
# Wire RO1 taxonomy labels here once the taxonomy is finalised.
# ---------------------------------------------------------------------------

_LEXICAL_KEYWORDS = frozenset({
    "jargon", "acronym", "acronyms", "voice", "passive", "active",
    "word", "language", "technical", "term", "terms", "vocabulary",
    "spelling", "grammar", "tone", "style",
})
_STRUCTURAL_KEYWORDS = frozenset({
    "code", "example", "examples", "bullet", "list", "reference",
    "references", "format", "section", "heading", "table", "image",
    "link", "footnote", "citation",
})


def _classify_constraint_type(text: str) -> str:
    """Provisional heuristic — to be replaced by RO1 taxonomy labels."""
    tokens = set(re.findall(r"\w+", text.lower()))
    if tokens & _LEXICAL_KEYWORDS:
        return "lexical"
    if tokens & _STRUCTURAL_KEYWORDS:
        return "structural"
    return "behavioral"


def detect_negation(text: str) -> Constraint:
    """Detect instruction-style negation cues and return a structured Constraint."""
    match = CUES_RE.search(text)
    if match is None:
        return Constraint(text=text, detected_negation=False)

    start, end = match.span(0)
    return Constraint(
        text=text,
        cue=match.group(0),
        cue_span=(start, end),
        scope_span=_scope_span(text, start),
        constraint_type=_classify_constraint_type(text),
        detected_negation=True,
    )


def _scope_span(text: str, cue_start: int) -> tuple[int, int]:
    """Return the (start, end) character span of the sentence containing *cue_start*.

    Uses ``finditer`` over the inter-sentence whitespace matches so that the
    offset is exact regardless of the separator width (single space, \\r\\n, …).
    """
    prev_end = 0
    for m in SENTENCE_SPLIT_RE.finditer(text):
        if m.start() > cue_start:
            return prev_end, m.start()
        prev_end = m.end()
    return prev_end, len(text)
