import re

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
SENTENCE_SPLIT_RE = re.compile(r"(?<=[\.\?\!\n])\s+")


def detect_negation(text: str) -> Constraint:
    """Detect instruction-style negation cues and return a structured constraint."""
    match = CUES_RE.search(text)
    if match is None:
        return Constraint(text=text, detected_negation=False)

    start, end = match.span(0)
    return Constraint(
        text=text,
        cue=match.group(0),
        cue_span=(start, end),
        scope_span=_scope_span(text, start),
        detected_negation=True,
    )


def _scope_span(text: str, cue_start: int) -> tuple[int, int]:
    offset = 0
    for sentence in SENTENCE_SPLIT_RE.split(text):
        sentence_start = offset
        sentence_end = offset + len(sentence)
        if sentence_start <= cue_start < sentence_end:
            return sentence_start, sentence_end
        offset = sentence_end + 1
    return 0, len(text)
