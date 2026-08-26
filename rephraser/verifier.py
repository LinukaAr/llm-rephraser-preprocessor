import os
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from .text import word_tokens


TOKEN_OVERLAP_MODEL = "token-overlap"
DEFAULT_EMBEDDING_MODEL = os.getenv("SIMILARITY_MODEL", TOKEN_OVERLAP_MODEL)
_MODEL_CACHE: Dict[str, object] = {}
SIMILARITY_STOPWORDS = {
    "a",
    "an",
    "and",
    "at",
    "do",
    "don",
    "dont",
    "not",
    "of",
    "please",
    "the",
    "to",
    "t",
}
NEGATED_ACTION_CUES = {
    ("don", "t"),
    ("dont",),
    ("do", "not"),
    ("never",),
    ("avoid",),
}
SAFE_AVOIDANCE_MARKERS = {
    "free",
    "instead",
    "omit",
    "omitted",
    "omitting",
    "leave",
    "leaving",
    "left",
    "out",
    "skip",
    "skipping",
}


def _content_tokens(text: str) -> set[str]:
    return {token for token in word_tokens(text) if token not in SIMILARITY_STOPWORDS}


def _prohibited_action_terms(text: str) -> Tuple[str, str] | None:
    tokens = word_tokens(text)
    for index in range(len(tokens)):
        for cue in NEGATED_ACTION_CUES:
            cue_end = index + len(cue)
            if tuple(tokens[index:cue_end]) != cue or cue_end >= len(tokens):
                continue

            action = tokens[cue_end]
            for token in tokens[cue_end + 1 :]:
                if token not in SIMILARITY_STOPWORDS:
                    return action, token
    return None


def repeats_prohibited_action(original: str, candidate: str) -> bool:
    terms = _prohibited_action_terms(original)
    if terms is None:
        return False

    action, object_token = terms
    candidate_tokens = set(word_tokens(candidate))
    if action not in candidate_tokens or object_token not in candidate_tokens:
        return False

    return not bool(candidate_tokens & SAFE_AVOIDANCE_MARKERS)


@dataclass(frozen=True)
class SimilarityResult:
    score: float
    model_name: str
    error: Optional[str] = None


def _overlap_score(a: str, b: str) -> float:
    set_a = _content_tokens(a)
    set_b = _content_tokens(b)
    if not set_a or not set_b:
        return 0.0

    intersection = len(set_a & set_b)
    denominator = (len(set_a) + len(set_b)) / 2
    if denominator == 0:
        return 0.0

    overlap = intersection / denominator
    source_recall = intersection / len(set_a)
    return max(overlap, source_recall)


def _embedding_score(a: str, b: str, model_name: str) -> float:
    from numpy import dot
    from numpy.linalg import norm
    from sentence_transformers import SentenceTransformer

    model = _MODEL_CACHE.get(model_name)
    if model is None:
        model = SentenceTransformer(model_name)
        _MODEL_CACHE[model_name] = model

    first, second = model.encode([a, b])
    denominator = norm(first) * norm(second)
    if denominator == 0:
        return 0.0
    return float(dot(first, second) / denominator)


def similarity_result(a: str, b: str, model_name: Optional[str] = None) -> SimilarityResult:
    selected_model = model_name or DEFAULT_EMBEDDING_MODEL
    if selected_model == TOKEN_OVERLAP_MODEL:
        return SimilarityResult(score=float(_overlap_score(a, b)), model_name=TOKEN_OVERLAP_MODEL)

    try:
        return SimilarityResult(
            score=_embedding_score(a, b, selected_model),
            model_name=selected_model,
        )
    except Exception as exc:
        return SimilarityResult(score=0.0, model_name=selected_model, error=str(exc))


def similarity_score(a: str, b: str, model_name: Optional[str] = None) -> float:
    """Compute semantic similarity. Use token-overlap only as an explicit test model."""
    return similarity_result(a, b, model_name=model_name).score


def passes_threshold(
    a: str,
    b: str,
    threshold: float = 0.85,
    model_name: Optional[str] = None,
) -> Tuple[bool, float]:
    score = similarity_score(a, b, model_name=model_name)
    return score >= threshold, score
