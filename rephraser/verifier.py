from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from .text import word_tokens


DEFAULT_EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"
TOKEN_OVERLAP_MODEL = "token-overlap"
_MODEL_CACHE: Dict[str, object] = {}


@dataclass(frozen=True)
class SimilarityResult:
    score: float
    model_name: str
    error: Optional[str] = None


def _overlap_score(a: str, b: str) -> float:
    tokens_a = word_tokens(a)
    tokens_b = word_tokens(b)
    if not tokens_a or not tokens_b:
        return 0.0

    set_a = set(tokens_a)
    set_b = set(tokens_b)
    denominator = (len(set_a) + len(set_b)) / 2
    if denominator == 0:
        return 0.0
    return len(set_a & set_b) / denominator


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
