import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .text import word_tokens


STOPWORDS = {
    "a",
    "an",
    "and",
    "do",
    "don",
    "explain",
    "include",
    "in",
    "mention",
    "not",
    "please",
    "say",
    "state",
    "the",
    "to",
    "t",
    "use",
    "using",
}


def _tokenize(text: str) -> List[str]:
    return [token for token in word_tokens(text) if token not in STOPWORDS]


def _vectorize(tokens: List[str]) -> Dict[str, int]:
    vector: Dict[str, int] = {}
    for token in tokens:
        vector[token] = vector.get(token, 0) + 1
    return vector


def _cosine_dict(a: Dict[str, int], b: Dict[str, int]) -> float:
    numerator = sum(value * b.get(key, 0) for key, value in a.items())
    norm_a = math.sqrt(sum(value * value for value in a.values()))
    norm_b = math.sqrt(sum(value * value for value in b.values()))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return numerator / (norm_a * norm_b)


class RetrievalRephraser:
    """Retrieval rephraser using curated negative-to-positive pilot pairs."""

    def __init__(self, pairs: Optional[List[Dict]] = None):
        self.pairs = pairs or []
        self._cache: List[Tuple[str, str, Dict[str, int]]] = []

    @classmethod
    def load_from_datafile(cls, path: Optional[Path] = None) -> "RetrievalRephraser":
        if path is None:
            base = Path(__file__).resolve().parents[1]
            path = base / "data" / "pilot_constraints.json"
        with open(path, "r", encoding="utf-8") as handle:
            return cls(json.load(handle))

    def fit(self, pairs: List[Dict]) -> None:
        self.pairs = pairs
        self._cache = [
            (
                str(pair.get("negative", "")),
                str(pair.get("positive", "")),
                _vectorize(_tokenize(str(pair.get("negative", "")))),
            )
            for pair in pairs
        ]

    def rephrase(self, text: str) -> Tuple[str, float]:
        if not self._cache and self.pairs:
            self.fit(self.pairs)

        query_vector = _vectorize(_tokenize(text))
        best_text = ""
        best_score = 0.0

        for _negative, positive, vector in self._cache:
            score = _cosine_dict(query_vector, vector)
            if score > best_score:
                best_score = score
                best_text = positive

        if best_score <= 0.0:
            return "", 0.0

        return best_text, float(best_score)
