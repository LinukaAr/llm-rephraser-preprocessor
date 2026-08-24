from rephraser.retrieval import RetrievalRephraser
from rephraser import verifier
from rephraser.verifier import passes_threshold, similarity_result


def test_retrieval_returns_candidate():
    rr = RetrievalRephraser.load_from_datafile()
    rr.fit(rr.pairs)
    candidate, score = rr.rephrase("Don't use jargon in the explanation.")
    assert candidate != ""
    assert score >= 0


def test_verifier_overlap():
    orig = "Don't use jargon in the explanation."
    cand = "Use plain, everyday language in the explanation."
    ok, s = passes_threshold(orig, cand, threshold=0.1, model_name="token-overlap")
    # With low threshold, candidate should pass using overlap fallback
    assert ok


def test_similarity_result_uses_explicit_token_overlap_model():
    result = similarity_result(
        "Do not use jargon in the explanation.",
        "Use plain language in the explanation.",
        model_name="token-overlap",
    )
    assert result.model_name == "token-overlap"
    assert result.error is None


def test_similarity_result_fails_closed_when_embedding_model_unavailable(monkeypatch):
    def raise_loader_error(_a, _b, _model_name):
        raise RuntimeError("model unavailable")

    monkeypatch.setattr(verifier, "_embedding_score", raise_loader_error)
    result = similarity_result(
        "Do not use jargon in the explanation.",
        "Use plain language in the explanation.",
        model_name="Qwen/Qwen3-Embedding-0.6B",
    )
    assert result.score == 0.0
    assert result.error == "model unavailable"
