from rephraser.retrieval import RetrievalRephraser
from rephraser import verifier
from rephraser.verifier import passes_threshold, repeats_prohibited_action, similarity_result


# ---------------------------------------------------------------------------
# Retrieval rephraser
# ---------------------------------------------------------------------------

def test_retrieval_returns_candidate():
    rr = RetrievalRephraser.load_from_datafile()
    rr.fit(rr.pairs)
    candidate, score = rr.rephrase("Don't use jargon in the explanation.")
    assert candidate != ""
    assert score >= 0


# ---------------------------------------------------------------------------
# Similarity verifier (token-overlap model — no model download required)
# ---------------------------------------------------------------------------

def test_verifier_overlap():
    orig = "Don't use jargon in the explanation."
    cand = "Use plain, everyday language in the explanation."
    ok, s = passes_threshold(orig, cand, threshold=0.1, model_name="token-overlap")
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


# ---------------------------------------------------------------------------
# repeats_prohibited_action guard
# ---------------------------------------------------------------------------

def test_repeats_prohibited_action_detects_repeated_action():
    """Candidate that still performs the forbidden action is flagged."""
    assert repeats_prohibited_action(
        "Don't use jargon",
        "use jargon in this context",
    )


def test_repeats_prohibited_action_accepts_safe_avoidance():
    """Candidate that omits the object via a safe marker is not flagged."""
    assert not repeats_prohibited_action(
        "Don't use jargon",
        "leave jargon out of the text",
    )


def test_repeats_prohibited_action_returns_false_when_no_negation():
    """Non-negative original sentence always returns False."""
    assert not repeats_prohibited_action(
        "Use plain language.",
        "Use plain language here.",
    )


def test_repeats_prohibited_action_returns_false_when_object_absent():
    """Candidate that mentions the action but not the object is not flagged."""
    assert not repeats_prohibited_action(
        "Don't use jargon",
        "use clear phrasing",
    )
