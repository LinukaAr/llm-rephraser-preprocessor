from rephraser.detector import detect_negation


def test_detect_simple_negation():
    c = detect_negation("Don't use jargon in the explanation.")
    assert c.detected_negation
    assert c.cue is not None


def test_detect_unpunctuated_dont():
    c = detect_negation("dont say i'm an Computer Science Undergraduate")
    assert c.detected_negation
    assert c.cue == "dont"


def test_detect_no_negation():
    c = detect_negation("Please explain the method in plain language.")
    assert not c.detected_negation


def test_detect_constraint_type_lexical():
    c = detect_negation("Don't use jargon in the explanation.")
    assert c.constraint_type == "lexical"


def test_detect_constraint_type_structural():
    c = detect_negation("Do not include code examples.")
    assert c.constraint_type == "structural"


def test_detect_constraint_type_behavioral():
    c = detect_negation("Never mention internal company names.")
    assert c.constraint_type == "behavioral"
