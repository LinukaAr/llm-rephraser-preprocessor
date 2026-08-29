from rephraser.config import PipelineConfig
from rephraser.pipeline import RephraserPipeline
from rephraser.types import Constraint


# Use token-overlap throughout so tests never trigger a model download.
_TEST_CONFIG = PipelineConfig(similarity_threshold=0.1, similarity_model="token-overlap")


def test_pipeline_pass_through_for_positive_text():
    pipeline = RephraserPipeline(config=_TEST_CONFIG)
    result = pipeline.process("Please explain the method in plain language.")
    assert result.output_text == "Please explain the method in plain language."
    assert result.strategy == "pass-through"
    assert result.passed_verifier


def test_pipeline_rephrases_negative_constraint():
    pipeline = RephraserPipeline(config=_TEST_CONFIG)
    result = pipeline.process("Don't use jargon in the explanation.")
    assert result.detected_constraint.detected_negation
    assert result.candidate_text != ""
    assert result.total_latency_ms >= 0


def test_pipeline_uses_generative_strategy_when_configured():
    from rephraser.generative import GenerativeRephraser

    generator = GenerativeRephraser(use_gemini=False)
    pipeline = RephraserPipeline(
        config=PipelineConfig(
            similarity_threshold=0.1,
            generation_strategy="generative",
            similarity_model="token-overlap",
        ),
        generative_rephraser=generator,
    )
    result = pipeline.process("Do not include code examples.")
    assert result.strategy == "generative"
    assert result.candidate_text == ""
    assert result.flagged
    assert result.fallback_reason == "gemini_disabled"
    assert result.output_text == "Do not include code examples."


def test_generator_fails_closed_without_gemini():
    from rephraser.generative import GenerativeRephraser

    generator = GenerativeRephraser(use_gemini=False)
    constraint = Constraint(text="don't tell her to come", detected_negation=True)
    result = generator.generate(constraint)
    assert result == ""
    assert generator.last_error == "gemini_disabled"


def test_pipeline_stacked_constraint_detects_first_cue():
    """The current implementation detects only the first negation cue.

    Stacked-constraint rephrasing (handling multiple cues in one sentence) is
    a Phase 5 extension. This test pins the existing behaviour so any future
    change to multi-cue handling is explicit.
    """
    pipeline = RephraserPipeline(config=_TEST_CONFIG)
    result = pipeline.process("Don't use jargon and avoid passive voice.")
    assert result.detected_constraint.detected_negation
    # First match in the sentence should be "don't"
    assert result.detected_constraint.cue is not None
    assert result.detected_constraint.cue.lower() == "don't"
