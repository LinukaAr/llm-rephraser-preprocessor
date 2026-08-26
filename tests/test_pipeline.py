from rephraser.config import PipelineConfig
from rephraser.pipeline import RephraserPipeline


def test_pipeline_pass_through_for_positive_text():
    pipeline = RephraserPipeline(config=PipelineConfig())
    result = pipeline.process("Please explain the method in plain language.")
    assert result.output_text == "Please explain the method in plain language."
    assert result.strategy == "pass-through"
    assert result.passed_verifier


def test_pipeline_rephrases_negative_constraint():
    pipeline = RephraserPipeline(config=PipelineConfig(similarity_threshold=0.1))
    result = pipeline.process("Don't use jargon in the explanation.")
    assert result.detected_constraint.detected_negation
    assert result.candidate_text != ""
    assert result.total_latency_ms >= 0


def test_pipeline_uses_generative_strategy_when_configured():
    from rephraser.generative import GenerativeRephraser

    generator = GenerativeRephraser(use_gemini=False)
    pipeline = RephraserPipeline(
        config=PipelineConfig(similarity_threshold=0.1, generation_strategy="generative"),
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
    result = generator.generate(type("Constraint", (), {"text": "don't tell her to come"})())
    assert result == ""
    assert generator.last_error == "gemini_disabled"
