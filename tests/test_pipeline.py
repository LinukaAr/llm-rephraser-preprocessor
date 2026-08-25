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
    pipeline = RephraserPipeline(config=PipelineConfig(similarity_threshold=0.1, generation_strategy="generative"))
    result = pipeline.process("Do not include code examples.")
    assert result.strategy == "generative"
    assert result.candidate_text != ""
    assert result.output_text != ""


def test_generator_preserves_prohibition_meaning_for_change_requests():
    from rephraser.generative import GenerativeRephraser

    generator = GenerativeRephraser(use_gemini=False)
    result = generator.generate(type("Constraint", (), {"text": "don't change the face of the person"})())
    assert "as-is" in result.lower() or "intact" in result.lower()
    assert "change" not in result.lower()

