from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Callable, List, Optional, Tuple, TypeVar

from .config import PipelineConfig
from .detector import detect_negation
from .generative import GenerativeRephraser
from .retrieval import RetrievalRephraser
from .types import Constraint
from .verifier import passes_threshold, repeats_prohibited_action, similarity_score


@dataclass
class StepTiming:
    name: str
    milliseconds: float


@dataclass
class RephraseResult:
    original_text: str
    output_text: str
    detected_constraint: Constraint
    candidate_text: str = ""
    strategy: str = "pass-through"
    similarity: float = 1.0
    passed_verifier: bool = True
    negation_leaked: bool = False
    flagged: bool = False
    fallback_reason: str = ""
    timings: List[StepTiming] = field(default_factory=list)

    @property
    def total_latency_ms(self) -> float:
        for item in reversed(self.timings):
            if item.name == "total":
                return item.milliseconds
        return sum(item.milliseconds for item in self.timings)

    @property
    def measured_latency_ms(self) -> float:
        return sum(item.milliseconds for item in self.timings if item.name != "total")


CandidateGenerator = Callable[[Constraint], str]
T = TypeVar("T")


class RephraserPipeline:
    """End-to-end pre-processor for converting negative constraints to positive ones."""

    def __init__(
        self,
        retrieval_rephraser: Optional[RetrievalRephraser] = None,
        generative_rephraser: Optional[GenerativeRephraser] = None,
        config: Optional[PipelineConfig] = None,
        candidate_generator: Optional[CandidateGenerator] = None,
    ):
        self.config = config or PipelineConfig()
        self.retrieval_rephraser = retrieval_rephraser or RetrievalRephraser.load_from_datafile(
            Path(self.config.dataset_path)
        )
        if not self.retrieval_rephraser._cache and self.retrieval_rephraser.pairs:
            self.retrieval_rephraser.fit(self.retrieval_rephraser.pairs)
        self.generative_rephraser = generative_rephraser or GenerativeRephraser()
        self.candidate_generator = candidate_generator

    @classmethod
    def from_config(cls, config_path: str | Path) -> "RephraserPipeline":
        config = PipelineConfig.from_json_file(config_path)
        return cls(config=config)

    def _timed(self, timings: List[StepTiming], name: str, fn: Callable[[], T]) -> T:
        start = perf_counter()
        value = fn()
        timings.append(StepTiming(name=name, milliseconds=(perf_counter() - start) * 1000.0))
        return value

    def _generate_candidate(self, constraint: Constraint, timings: List[StepTiming]) -> Tuple[str, str]:
        if self.candidate_generator is not None:
            candidate = self._timed(timings, "generative_candidate", lambda: self.candidate_generator(constraint))
            return candidate, "generative"

        if self.config.generation_strategy == "generative":
            candidate = self._timed(timings, "generative_candidate", lambda: self.generative_rephraser.generate(constraint))
            return candidate, "generative"

        candidate = self._timed(
            timings,
            "retrieval_candidate",
            lambda: self.retrieval_rephraser.rephrase(constraint.text)[0],
        )
        return candidate, "retrieval"

    def process(self, text: str) -> RephraseResult:
        timings: List[StepTiming] = []
        total_start = perf_counter()

        constraint = self._timed(timings, "detect_negation", lambda: detect_negation(text))

        if not constraint.detected_negation:
            timings.append(StepTiming(name="total", milliseconds=(perf_counter() - total_start) * 1000.0))
            return RephraseResult(
                original_text=text,
                output_text=text,
                detected_constraint=constraint,
                strategy="pass-through",
                similarity=1.0,
                passed_verifier=True,
                timings=timings,
            )

        candidate_text, strategy = self._generate_candidate(constraint, timings)
        if not candidate_text:
            fallback_reason = "no_candidate"
            if strategy == "generative":
                fallback_reason = getattr(self.generative_rephraser, "last_error", "") or fallback_reason
            output_text = text if self.config.fallback_to_original else ""
            timings.append(StepTiming(name="total", milliseconds=(perf_counter() - total_start) * 1000.0))
            return RephraseResult(
                original_text=text,
                output_text=output_text,
                detected_constraint=constraint,
                candidate_text=candidate_text,
                strategy=strategy,
                similarity=0.0,
                passed_verifier=False,
                negation_leaked=False,
                flagged=True,
                fallback_reason=fallback_reason,
                timings=timings,
            )

        similarity_value = self._timed(
            timings,
            "similarity_score",
            lambda: similarity_score(text, candidate_text, model_name=self.config.similarity_model),
        )
        passed, _ = passes_threshold(
            text,
            candidate_text,
            threshold=self.config.similarity_threshold,
            model_name=self.config.similarity_model,
        )

        negation_leaked = detect_negation(candidate_text).detected_negation
        repeated_prohibited_action = repeats_prohibited_action(text, candidate_text)
        passed_verifier = passed and not negation_leaked and not repeated_prohibited_action

        if passed_verifier:
            output_text = candidate_text
            fallback_reason = ""
            flagged = False
        else:
            output_text = text if self.config.fallback_to_original else ""
            fallback_reason = "verifier_rejected"
            flagged = True

        timings.append(StepTiming(name="total", milliseconds=(perf_counter() - total_start) * 1000.0))
        return RephraseResult(
            original_text=text,
            output_text=output_text,
            detected_constraint=constraint,
            candidate_text=candidate_text,
            strategy=strategy,
            similarity=float(similarity_value),
            passed_verifier=passed_verifier,
            negation_leaked=negation_leaked,
            flagged=flagged,
            fallback_reason=fallback_reason,
            timings=timings,
        )
