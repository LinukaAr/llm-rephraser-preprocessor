from dataclasses import dataclass
from pathlib import Path
import json
from typing import Any, Dict, Union


@dataclass
class PipelineConfig:
    # FR-M2 threshold: 0.85 per spec. Lower values for dev/pilot runs only.
    similarity_threshold: float = 0.85
    fallback_to_original: bool = True
    dataset_path: str = "data/pilot_constraints.json"
    generation_strategy: str = "retrieval"
    # Similarity model used by the verifier.
    # Production / evaluation runs: "Qwen/Qwen3-Embedding-0.6B"
    # Lightweight / test runs:      "token-overlap"
    similarity_model: str = "token-overlap"

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PipelineConfig":
        return cls(
            similarity_threshold=float(data.get("similarity_threshold", 0.85)),
            fallback_to_original=bool(data.get("fallback_to_original", True)),
            dataset_path=str(data.get("dataset_path", "data/pilot_constraints.json")),
            generation_strategy=str(data.get("generation_strategy", "retrieval")),
            similarity_model=str(data.get("similarity_model", "token-overlap")),
        )

    @classmethod
    def from_json_file(cls, path: Union[str, Path]) -> "PipelineConfig":
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        return cls.from_dict(data)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "similarity_threshold": self.similarity_threshold,
            "fallback_to_original": self.fallback_to_original,
            "dataset_path": self.dataset_path,
            "generation_strategy": self.generation_strategy,
            "similarity_model": self.similarity_model,
        }
