from dataclasses import dataclass
from pathlib import Path
import json
from typing import Any, Dict, Union


@dataclass
class PipelineConfig:
    similarity_threshold: float = 0.6
    fallback_to_original: bool = True
    dataset_path: str = "data/pilot_constraints.json"
    generation_strategy: str = "retrieval"

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PipelineConfig":
        return cls(
            similarity_threshold=float(data.get("similarity_threshold", 0.6)),
            fallback_to_original=bool(data.get("fallback_to_original", True)),
            dataset_path=str(data.get("dataset_path", "data/pilot_constraints.json")),
            generation_strategy=str(data.get("generation_strategy", "retrieval")),
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
        }
