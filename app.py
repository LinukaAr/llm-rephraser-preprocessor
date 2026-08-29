from dataclasses import asdict
from functools import lru_cache
import os
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from rephraser.pipeline import RephraserPipeline


DEFAULT_CONFIG_PATH = Path("config/pipeline_config.json")
DEFAULT_CONFIG_ENV = "PIPELINE_CONFIG_PATH"


class RephraseRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000, description="Negative constraint text to rephrase")


class RephraseResponse(BaseModel):
    original_text: str
    output_text: str
    candidate_text: str
    strategy: str
    similarity: float
    passed_verifier: bool
    negation_leaked: bool
    flagged: bool
    fallback_reason: str
    detected_constraint: dict
    timings: list[dict]


app = FastAPI(
    title="Negation Rephraser",
    description="Hosted API for converting negative constraints into safer positive instructions.",
    version="1.0.0",
)


@lru_cache(maxsize=4)
def _load_pipeline(config_path: str) -> RephraserPipeline:
    return RephraserPipeline.from_config(Path(config_path))


def _get_pipeline(config_path: Optional[str] = None) -> RephraserPipeline:
    selected_path = config_path or os.getenv(DEFAULT_CONFIG_ENV) or str(DEFAULT_CONFIG_PATH)
    candidate_path = Path(selected_path)
    if not candidate_path.exists():
        raise HTTPException(status_code=404, detail=f"Config file not found: {candidate_path}")
    return _load_pipeline(str(candidate_path.resolve()))


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/rephrase", response_model=RephraseResponse)
def rephrase(request: RephraseRequest) -> dict:
    pipeline = _get_pipeline()
    result = pipeline.process(request.text)
    payload = asdict(result)
    payload["detected_constraint"] = asdict(result.detected_constraint)
    return payload


@app.get("/")
def root() -> dict:
    return {
        "service": "negation-rephraser",
        "health": "/health",
        "endpoint": "/rephrase",
    }
