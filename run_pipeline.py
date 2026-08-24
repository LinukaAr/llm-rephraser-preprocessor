import argparse
import json
from pathlib import Path

from rephraser.pipeline import RephraserPipeline


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the rephraser pre-processor on one input string.")
    parser.add_argument("text", help="Negative constraint text to process")
    parser.add_argument(
        "--config",
        default="config/pipeline_config.json",
        help="Path to pipeline config JSON (default: config/pipeline_config.json)",
    )
    args = parser.parse_args()

    pipeline = RephraserPipeline.from_config(Path(args.config))
    result = pipeline.process(args.text)

    payload = {
        "original_text": result.original_text,
        "output_text": result.output_text,
        "candidate_text": result.candidate_text,
        "strategy": result.strategy,
        "similarity": result.similarity,
        "passed_verifier": result.passed_verifier,
        "negation_leaked": result.negation_leaked,
        "flagged": result.flagged,
        "fallback_reason": result.fallback_reason,
        "timings": [{"name": t.name, "milliseconds": round(t.milliseconds, 3)} for t in result.timings],
    }
    print(json.dumps(payload, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
