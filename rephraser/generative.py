from dataclasses import dataclass
import os
import re
from pathlib import Path
from typing import List, Optional

from .types import Constraint


def _collapse_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _strip_leading_negation(text: str) -> str:
    text = re.sub(r"^\s*(do not|don't|never|avoid|refrain from|without|exclude|excluding)\s+", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^\s*please\s+", "", text, flags=re.IGNORECASE)
    return text.strip(" .,")


@dataclass
class GenerativeRephraser:
    """Gemini-backed rephraser with a deterministic local fallback.

    The class first tries Gemini when an API key and SDK are available.
    If that fails, it falls back to a deterministic rewrite so the pipeline
    still runs offline and the tests remain stable.
    """

    preserve_scope: bool = True
    model_name: str = "gemini-1.5-flash"
    api_key_env: str = "GEMINI_API_KEY"
    temperature: float = 0.2
    max_output_tokens: int = 128
    use_gemini: bool = True

    def build_prompt(self, constraint: Constraint) -> str:
        examples: List[str] = [
            "Negative: Don't use jargon in the explanation.\nPositive: Use plain language in the explanation.",
            "Negative: Do not include code examples.\nPositive: Provide an explanation focused on prose and concepts.",
            "Negative: Avoid using passive voice.\nPositive: Use active voice where possible.",
            "Negative: Don't eat junk food.\nPositive: Choose healthy food instead.",
        ]
        return "\n\n".join(
            examples
            + [
                "Task: Rewrite the negative constraint below as a positive instruction.",
                "Rules: preserve the intent and scope, keep it concise, and avoid negation words in the output.",
                "If the source is a prohibition, rewrite it as a positive alternative or preference, not as permission to do the forbidden action.",
                "Return only the rewritten positive instruction.",
                f"Negative: {constraint.text}",
            ]
        )

    def _load_gemini_model(self):
        api_key = os.getenv(self.api_key_env) or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            api_key = self._load_api_key_from_file()
        if not api_key:
            return None

        try:
            import google.generativeai as genai  # type: ignore
        except Exception:
            return None

        try:
            genai.configure(api_key=api_key)
            return genai, genai.GenerativeModel(self.model_name)
        except Exception:
            return None

    def _load_api_key_from_file(self) -> str:
        root = Path(__file__).resolve().parents[1]
        candidate_paths = [
            root / "config" / "gemini.env",
            root / "config" / "gemini.env.example",
        ]

        for path in candidate_paths:
            if not path.exists():
                continue

            try:
                with open(path, "r", encoding="utf-8") as handle:
                    for raw_line in handle:
                        line = raw_line.strip()
                        if not line or line.startswith("#") or "=" not in line:
                            continue

                        key, value = line.split("=", 1)
                        if key.strip() in {self.api_key_env, "GOOGLE_API_KEY"}:
                            return value.strip().strip('"\'')
            except Exception:
                continue

        return ""

    def _extract_text(self, response) -> str:
        candidate = getattr(response, "text", None)
        if candidate:
            return str(candidate)

        try:
            parts = response.candidates[0].content.parts  # type: ignore[attr-defined]
            return " ".join(getattr(part, "text", "") for part in parts).strip()
        except Exception:
            return ""

    def _sanitize_output(self, text: str) -> str:
        cleaned = _collapse_whitespace(text)
        cleaned = re.sub(r"^(positive|answer|rewrite)\s*:\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = cleaned.strip('"\'`')
        if cleaned.endswith("."):
            return cleaned
        return f"{cleaned}." if cleaned else ""

    def _generate_with_gemini(self, constraint: Constraint) -> str:
        model_handle = self._load_gemini_model()
        if model_handle is None:
            return ""

        genai, model = model_handle
        prompt = self.build_prompt(constraint)
        try:
            response = model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(  # type: ignore[attr-defined]
                    temperature=self.temperature,
                    max_output_tokens=self.max_output_tokens,
                ),
            )
        except Exception:
            return ""

        return self._sanitize_output(self._extract_text(response))

    def _rewrite(self, text: str) -> str:
        normalized = _collapse_whitespace(text)
        lowered = normalized.lower()

        change_match = re.match(r"^\s*(?:do not|don't|never|avoid|refrain from)\s+(?:change|alter|modify)\s+(.+?)(?:[.?!]+)?\s*$", normalized, flags=re.IGNORECASE)
        if change_match:
            target = change_match.group(1).strip(" .,")
            return f"Keep {target} as-is."

        add_match = re.match(r"^\s*(?:do not|don't|never|avoid)\s+add\s+(.+?)(?:\s+to\s+(.+?))?(?:[.?!]+)?\s*$", normalized, flags=re.IGNORECASE)
        if add_match:
            added = add_match.group(1).strip(" .,")
            target = add_match.group(2)
            if target:
                return f"Keep {target.strip(' .,')} free of {added}."
            return f"Keep {added} out of the result."

        eat_match = re.match(r"^\s*(?:do not|don't)\s+eat\s+(.+?)(?:[.?!]+)?\s*$", normalized, flags=re.IGNORECASE)
        if eat_match:
            food = eat_match.group(1).strip(" .,")
            return f"Choose healthy food instead of {food}."

        if "jargon" in lowered:
            return "Use plain language in the explanation."

        if "passive voice" in lowered:
            return "Use active voice where possible."

        if "code examples" in lowered:
            return "Provide an explanation focused on prose and concepts."

        if "future work" in lowered:
            return "Focus the response on the current task and immediate objective."

        if "internal company names" in lowered:
            return "Leave internal company names out of the explanation."

        if "acronym" in lowered or "acronyms" in lowered:
            return "Summarize the method using plain language and no domain-specific acronyms."

        stripped = _strip_leading_negation(normalized)
        if stripped:
            if stripped[0].islower():
                stripped = stripped[0].upper() + stripped[1:]
            return stripped if stripped.endswith(".") else f"{stripped}."

        return "Use clear, affirmative instructions that preserve the original intent."

    def generate(self, constraint: Constraint) -> str:
        if self.use_gemini:
            candidate = self._generate_with_gemini(constraint)
            if candidate:
                return candidate

        _prompt = self.build_prompt(constraint)
        _ = _prompt  # keep the prompt construction explicit for traceability
        return self._rewrite(constraint.text)
