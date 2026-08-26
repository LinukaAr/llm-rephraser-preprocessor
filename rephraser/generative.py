from dataclasses import dataclass
import os
import re
from pathlib import Path
from typing import List, Optional

from .types import Constraint


def _collapse_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


@dataclass
class GenerativeRephraser:
    """Gemini-backed rephraser.

    The class tries Gemini when an API key and SDK are available. If that is
    unavailable, it returns no candidate and records ``last_error`` so the
    pipeline can fail closed without hiding why generation failed.
    """

    preserve_scope: bool = True
    model_name: str = "gemini-3.5-flash-lite"
    api_key_env: str = "GEMINI_API_KEY"
    temperature: float = 0.2
    max_output_tokens: int = 512
    use_gemini: bool = True
    request_timeout_seconds: float = 20.0
    last_error: str = ""

    def build_prompt(self, constraint: Constraint) -> str:
        examples: List[str] = [
            'Input: "Don\'t use jargon in the explanation."\nOutput: "Use plain language in the explanation."',
            'Input: "Do not include code examples."\nOutput: "Include prose explanations instead of code examples."',
            'Input: "Avoid using passive voice."\nOutput: "Use active voice where possible."',
            'Input: "Don\'t add sugar."\nOutput: "Add only sugar-free ingredients."',
        ]
        return "\n\n".join(
            examples
            + [
                "Rewrite the final input as one affirmative instruction sentence.",
                "Preserve the original intent and scope.",
                "Keep the main action verb and exact key object nouns when an affirmative rewrite can still preserve the meaning.",
                "Do not replace key object nouns with synonyms.",
                "The output must not contain negation cues such as no, not, don't, never, without, avoid, exclude, or other than.",
                "Correct obvious spelling mistakes while preserving meaning.",
                "Treat full stop or fulstop as the period punctuation mark at the end of a sentence.",
                "Return only the rewritten sentence, with no label, list, commentary, or quoted text.",
                f'Input: "{constraint.text}"',
                "Output:",
            ]
        )

    def _load_gemini_model(self):
        self.last_error = ""
        api_key = os.getenv(self.api_key_env) or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            api_key = self._load_api_key_from_file()
        if not api_key:
            self.last_error = "gemini_api_key_missing"
            return None

        try:
            from google import genai  # type: ignore
            from google.genai import types  # type: ignore
        except Exception as exc:
            self.last_error = f"gemini_sdk_missing: install google-genai ({type(exc).__name__}: {exc})"
            return None

        try:
            client = genai.Client(
                api_key=api_key,
                http_options=types.HttpOptions(timeout=int(self.request_timeout_seconds * 1000)),
            )
            return types, client
        except Exception as exc:
            self.last_error = f"gemini_config_error: {type(exc).__name__}: {exc}"
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
        try:
            candidate = getattr(response, "text", None)
        except Exception:
            candidate = None
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

        types, client = model_handle
        prompt = self.build_prompt(constraint)
        try:
            response = client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=self.temperature,
                    max_output_tokens=self.max_output_tokens,
                    automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
                    thinking_config=types.ThinkingConfig(
                        include_thoughts=False,
                        thinking_level=types.ThinkingLevel.LOW,
                    ),
                ),
            )
        except Exception as exc:
            self.last_error = f"gemini_generate_error: {type(exc).__name__}: {exc}"
            return ""

        candidate = self._sanitize_output(self._extract_text(response))
        if not candidate:
            self.last_error = "gemini_empty_response"
        return candidate

    def generate(self, constraint: Constraint) -> str:
        self.last_error = ""
        if not self.use_gemini:
            self.last_error = "gemini_disabled"
            return ""

        if self.use_gemini:
            candidate = self._generate_with_gemini(constraint)
            if candidate:
                return candidate

        _prompt = self.build_prompt(constraint)
        _ = _prompt  # keep the prompt construction explicit for traceability
        return ""
