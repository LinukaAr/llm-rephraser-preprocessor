import re
from typing import List


def collapse_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def word_tokens(text: str) -> List[str]:
    return re.findall(r"\w+", text.lower())
