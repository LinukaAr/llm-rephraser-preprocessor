from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple


@dataclass
class Constraint:
    """Structured representation of a detected instruction constraint."""

    text: str
    cue: Optional[str] = None
    cue_span: Optional[Tuple[int, int]] = None
    scope_span: Optional[Tuple[int, int]] = None
    constraint_type: Optional[str] = None
    detected_negation: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
