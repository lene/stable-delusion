"""
Token usage entry data model.
"""

__author__ = "Lene Preuss <lene.preuss@gmail.com>"

from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional


@dataclass
class TokenUsageEntry:
    """Represents a single token usage record."""

    timestamp: str
    model: str
    tokens: int
    operation: str
    prompt_hash: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
