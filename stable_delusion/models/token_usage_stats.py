"""
Token usage statistics data model.
"""

__author__ = "Lene Preuss <lene.preuss@gmail.com>"

from dataclasses import dataclass, asdict
from typing import Dict, Any


@dataclass
class TokenUsageStats:
    """Aggregate token usage statistics."""

    total_tokens: int
    total_requests: int
    tokens_by_model: Dict[str, int]
    requests_by_model: Dict[str, int]
    tokens_by_operation: Dict[str, int]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
