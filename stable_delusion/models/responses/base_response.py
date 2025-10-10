"""Base response class for API responses."""

__author__ = "Lene Preuss <lene.preuss@gmail.com>"

from dataclasses import dataclass, asdict
from typing import Dict, Any


@dataclass
class BaseResponse:
    """Base response DTO with common fields."""

    success: bool
    message: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
