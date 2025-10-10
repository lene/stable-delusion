"""Error response class."""

__author__ = "Lene Preuss <lene.preuss@gmail.com>"

from dataclasses import dataclass
from typing import Optional

from stable_delusion.models.responses.base_response import BaseResponse


@dataclass
class ErrorResponse(BaseResponse):
    """Response DTO for error conditions."""

    error_code: Optional[str] = None
    details: Optional[str] = None

    def __init__(
        self, message: str, error_code: Optional[str] = None, details: Optional[str] = None
    ) -> None:
        super().__init__(success=False, message=message)
        self.error_code = error_code
        self.details = details
