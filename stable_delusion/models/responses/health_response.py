"""Health check response class."""

__author__ = "Lene Preuss <lene.preuss@gmail.com>"

from dataclasses import dataclass

from stable_delusion.models.responses.base_response import BaseResponse


@dataclass
class HealthResponse(BaseResponse):
    """Response DTO for health check endpoint."""

    service: str
    version: str
    status: str

    def __init__(
        self, service: str = "NanoAPIClient", version: str = "1.0.0", status: str = "healthy"
    ) -> None:
        super().__init__(success=True, message=f"Service {status}")
        self.service = service
        self.version = version
        self.status = status
