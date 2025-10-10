"""API information response class."""

__author__ = "Lene Preuss <lene.preuss@gmail.com>"

from dataclasses import dataclass
from typing import Dict

from stable_delusion.models.responses.base_response import BaseResponse


@dataclass
class APIInfoResponse(BaseResponse):
    """Response DTO for API information endpoint."""

    name: str
    description: str
    version: str
    endpoints: Dict[str, str]

    def __init__(self) -> None:
        super().__init__(success=True, message="API information retrieved")
        self.name = "NanoAPIClient API"
        self.description = "Flask web API for image generation using Google Gemini AI"
        self.version = "1.0.0"
        self.endpoints = {
            "/": "API information",
            "/health": "Health check",
            "/generate": "Generate images from prompt and reference images",
            "/openapi.json": "OpenAPI specification",
        }
