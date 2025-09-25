"""
Response DTOs for NanoAPIClient API endpoints.
Defines the structure of API responses with consistent formatting.
"""

__author__ = "Lene Preuss <lene.preuss@gmail.com>"

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Any


@dataclass
class BaseResponse:
    """Base response DTO with common fields."""

    success: bool
    message: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert response to dictionary for JSON serialization."""
        return asdict(self)


@dataclass
class ErrorResponse(BaseResponse):
    """Response DTO for error conditions."""

    error_code: Optional[str] = None
    details: Optional[str] = None

    def __init__(self, message: str, error_code: Optional[str] = None,
                 details: Optional[str] = None) -> None:
        """Initialize error response."""
        super().__init__(success=False, message=message)
        self.error_code = error_code
        self.details = details


@dataclass
class GenerateImageResponse(BaseResponse):  # pylint: disable=too-many-instance-attributes
    """Response DTO for image generation endpoint."""

    generated_file: Optional[Path]
    prompt: str
    project_id: str
    location: str
    scale: Optional[int]
    saved_files: List[Path]
    output_dir: Path
    upscaled: bool

    def __init__(self, *, generated_file: Optional[Path], prompt: str,
                 project_id: str, location: str, scale: Optional[int],
                 saved_files: List[Path], output_dir: Path) -> None:
        # pylint: disable=too-many-arguments
        """Initialize generation response."""
        super().__init__(
            success=generated_file is not None,
            message="Image generated successfully" if generated_file
            else "Image generation failed"
        )
        self.generated_file = generated_file
        self.prompt = prompt
        self.project_id = project_id
        self.location = location
        self.scale = scale
        self.saved_files = saved_files
        self.output_dir = output_dir
        self.upscaled = scale is not None

    def to_dict(self) -> Dict[str, Any]:
        """Convert response to dictionary for JSON serialization."""
        data = super().to_dict()
        # Convert Path objects to strings for JSON serialization
        if self.generated_file:
            data["generated_file"] = str(self.generated_file)
        data["saved_files"] = [str(f) for f in self.saved_files]
        data["output_dir"] = str(self.output_dir)
        return data


@dataclass
class UpscaleImageResponse(BaseResponse):
    """Response DTO for image upscaling."""

    upscaled_file: Optional[Path]
    original_file: Path
    scale_factor: str
    project_id: str
    location: str

    def __init__(self, *, upscaled_file: Optional[Path], original_file: Path,
                 scale_factor: str, project_id: str, location: str) -> None:
        # pylint: disable=too-many-arguments
        """Initialize upscaling response."""
        super().__init__(
            success=upscaled_file is not None,
            message="Image upscaled successfully" if upscaled_file
            else "Image upscaling failed"
        )
        self.upscaled_file = upscaled_file
        self.original_file = original_file
        self.scale_factor = scale_factor
        self.project_id = project_id
        self.location = location

    def to_dict(self) -> Dict[str, Any]:
        """Convert response to dictionary for JSON serialization."""
        data = super().to_dict()
        # Convert Path objects to strings for JSON serialization
        if self.upscaled_file:
            data["upscaled_file"] = str(self.upscaled_file)
        data["original_file"] = str(self.original_file)
        return data


@dataclass
class HealthResponse(BaseResponse):
    """Response DTO for health check endpoint."""

    service: str
    version: str
    status: str

    def __init__(self, service: str = "NanoAPIClient", version: str = "1.0.0",
                 status: str = "healthy") -> None:
        """Initialize health response."""
        super().__init__(success=True, message=f"Service {status}")
        self.service = service
        self.version = version
        self.status = status


@dataclass
class APIInfoResponse(BaseResponse):
    """Response DTO for API information endpoint."""

    name: str
    description: str
    version: str
    endpoints: Dict[str, str]

    def __init__(self) -> None:
        """Initialize API info response."""
        super().__init__(success=True, message="API information retrieved")
        self.name = "NanoAPIClient API"
        self.description = "Flask web API for image generation using Google Gemini AI"
        self.version = "1.0.0"
        self.endpoints = {
            "/": "API information",
            "/health": "Health check",
            "/generate": "Generate images from prompt and reference images",
            "/openapi.json": "OpenAPI specification"
        }
