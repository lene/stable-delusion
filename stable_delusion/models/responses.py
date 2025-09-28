"""
Response DTOs for NanoAPIClient API endpoints.
Defines the structure of API responses with consistent formatting.
"""

__author__ = "Lene Preuss <lene.preuss@gmail.com>"

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Any

from stable_delusion.models.client_config import GCPConfig, ImageGenerationConfig


@dataclass
class BaseResponse:
    """Base response DTO with common fields."""

    success: bool
    message: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


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


@dataclass
class GenerateImageResponse(BaseResponse):
    """Response DTO for image generation endpoint."""

    image_config: ImageGenerationConfig
    gcp_config: GCPConfig
    upscaled: bool

    def __init__(self, *, image_config: ImageGenerationConfig, gcp_config: GCPConfig) -> None:
        super().__init__(
            success=image_config.generated_file is not None,
            message=(
                "Image generated successfully"
                if image_config.generated_file
                else "Image generation failed"
            ),
        )
        self.image_config = image_config
        self.gcp_config = gcp_config
        self.upscaled = image_config.scale is not None

    @property
    def generated_file(self) -> Optional[Path]:
        return self.image_config.generated_file

    @property
    def prompt(self) -> str:
        return self.image_config.prompt

    @property
    def scale(self) -> Optional[int]:
        return self.image_config.scale

    @property
    def saved_files(self) -> List[Path]:
        return self.image_config.saved_files or []

    @property
    def output_dir(self) -> Optional[Path]:
        return self.image_config.output_dir

    @property
    def project_id(self) -> Optional[str]:
        return self.gcp_config.project_id

    @property
    def location(self) -> Optional[str]:
        return self.gcp_config.location

    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        # Convert Path objects to strings for JSON serialization
        if self.generated_file:
            data["generated_file"] = str(self.generated_file)
        data["saved_files"] = [str(f) for f in self.saved_files]
        if self.output_dir:
            data["output_dir"] = str(self.output_dir)

        # Flatten image config for API backward compatibility
        data["prompt"] = self.prompt
        data["scale"] = self.scale
        data["upscaled"] = self.upscaled

        # Flatten GCP config for API backward compatibility
        data["project_id"] = self.gcp_config.project_id
        data["location"] = self.gcp_config.location

        return data


@dataclass
class UpscaleImageResponse(BaseResponse):
    """Response DTO for image upscaling."""

    upscaled_file: Optional[Path]
    original_file: Path
    scale_factor: str
    gcp_config: GCPConfig

    def __init__(
        self,
        *,
        upscaled_file: Optional[Path],
        original_file: Path,
        scale_factor: str,
        gcp_config: GCPConfig,
    ) -> None:
        super().__init__(
            success=upscaled_file is not None,
            message="Image upscaled successfully" if upscaled_file else "Image upscaling failed",
        )
        self.upscaled_file = upscaled_file
        self.original_file = original_file
        self.scale_factor = scale_factor
        self.gcp_config = gcp_config

    @property
    def project_id(self) -> Optional[str]:
        return self.gcp_config.project_id

    @property
    def location(self) -> Optional[str]:
        return self.gcp_config.location

    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        # Convert Path objects to strings for JSON serialization
        if self.upscaled_file:
            data["upscaled_file"] = str(self.upscaled_file)
        data["original_file"] = str(self.original_file)

        # Flatten GCP config for API backward compatibility
        data["project_id"] = self.gcp_config.project_id
        data["location"] = self.gcp_config.location

        return data


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
