"""
Request DTOs for NanoAPIClient API endpoints.
Defines the structure of incoming API requests with validation.
"""

__author__ = "Lene Preuss <lene.preuss@gmail.com>"

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from nano_api.exceptions import ValidationError


@dataclass
class GenerateImageRequest:
    """Request DTO for image generation endpoint."""

    prompt: str
    images: List[Path]
    project_id: Optional[str] = None
    location: Optional[str] = None
    output_dir: Optional[Path] = None
    scale: Optional[int] = None
    custom_output: Optional[str] = None

    def __post_init__(self) -> None:
        """Validate request data after initialization."""
        if not self.prompt or not self.prompt.strip():
            raise ValidationError(
                "Prompt cannot be empty",
                field="prompt",
                value=self.prompt
            )

        if not self.images:
            raise ValidationError(
                "At least one image is required",
                field="images"
            )

        # Validate scale if provided
        if self.scale is not None and self.scale not in [2, 4]:
            raise ValidationError(
                "Scale must be 2 or 4",
                field="scale",
                value=str(self.scale)
            )

        # Ensure output_dir is Path object if provided as string
        if isinstance(self.output_dir, str):
            self.output_dir = Path(self.output_dir)


@dataclass
class UpscaleImageRequest:
    """Request DTO for image upscaling."""

    image_path: Path
    scale_factor: str = "x2"
    project_id: Optional[str] = None
    location: Optional[str] = None

    def __post_init__(self) -> None:
        """Validate request data after initialization."""
        if not self.image_path:
            raise ValidationError(
                "Image path is required",
                field="image_path"
            )

        valid_scales = ["x2", "x4"]
        if self.scale_factor not in valid_scales:
            raise ValidationError(
                f"Scale factor must be one of {valid_scales}",
                field="scale_factor",
                value=self.scale_factor
            )

        # Ensure image_path is Path object if provided as string
        if isinstance(self.image_path, str):
            self.image_path = Path(self.image_path)


@dataclass
class ConfigurationRequest:
    """Request DTO for configuration updates."""

    project_id: Optional[str] = None
    location: Optional[str] = None
    upload_folder: Optional[Path] = None
    default_output_dir: Optional[Path] = None
    flask_debug: Optional[bool] = None

    def __post_init__(self) -> None:
        """Validate and normalize request data."""
        # Convert string paths to Path objects if needed
        if isinstance(self.upload_folder, str):
            self.upload_folder = Path(self.upload_folder)
        if isinstance(self.default_output_dir, str):
            self.default_output_dir = Path(self.default_output_dir)
