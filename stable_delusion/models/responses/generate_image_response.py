"""Image generation response class."""

__author__ = "Lene Preuss <lene.preuss@gmail.com>"

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Any

from stable_delusion.models.client_config import GCPConfig, ImageGenerationConfig
from stable_delusion.models.responses.base_response import BaseResponse


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
