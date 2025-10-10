"""Image upscaling response class."""

__author__ = "Lene Preuss <lene.preuss@gmail.com>"

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Any

from stable_delusion.models.client_config import GCPConfig
from stable_delusion.models.responses.base_response import BaseResponse


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
