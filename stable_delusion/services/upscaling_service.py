"""
Concrete implementation of image upscaling service using Google Vertex AI.
Wraps the existing upscale functionality in a service interface.
"""

__author__ = "Lene Preuss <lene.preuss@gmail.com>"

from typing import Optional

from stable_delusion.config import ConfigManager
from stable_delusion.models.requests import UpscaleImageRequest
from stable_delusion.models.responses import UpscaleImageResponse
from stable_delusion.models.client_config import GCPConfig
from stable_delusion.services.interfaces import ImageUpscalingService
from stable_delusion.upscale import upscale_image


class VertexAIUpscalingService(ImageUpscalingService):
    """Concrete implementation of image upscaling using Vertex AI."""

    def __init__(self, project_id: Optional[str] = None, location: Optional[str] = None) -> None:
        config = ConfigManager.get_config()
        self.project_id = project_id or config.project_id
        self.location = location or config.location

    @classmethod
    def create(
        cls, project_id: Optional[str] = None, location: Optional[str] = None
    ) -> "VertexAIUpscalingService":
        return cls(project_id=project_id, location=location)

    def upscale_image(self, request: UpscaleImageRequest) -> UpscaleImageResponse:
        # Use the existing upscale_image function
        upscale_image(
            request.image_path,
            request.project_id or self.project_id,
            request.location or self.location,
            upscale_factor=request.scale_factor,
        )

        # For now, we don't save the upscaled image to a file
        # The response will contain the PIL Image object itself
        upscaled_file = None

        # Create response DTO
        return UpscaleImageResponse(
            upscaled_file=upscaled_file,
            original_file=request.image_path,
            scale_factor=request.scale_factor,
            gcp_config=GCPConfig(
                project_id=request.project_id or self.project_id,
                location=request.location or self.location,
            ),
        )
