"""
Concrete implementation of image upscaling service using Google Vertex AI.
Wraps the existing upscale functionality in a service interface.
"""

__author__ = "Lene Preuss <lene.preuss@gmail.com>"

from typing import Optional

from PIL import Image

from nano_api.config import ConfigManager
from nano_api.models.requests import UpscaleImageRequest
from nano_api.models.responses import UpscaleImageResponse
from nano_api.services.interfaces import ImageUpscalingService
from nano_api.upscale import upscale_image


class VertexAIUpscalingService(ImageUpscalingService):
    """Concrete implementation of image upscaling using Vertex AI."""

    def __init__(self, project_id: Optional[str] = None,
                 location: Optional[str] = None) -> None:
        """
        Initialize with optional project configuration.

        Args:
            project_id: Google Cloud project ID
            location: Google Cloud region
        """
        config = ConfigManager.get_config()
        self.project_id = project_id or config.project_id
        self.location = location or config.location

    @classmethod
    def create(cls, project_id: Optional[str] = None,
               location: Optional[str] = None) -> 'VertexAIUpscalingService':
        """
        Create service instance with default configuration.

        Args:
            project_id: Google Cloud project ID
            location: Google Cloud region

        Returns:
            Configured service instance
        """
        return cls(project_id=project_id, location=location)

    def upscale_image(self, request: UpscaleImageRequest) -> UpscaleImageResponse:
        """
        Upscale an image using Vertex AI.

        Args:
            request: Upscaling request containing image and scale parameters

        Returns:
            Response containing upscaled image and metadata

        Raises:
            UpscalingError: If upscaling fails
            AuthenticationError: If authentication fails
        """
        # Use the existing upscale_image function
        upscaled_image: Image.Image = upscale_image(
            request.image_path,
            request.project_id or self.project_id,
            request.location or self.location,
            upscale_factor=request.scale_factor
        )

        # For now, we don't save the upscaled image to a file
        # The response will contain the PIL Image object itself
        upscaled_file = None

        # Create response DTO
        return UpscaleImageResponse(
            upscaled_file=upscaled_file,
            original_file=request.image_path,
            scale_factor=request.scale_factor,
            project_id=request.project_id or self.project_id,
            location=request.location or self.location
        )
