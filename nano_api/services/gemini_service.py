"""
Concrete implementation of image generation service using Google Gemini API.
Wraps the existing GeminiClient functionality in a service interface.
"""

__author__ = "Lene Preuss <lene.preuss@gmail.com>"

from pathlib import Path
from typing import List, Optional, TYPE_CHECKING

from nano_api.config import ConfigManager
from nano_api.models.requests import GenerateImageRequest
from nano_api.models.responses import GenerateImageResponse
from nano_api.repositories.interfaces import ImageRepository
from nano_api.services.interfaces import ImageGenerationService

if TYPE_CHECKING:
    from nano_api.generate import GeminiClient


class GeminiImageGenerationService(ImageGenerationService):
    """Concrete implementation of image generation using Gemini API."""

    def __init__(self, client: 'GeminiClient',
                 image_repository: Optional[ImageRepository] = None) -> None:
        """
        Initialize with a configured GeminiClient.

        Args:
            client: Configured GeminiClient instance
            image_repository: Optional image repository for advanced operations
        """
        self.client = client
        self.image_repository = image_repository

    @classmethod
    def create(cls, project_id: Optional[str] = None,
               location: Optional[str] = None,
               output_dir: Optional[Path] = None,
               image_repository: Optional[ImageRepository] = None) -> \
            'GeminiImageGenerationService':
        """
        Create service instance with default configuration.

        Args:
            project_id: Google Cloud project ID
            location: Google Cloud region
            output_dir: Output directory for generated images
            image_repository: Optional image repository for advanced operations

        Returns:
            Configured service instance
        """
        from nano_api.generate import GeminiClient
        client = GeminiClient(
            project_id=project_id,
            location=location,
            output_dir=output_dir
        )
        return cls(client, image_repository)

    def generate_image(self, request: GenerateImageRequest) -> GenerateImageResponse:
        """Generate an image using Gemini API."""
        config = ConfigManager.get_config()

        # Generate the image
        generated_file = self.client.generate_hires_image_in_one_shot(
            request.prompt,
            request.images,
            scale=request.scale
        )

        # Create response DTO
        return GenerateImageResponse(
            generated_file=generated_file,
            prompt=request.prompt,
            project_id=request.project_id or config.project_id,
            location=request.location or config.location,
            scale=request.scale,
            saved_files=request.images,
            output_dir=request.output_dir or config.default_output_dir
        )

    def upload_files(self, image_paths: List[Path]) -> List[str]:
        """Upload files using the Gemini client."""
        uploaded_files = self.client.upload_files(image_paths)
        return [str(file.uri) for file in uploaded_files]
