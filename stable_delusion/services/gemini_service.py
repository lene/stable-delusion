"""
Concrete implementation of image generation service using Google Gemini API.
Wraps the existing GeminiClient functionality in a service interface.
"""

__author__ = "Lene Preuss <lene.preuss@gmail.com>"

from pathlib import Path
from typing import List, Optional, TYPE_CHECKING

from stable_delusion.config import ConfigManager
from stable_delusion.models.requests import GenerateImageRequest
from stable_delusion.models.responses import GenerateImageResponse
from stable_delusion.models.client_config import (
    GeminiClientConfig,
    GCPConfig,
    StorageConfig,
    ImageGenerationConfig,
)
from stable_delusion.repositories.interfaces import ImageRepository
from stable_delusion.services.interfaces import ImageGenerationService

if TYPE_CHECKING:
    from stable_delusion.generate import GeminiClient


class GeminiImageGenerationService(ImageGenerationService):
    """Concrete implementation of image generation using Gemini API."""

    def __init__(
        self, client: "GeminiClient", image_repository: Optional[ImageRepository] = None
    ) -> None:
        self.client = client
        self.image_repository = image_repository

    @classmethod
    def create(
        cls,
        project_id: Optional[str] = None,
        location: Optional[str] = None,
        output_dir: Optional[Path] = None,
        image_repository: Optional[ImageRepository] = None,
    ) -> "GeminiImageGenerationService":
        from stable_delusion.generate import GeminiClient

        # Create configuration with provided parameters
        client_config = GeminiClientConfig(
            gcp=GCPConfig(project_id=project_id, location=location),
            storage=StorageConfig(output_dir=output_dir),
        )
        client = GeminiClient(client_config)
        return cls(client, image_repository)

    def generate_image(self, request: GenerateImageRequest) -> GenerateImageResponse:
        config = ConfigManager.get_config()

        # Generate the image
        generated_file = self.client.generate_hires_image_in_one_shot(
            request.prompt, request.images, scale=request.scale
        )

        # Create response DTO
        return GenerateImageResponse(
            image_config=ImageGenerationConfig(
                generated_file=generated_file,
                prompt=request.prompt,
                scale=request.scale,
                saved_files=request.images,
                output_dir=request.output_dir or config.default_output_dir,
            ),
            gcp_config=GCPConfig(
                project_id=request.project_id or config.project_id,
                location=request.location or config.location,
            ),
        )

    def upload_files(self, image_paths: List[Path]) -> List[str]:
        uploaded_files = self.client.upload_files(image_paths)
        return [str(file.uri) for file in uploaded_files]
