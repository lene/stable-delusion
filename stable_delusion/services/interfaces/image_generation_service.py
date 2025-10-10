"""Abstract service interface for image generation."""

__author__ = "Lene Preuss <lene.preuss@gmail.com>"

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List

from stable_delusion.models.requests import GenerateImageRequest
from stable_delusion.models.responses import GenerateImageResponse


class ImageGenerationService(ABC):
    """Abstract service interface for image generation."""

    @abstractmethod
    def generate_image(self, request: GenerateImageRequest) -> GenerateImageResponse:
        """
        Generate an image based on the provided request.

        Args:
            request: Image generation request containing prompt and parameters

        Returns:
            Response containing generated image path and metadata

        Raises:
            ImageGenerationError: If generation fails
            ValidationError: If request is invalid
        """

    @abstractmethod
    def upload_files(self, image_paths: List[Path]) -> List[str]:
        """
        Upload reference images to the service.

        Args:
            image_paths: List of paths to image files

        Returns:
            List of upload identifiers

        Raises:
            FileOperationError: If file upload fails
        """
