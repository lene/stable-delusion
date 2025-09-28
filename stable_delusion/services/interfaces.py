"""
Service interface definitions for NanoAPIClient.
Defines abstract base classes for external service integrations.
"""

__author__ = "Lene Preuss <lene.preuss@gmail.com>"

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List

from stable_delusion.models.requests import GenerateImageRequest, UpscaleImageRequest
from stable_delusion.models.responses import GenerateImageResponse, UpscaleImageResponse


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


class ImageUpscalingService(ABC):
    """Abstract service interface for image upscaling."""

    @abstractmethod
    def upscale_image(self, request: UpscaleImageRequest) -> UpscaleImageResponse:
        """
        Upscale an image based on the provided request.

        Args:
            request: Upscaling request containing image and scale parameters

        Returns:
            Response containing upscaled image and metadata

        Raises:
            UpscalingError: If upscaling fails
            AuthenticationError: If authentication fails
        """
