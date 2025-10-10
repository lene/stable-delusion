"""Abstract service interface for image upscaling."""

__author__ = "Lene Preuss <lene.preuss@gmail.com>"

from abc import ABC, abstractmethod

from stable_delusion.models.requests import UpscaleImageRequest
from stable_delusion.models.responses import UpscaleImageResponse


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
