"""
Request DTOs for NanoAPIClient API endpoints.
Defines the structure of incoming API requests with validation.
"""

__author__ = "Lene Preuss <lene.preuss@gmail.com>"

from stable_delusion.models.requests.generate_image_request import GenerateImageRequest
from stable_delusion.models.requests.upscale_image_request import UpscaleImageRequest

__all__ = [
    "GenerateImageRequest",
    "UpscaleImageRequest",
]
