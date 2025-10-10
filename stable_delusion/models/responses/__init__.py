"""
Response DTOs for NanoAPIClient API endpoints.
Defines the structure of API responses with consistent formatting.
"""

__author__ = "Lene Preuss <lene.preuss@gmail.com>"

from stable_delusion.models.responses.base_response import BaseResponse
from stable_delusion.models.responses.error_response import ErrorResponse
from stable_delusion.models.responses.generate_image_response import GenerateImageResponse
from stable_delusion.models.responses.upscale_image_response import UpscaleImageResponse
from stable_delusion.models.responses.health_response import HealthResponse
from stable_delusion.models.responses.api_info_response import APIInfoResponse

__all__ = [
    "BaseResponse",
    "ErrorResponse",
    "GenerateImageResponse",
    "UpscaleImageResponse",
    "HealthResponse",
    "APIInfoResponse",
]
