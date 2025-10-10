"""
Service interface definitions for NanoAPIClient.
Defines abstract base classes for external service integrations.
"""

__author__ = "Lene Preuss <lene.preuss@gmail.com>"

from stable_delusion.services.interfaces.image_generation_service import (
    ImageGenerationService,
)
from stable_delusion.services.interfaces.image_upscaling_service import ImageUpscalingService

__all__ = [
    "ImageGenerationService",
    "ImageUpscalingService",
]
