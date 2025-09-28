"""
Service factory for creating service instances with dependencies.
Provides centralized service creation and dependency injection.
"""

__author__ = "Lene Preuss <lene.preuss@gmail.com>"

import logging
from pathlib import Path
from typing import Optional

from stable_delusion.factories.repository_factory import RepositoryFactory
from stable_delusion.services.file_service import LocalFileService
from stable_delusion.services.gemini_service import GeminiImageGenerationService
from stable_delusion.services.interfaces import (
    FileService as FileServiceInterface,
    ImageGenerationService,
    ImageUpscalingService,
)
from stable_delusion.services.upscaling_service import VertexAIUpscalingService


class ServiceFactory:
    """Factory for creating service instances with proper dependencies."""

    @staticmethod
    def create_file_service() -> FileServiceInterface:
        image_repo, file_repo, _ = RepositoryFactory.create_all_repositories()
        return LocalFileService(image_repository=image_repo, file_repository=file_repo)

    @staticmethod
    def create_image_generation_service(
        project_id: Optional[str] = None,
        location: Optional[str] = None,
        output_dir: Optional[Path] = None,
        storage_type: Optional[str] = None,
        model: Optional[str] = None,
    ) -> ImageGenerationService:
        """
        Create image generation service with dependencies.

        Args:
            project_id: Google Cloud project ID
            location: Google Cloud region
            output_dir: Output directory for generated images
            storage_type: Storage backend type ('local' or 's3')
            model: Model to use ('gemini' or 'seedream')

        Returns:
            Configured image generation service instance
        """
        # Handle storage type override
        if storage_type:
            from stable_delusion.config import ConfigManager

            config = ConfigManager.get_config()
            original_storage_type = config.storage_type
            config.storage_type = storage_type
            image_repo = RepositoryFactory.create_image_repository()
            config.storage_type = original_storage_type
        else:
            image_repo = RepositoryFactory.create_image_repository()

        # Create service based on model selection
        model = model or "gemini"  # Default to gemini for backward compatibility

        logging.info("🏭 ServiceFactory creating service for model: %s", model)

        if model == "seedream":
            logging.info("🌱 Creating SeedreamImageGenerationService")
            from stable_delusion.services.seedream_service import SeedreamImageGenerationService

            service = SeedreamImageGenerationService.create(
                output_dir=output_dir, image_repository=image_repo
            )
            logging.info("✅ SeedreamImageGenerationService created: %s", service)
            return service
        # Default to Gemini
        logging.info("🔷 Creating GeminiImageGenerationService")
        gemini_service = GeminiImageGenerationService.create(
            project_id=project_id,
            location=location,
            output_dir=output_dir,
            image_repository=image_repo,
        )
        logging.info("✅ GeminiImageGenerationService created: %s", gemini_service)
        return gemini_service

    @staticmethod
    def create_upscaling_service(
        project_id: Optional[str] = None, location: Optional[str] = None
    ) -> ImageUpscalingService:
        """
        Create image upscaling service.

        Args:
            project_id: Google Cloud project ID
            location: Google Cloud region

        Returns:
            Configured image upscaling service instance
        """
        return VertexAIUpscalingService.create(project_id=project_id, location=location)

    @classmethod
    def create_all_services(
        cls,
        project_id: Optional[str] = None,
        location: Optional[str] = None,
        output_dir: Optional[Path] = None,
    ) -> tuple[FileServiceInterface, ImageGenerationService, ImageUpscalingService]:
        """
        Create all service instances with dependencies.

        Args:
            project_id: Google Cloud project ID
            location: Google Cloud region
            output_dir: Output directory for generated images

        Returns:
            Tuple of (file_service, generation_service, upscaling_service)
        """
        return (
            cls.create_file_service(),
            cls.create_image_generation_service(
                project_id=project_id, location=location, output_dir=output_dir
            ),
            cls.create_upscaling_service(project_id=project_id, location=location),
        )
