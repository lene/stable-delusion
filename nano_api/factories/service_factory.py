"""
Service factory for creating service instances with dependencies.
Provides centralized service creation and dependency injection.
"""

__author__ = "Lene Preuss <lene.preuss@gmail.com>"

from pathlib import Path
from typing import Optional

from nano_api.factories.repository_factory import RepositoryFactory
from nano_api.services.file_service import LocalFileService
from nano_api.services.gemini_service import GeminiImageGenerationService
from nano_api.services.interfaces import (
    FileService as FileServiceInterface,
    ImageGenerationService,
    ImageUpscalingService
)
from nano_api.services.upscaling_service import VertexAIUpscalingService


class ServiceFactory:
    """Factory for creating service instances with proper dependencies."""

    @staticmethod
    def create_file_service() -> FileServiceInterface:
        """
        Create file service with repository dependencies.

        Returns:
            Configured file service instance
        """
        image_repo, file_repo, _ = RepositoryFactory.create_all_repositories()
        return LocalFileService(
            image_repository=image_repo,
            file_repository=file_repo
        )

    @staticmethod
    def create_image_generation_service(
            project_id: Optional[str] = None,
            location: Optional[str] = None,
            output_dir: Optional[Path] = None) -> ImageGenerationService:
        """
        Create image generation service with dependencies.

        Args:
            project_id: Google Cloud project ID
            location: Google Cloud region
            output_dir: Output directory for generated images

        Returns:
            Configured image generation service instance
        """
        image_repo = RepositoryFactory.create_image_repository()
        return GeminiImageGenerationService.create(
            project_id=project_id,
            location=location,
            output_dir=output_dir,
            image_repository=image_repo
        )

    @staticmethod
    def create_upscaling_service(
            project_id: Optional[str] = None,
            location: Optional[str] = None) -> ImageUpscalingService:
        """
        Create image upscaling service.

        Args:
            project_id: Google Cloud project ID
            location: Google Cloud region

        Returns:
            Configured image upscaling service instance
        """
        return VertexAIUpscalingService.create(
            project_id=project_id,
            location=location
        )

    @classmethod
    def create_all_services(
            cls,
            project_id: Optional[str] = None,
            location: Optional[str] = None,
            output_dir: Optional[Path] = None) -> tuple[FileServiceInterface,
                                                        ImageGenerationService,
                                                        ImageUpscalingService]:
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
                project_id=project_id,
                location=location,
                output_dir=output_dir
            ),
            cls.create_upscaling_service(
                project_id=project_id,
                location=location
            )
        )
