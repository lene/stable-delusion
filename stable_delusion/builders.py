"""
Simple builder functions to replace factory pattern.
Provides direct instantiation with clear dependencies.
"""

__author__ = "Lene Preuss <lene.preuss@gmail.com>"

import logging
from pathlib import Path
from typing import Optional

from stable_delusion.config import ConfigManager
from stable_delusion.repositories.interfaces import (
    ImageRepository,
    FileRepository,
    MetadataRepository,
)
from stable_delusion.services.interfaces import (
    FileService as FileServiceInterface,
    ImageGenerationService,
    ImageUpscalingService,
)


def create_image_repository(storage_type: Optional[str] = None) -> ImageRepository:
    """Create image repository based on storage type."""
    config = ConfigManager.get_config()
    storage = storage_type or config.storage_type

    if storage == "s3":
        from stable_delusion.repositories.s3_image_repository import S3ImageRepository
        return S3ImageRepository(config)

    from stable_delusion.repositories.local_image_repository import LocalImageRepository
    return LocalImageRepository()


def create_file_repository(storage_type: Optional[str] = None) -> FileRepository:
    """Create file repository based on storage type."""
    config = ConfigManager.get_config()
    storage = storage_type or config.storage_type

    if storage == "s3":
        from stable_delusion.repositories.s3_file_repository import S3FileRepository
        return S3FileRepository(config)

    from stable_delusion.repositories.local_file_repository import LocalFileRepository
    return LocalFileRepository()


def create_metadata_repository(storage_type: Optional[str] = None) -> MetadataRepository:
    """Create metadata repository based on storage type."""
    config = ConfigManager.get_config()
    storage = storage_type or config.storage_type

    if storage == "s3":
        from stable_delusion.repositories.s3_metadata_repository import S3MetadataRepository
        return S3MetadataRepository(config)

    from stable_delusion.repositories.local_metadata_repository import LocalMetadataRepository
    return LocalMetadataRepository(config)


def create_file_service() -> FileServiceInterface:
    """Create file service with repositories."""
    from stable_delusion.services.file_service import LocalFileService

    image_repo = create_image_repository()
    file_repo = create_file_repository()
    return LocalFileService(image_repository=image_repo, file_repository=file_repo)


def create_image_generation_service(
    project_id: Optional[str] = None,
    location: Optional[str] = None,
    output_dir: Optional[Path] = None,
    storage_type: Optional[str] = None,
    model: Optional[str] = None,
) -> ImageGenerationService:
    """Create image generation service based on model."""
    image_repo = create_image_repository(storage_type)
    model = model or "gemini"  # Default to gemini for backward compatibility

    logging.info("🏗️ Creating image generation service for model: %s", model)

    if model == "seedream":
        from stable_delusion.services.seedream_service import SeedreamImageGenerationService
        logging.info("🌱 Creating SeedreamImageGenerationService")
        service = SeedreamImageGenerationService.create(
            output_dir=output_dir, image_repository=image_repo
        )
        logging.info("✅ SeedreamImageGenerationService created")
        return service

    from stable_delusion.services.gemini_service import GeminiImageGenerationService
    logging.info("🔷 Creating GeminiImageGenerationService")
    gemini_service = GeminiImageGenerationService.create(
        project_id=project_id,
        location=location,
        output_dir=output_dir,
        image_repository=image_repo,
    )
    logging.info("✅ GeminiImageGenerationService created")
    return gemini_service


def create_upscaling_service(
    project_id: Optional[str] = None, location: Optional[str] = None
) -> ImageUpscalingService:
    """Create upscaling service."""
    from stable_delusion.services.upscaling_service import VertexAIUpscalingService
    return VertexAIUpscalingService.create(project_id=project_id, location=location)


def create_all_services(
    project_id: Optional[str] = None,
    location: Optional[str] = None,
    output_dir: Optional[Path] = None,
) -> tuple[FileServiceInterface, ImageGenerationService, ImageUpscalingService]:
    """Create all services."""
    return (
        create_file_service(),
        create_image_generation_service(
            project_id=project_id, location=location, output_dir=output_dir
        ),
        create_upscaling_service(project_id=project_id, location=location),
    )
