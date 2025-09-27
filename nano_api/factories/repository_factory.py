"""
Repository factory for creating repository instances.
Provides centralized repository creation and configuration with support for multiple
storage backends.
"""

__author__ = "Lene Preuss <lene.preuss@gmail.com>"

from nano_api.config import ConfigManager
from nano_api.repositories.interfaces import (
    ImageRepository,
    FileRepository,
    UploadRepository,
    MetadataRepository,
)
from nano_api.repositories.local_file_repository import LocalFileRepository
from nano_api.repositories.local_image_repository import LocalImageRepository
from nano_api.repositories.upload_repository import LocalUploadRepository


class RepositoryFactory:
    """Factory for creating repository instances."""

    @staticmethod
    def create_image_repository() -> ImageRepository:
        config = ConfigManager.get_config()

        if config.storage_type == "s3":
            from nano_api.repositories.s3_image_repository import S3ImageRepository

            return S3ImageRepository(config)

        return LocalImageRepository()

    @staticmethod
    def create_file_repository() -> FileRepository:
        config = ConfigManager.get_config()

        if config.storage_type == "s3":
            from nano_api.repositories.s3_file_repository import S3FileRepository

            return S3FileRepository(config)

        return LocalFileRepository()

    @staticmethod
    def create_upload_repository() -> UploadRepository:
        return LocalUploadRepository()

    @staticmethod
    def create_metadata_repository() -> MetadataRepository:
        config = ConfigManager.get_config()

        if config.storage_type == "s3":
            from nano_api.repositories.s3_metadata_repository import S3MetadataRepository

            return S3MetadataRepository(config)

        from nano_api.repositories.local_metadata_repository import LocalMetadataRepository

        return LocalMetadataRepository(config)

    @classmethod
    def create_all_repositories(cls) -> tuple[ImageRepository, FileRepository, UploadRepository]:
        return (
            cls.create_image_repository(),
            cls.create_file_repository(),
            cls.create_upload_repository(),
        )

    @classmethod
    def create_all_repositories_with_metadata(
        cls,
    ) -> tuple[ImageRepository, FileRepository, UploadRepository, MetadataRepository]:
        return (
            cls.create_image_repository(),
            cls.create_file_repository(),
            cls.create_upload_repository(),
            cls.create_metadata_repository(),
        )
