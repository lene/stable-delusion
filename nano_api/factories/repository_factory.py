"""
Repository factory for creating repository instances.
Provides centralized repository creation and configuration.
"""

__author__ = "Lene Preuss <lene.preuss@gmail.com>"

from nano_api.repositories.interfaces import (
    ImageRepository, FileRepository, UploadRepository
)
from nano_api.repositories.local_file_repository import LocalFileRepository
from nano_api.repositories.local_image_repository import LocalImageRepository
from nano_api.repositories.upload_repository import LocalUploadRepository


class RepositoryFactory:
    """Factory for creating repository instances."""

    @staticmethod
    def create_image_repository() -> ImageRepository:
        """
        Create image repository instance.

        Returns:
            Configured image repository instance
        """
        return LocalImageRepository()

    @staticmethod
    def create_file_repository() -> FileRepository:
        """
        Create file repository instance.

        Returns:
            Configured file repository instance
        """
        return LocalFileRepository()

    @staticmethod
    def create_upload_repository() -> UploadRepository:
        """
        Create upload repository instance.

        Returns:
            Configured upload repository instance
        """
        return LocalUploadRepository()

    @classmethod
    def create_all_repositories(cls) -> tuple[ImageRepository,
                                              FileRepository,
                                              UploadRepository]:
        """
        Create all repository instances at once.

        Returns:
            Tuple of (image_repo, file_repo, upload_repo)
        """
        return (
            cls.create_image_repository(),
            cls.create_file_repository(),
            cls.create_upload_repository()
        )
