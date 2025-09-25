"""
Concrete implementation of file operations service using repositories.
Provides file I/O operations with proper validation and error handling.
"""

__author__ = "Lene Preuss <lene.preuss@gmail.com>"

from pathlib import Path

from PIL import Image

from nano_api.repositories.interfaces import ImageRepository, FileRepository
from nano_api.services.interfaces import FileService


class LocalFileService(FileService):
    """Concrete implementation of file operations using repositories."""

    def __init__(self, image_repository: ImageRepository,
                 file_repository: FileRepository) -> None:
        """
        Initialize with repository dependencies.

        Args:
            image_repository: Repository for image operations
            file_repository: Repository for generic file operations
        """
        self.image_repository = image_repository
        self.file_repository = file_repository

    def save_image(self, image: Image.Image, file_path: Path) -> Path:
        """
        Save an image using the image repository.

        Args:
            image: PIL Image object to save
            file_path: Destination path for the image

        Returns:
            Path where the image was saved

        Raises:
            FileOperationError: If save operation fails
        """
        return self.image_repository.save_image(image, file_path)

    def load_image(self, file_path: Path) -> Image.Image:
        """
        Load an image using the image repository.

        Args:
            file_path: Path to the image file

        Returns:
            PIL Image object

        Raises:
            FileOperationError: If load operation fails
        """
        return self.image_repository.load_image(file_path)

    def validate_image_file(self, file_path: Path) -> bool:
        """
        Validate an image file using the image repository.

        Args:
            file_path: Path to validate

        Returns:
            True if file is a valid image

        Raises:
            FileOperationError: If validation fails
        """
        return self.image_repository.validate_image_file(file_path)
