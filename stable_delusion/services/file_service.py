"""
Concrete implementation of file operations service using repositories.
Provides file I/O operations with proper validation and error handling.
"""

__author__ = "Lene Preuss <lene.preuss@gmail.com>"

from pathlib import Path

from PIL import Image

from stable_delusion.repositories.interfaces import ImageRepository, FileRepository
from stable_delusion.services.interfaces import FileService


class LocalFileService(FileService):
    """Concrete implementation of file operations using repositories."""

    def __init__(self, image_repository: ImageRepository, file_repository: FileRepository) -> None:
        self.image_repository = image_repository
        self.file_repository = file_repository

    def save_image(self, image: Image.Image, file_path: Path) -> Path:
        return self.image_repository.save_image(image, file_path)

    def load_image(self, file_path: Path) -> Image.Image:
        return self.image_repository.load_image(file_path)

    def validate_image_file(self, file_path: Path) -> bool:
        return self.image_repository.validate_image_file(file_path)
