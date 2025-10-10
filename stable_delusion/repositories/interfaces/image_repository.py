"""Abstract repository interface for image storage operations."""

__author__ = "Lene Preuss <lene.preuss@gmail.com>"

from abc import ABC, abstractmethod
from pathlib import Path

from PIL import Image


class ImageRepository(ABC):
    """Abstract repository interface for image storage and retrieval operations."""

    @abstractmethod
    def save_image(self, image: Image.Image, file_path: Path) -> Path:
        pass

    @abstractmethod
    def load_image(self, file_path: Path) -> Image.Image:
        pass

    @abstractmethod
    def validate_image_file(self, file_path: Path) -> bool:
        pass

    @abstractmethod
    def generate_image_path(self, base_name: str, output_dir: Path) -> Path:
        pass
