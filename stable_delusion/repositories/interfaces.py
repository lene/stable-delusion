"""
Repository interface definitions for NanoAPIClient.
Defines abstract base classes for data persistence operations.
"""

__author__ = "Lene Preuss <lene.preuss@gmail.com>"

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional, TYPE_CHECKING

from PIL import Image
from werkzeug.datastructures import FileStorage

if TYPE_CHECKING:
    from stable_delusion.models.metadata import GenerationMetadata


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


class FileRepository(ABC):
    """Abstract repository interface for file operations including uploads."""

    @abstractmethod
    def exists(self, file_path: Path) -> bool:
        pass

    @abstractmethod
    def create_directory(self, dir_path: Path) -> Path:
        pass

    @abstractmethod
    def delete_file(self, file_path: Path) -> bool:
        pass

    @abstractmethod
    def move_file(self, source: Path, destination: Path) -> Path:
        pass

    @abstractmethod
    def save_uploaded_files(self, files: List[FileStorage], upload_dir: Path) -> List[Path]:
        pass

    @abstractmethod
    def generate_secure_filename(
        self, filename: Optional[str], timestamp: Optional[str] = None
    ) -> str:
        pass

    @abstractmethod
    def cleanup_old_uploads(self, upload_dir: Path, max_age_hours: int = 24) -> int:
        pass

    @abstractmethod
    def validate_uploaded_file(self, file: FileStorage) -> bool:
        pass


class MetadataRepository(ABC):
    """Abstract repository interface for metadata storage and retrieval operations."""

    @abstractmethod
    def save_metadata(self, metadata: "GenerationMetadata") -> str:
        pass

    @abstractmethod
    def load_metadata(self, metadata_key: str) -> "GenerationMetadata":
        pass

    @abstractmethod
    def metadata_exists(self, content_hash: str) -> Optional[str]:
        pass

    @abstractmethod
    def list_metadata_by_hash_prefix(self, hash_prefix: str) -> List[str]:
        pass
