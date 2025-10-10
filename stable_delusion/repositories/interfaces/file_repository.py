"""Abstract repository interface for file operations."""

__author__ = "Lene Preuss <lene.preuss@gmail.com>"

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional

from werkzeug.datastructures import FileStorage


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
