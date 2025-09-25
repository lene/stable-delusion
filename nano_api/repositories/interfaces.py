"""
Repository interface definitions for NanoAPIClient.
Defines abstract base classes for data persistence operations.
"""

__author__ = "Lene Preuss <lene.preuss@gmail.com>"

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional

from PIL import Image
from werkzeug.datastructures import FileStorage


class ImageRepository(ABC):
    """Abstract repository interface for image storage and retrieval operations."""

    @abstractmethod
    def save_image(self, image: Image.Image, file_path: Path) -> Path:
        """
        Save an image to the specified path.

        Args:
            image: PIL Image object to save
            file_path: Destination path for the image

        Returns:
            Path where the image was saved

        Raises:
            FileOperationError: If save operation fails
        """

    @abstractmethod
    def load_image(self, file_path: Path) -> Image.Image:
        """
        Load an image from the specified path.

        Args:
            file_path: Path to the image file

        Returns:
            PIL Image object

        Raises:
            FileOperationError: If load operation fails
        """

    @abstractmethod
    def validate_image_file(self, file_path: Path) -> bool:
        """
        Validate that a file is a readable image.

        Args:
            file_path: Path to validate

        Returns:
            True if file is a valid image

        Raises:
            FileOperationError: If validation fails
        """

    @abstractmethod
    def generate_image_path(self, base_name: str, output_dir: Path) -> Path:
        """
        Generate a unique image file path with timestamp.

        Args:
            base_name: Base name for the file
            output_dir: Directory to save the image

        Returns:
            Generated unique file path
        """


class FileRepository(ABC):
    """Abstract repository interface for generic file operations."""

    @abstractmethod
    def exists(self, file_path: Path) -> bool:
        """
        Check if a file exists.

        Args:
            file_path: Path to check

        Returns:
            True if file exists
        """

    @abstractmethod
    def create_directory(self, dir_path: Path) -> Path:
        """
        Create a directory if it doesn't exist.

        Args:
            dir_path: Directory path to create

        Returns:
            Created directory path

        Raises:
            FileOperationError: If directory creation fails
        """

    @abstractmethod
    def delete_file(self, file_path: Path) -> bool:
        """
        Delete a file if it exists.

        Args:
            file_path: File path to delete

        Returns:
            True if file was deleted, False if it didn't exist

        Raises:
            FileOperationError: If deletion fails
        """

    @abstractmethod
    def move_file(self, source: Path, destination: Path) -> Path:
        """
        Move a file from source to destination.

        Args:
            source: Source file path
            destination: Destination file path

        Returns:
            Destination path

        Raises:
            FileOperationError: If move operation fails
        """


class UploadRepository(ABC):
    """Abstract repository interface for handling uploaded file operations."""

    @abstractmethod
    def save_uploaded_files(self, files: List[FileStorage],
                            upload_dir: Path) -> List[Path]:
        """
        Save uploaded files to the specified directory.

        Args:
            files: List of uploaded FileStorage objects
            upload_dir: Directory to save files

        Returns:
            List of saved file paths

        Raises:
            FileOperationError: If save operation fails
        """

    @abstractmethod
    def generate_secure_filename(self, filename: Optional[str],
                                 timestamp: Optional[str] = None) -> str:
        """
        Generate a secure filename for upload.

        Args:
            filename: Original filename (can be None)
            timestamp: Optional timestamp string

        Returns:
            Secure filename string
        """

    @abstractmethod
    def cleanup_old_uploads(self, upload_dir: Path,
                            max_age_hours: int = 24) -> int:
        """
        Clean up old uploaded files.

        Args:
            upload_dir: Upload directory to clean
            max_age_hours: Maximum age of files to keep

        Returns:
            Number of files cleaned up

        Raises:
            FileOperationError: If cleanup fails
        """

    @abstractmethod
    def validate_uploaded_file(self, file: FileStorage) -> bool:
        """
        Validate an uploaded file.

        Args:
            file: FileStorage object to validate

        Returns:
            True if file is valid

        Raises:
            ValidationError: If file is invalid
        """
