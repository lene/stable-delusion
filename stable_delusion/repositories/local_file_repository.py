"""
Local filesystem implementation of file repository.
Handles generic file operations on local filesystem.
"""

__author__ = "Lene Preuss <lene.preuss@gmail.com>"

from pathlib import Path

from stable_delusion.exceptions import FileOperationError
from stable_delusion.repositories.interfaces import FileRepository


class LocalFileRepository(FileRepository):
    """Local filesystem implementation of file repository."""

    def exists(self, file_path: Path) -> bool:
        """
        Check if a file exists on the local filesystem.

        Args:
            file_path: Path to check

        Returns:
            True if file exists
        """
        return file_path.exists()

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
        try:
            dir_path.mkdir(parents=True, exist_ok=True)
            return dir_path
        except (OSError, IOError) as e:
            raise FileOperationError(
                f"Failed to create directory: {dir_path}",
                file_path=str(dir_path),
                operation="create_directory",
            ) from e

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
        if not file_path.exists():
            return False

        try:
            file_path.unlink()
            return True
        except (OSError, IOError) as e:
            raise FileOperationError(
                f"Failed to delete file: {file_path}", file_path=str(file_path), operation="delete"
            ) from e

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
        try:
            # Ensure destination directory exists
            destination.parent.mkdir(parents=True, exist_ok=True)

            # Move the file
            source.rename(destination)
            return destination
        except (OSError, IOError) as e:
            raise FileOperationError(
                f"Failed to move file from {source} to {destination}",
                file_path=str(source),
                operation="move",
            ) from e
