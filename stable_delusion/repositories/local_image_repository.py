"""
Local filesystem implementation of image repository.
Handles image storage and retrieval operations on local filesystem.
"""

__author__ = "Lene Preuss <lene.preuss@gmail.com>"

from pathlib import Path

from PIL import Image

from stable_delusion.exceptions import FileOperationError
from stable_delusion.repositories.interfaces import ImageRepository
from stable_delusion.utils import generate_timestamped_filename


class LocalImageRepository(ImageRepository):
    """Local filesystem implementation of image repository."""

    def save_image(self, image: Image.Image, file_path: Path) -> Path:
        """
        Save an image to the local filesystem.

        Args:
            image: PIL Image object to save
            file_path: Destination path for the image

        Returns:
            Path where the image was saved

        Raises:
            FileOperationError: If save operation fails
        """
        try:
            # Ensure parent directory exists
            file_path.parent.mkdir(parents=True, exist_ok=True)

            # Save the image
            image.save(str(file_path))

            return file_path
        except (OSError, IOError) as e:
            raise FileOperationError(
                f"Failed to save image to {file_path}", file_path=str(file_path), operation="save"
            ) from e

    def load_image(self, file_path: Path) -> Image.Image:
        """
        Load an image from the local filesystem.

        Args:
            file_path: Path to the image file

        Returns:
            PIL Image object

        Raises:
            FileOperationError: If load operation fails
        """
        try:
            return Image.open(file_path)
        except (FileNotFoundError, OSError, IOError) as e:
            raise FileOperationError(
                f"Failed to load image from {file_path}", file_path=str(file_path), operation="load"
            ) from e

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
        if not file_path.exists():
            raise FileOperationError(
                f"File does not exist: {file_path}", file_path=str(file_path), operation="validate"
            )

        if not file_path.is_file():
            raise FileOperationError(
                f"Path is not a file: {file_path}", file_path=str(file_path), operation="validate"
            )

        try:
            with Image.open(file_path) as img:
                img.verify()
            return True
        except (OSError, IOError) as e:
            raise FileOperationError(
                f"File is not a valid image: {file_path}",
                file_path=str(file_path),
                operation="validate",
            ) from e

    def generate_image_path(self, base_name: str, output_dir: Path) -> Path:
        """
        Generate a unique image file path with timestamp.

        Args:
            base_name: Base name for the file
            output_dir: Directory to save the image

        Returns:
            Generated unique file path
        """
        # Use existing utility function to generate timestamped filename
        filename = generate_timestamped_filename(base_name)
        return output_dir / filename
