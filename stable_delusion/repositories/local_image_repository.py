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
        try:
            return Image.open(file_path)
        except (FileNotFoundError, OSError, IOError) as e:
            raise FileOperationError(
                f"Failed to load image from {file_path}", file_path=str(file_path), operation="load"
            ) from e

    def validate_image_file(self, file_path: Path) -> bool:
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
        # Use existing utility function to generate timestamped filename
        filename = generate_timestamped_filename(base_name)
        return output_dir / filename
