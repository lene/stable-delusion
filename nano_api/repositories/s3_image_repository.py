"""
S3-based image repository implementation.
Provides cloud storage for images using Amazon S3.
"""

__author__ = "Lene Preuss <lene.preuss@gmail.com>"

import io
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from PIL import Image

from nano_api.config import Config
from nano_api.exceptions import FileOperationError, ValidationError
from nano_api.repositories.interfaces import ImageRepository
from nano_api.repositories.s3_client import (S3ClientManager, generate_s3_key,
                                             build_s3_url)

if TYPE_CHECKING:
    from mypy_boto3_s3 import S3Client


class S3ImageRepository(ImageRepository):
    """S3-based implementation of ImageRepository interface."""

    def __init__(self, config: Config):
        """
        Initialize S3 image repository.

        Args:
            config: Application configuration containing S3 settings
        """
        self.config = config
        self.s3_client: 'S3Client' = S3ClientManager.create_s3_client(config)
        # S3ClientManager validation ensures bucket_name is not None
        self.bucket_name: str = config.s3_bucket  # type: ignore[assignment]
        self.key_prefix = "images/"

    def save_image(self, image: Image.Image, file_path: Path) -> Path:
        """
        Save an image to S3 storage.

        Args:
            image: PIL Image object to save
            file_path: S3 key path for the image

        Returns:
            S3 URL path where the image was saved

        Raises:
            FileOperationError: If image save fails
        """
        try:
            # Generate S3 key from file path
            s3_key = generate_s3_key(str(file_path), self.key_prefix)

            # Convert PIL image to bytes
            image_buffer = io.BytesIO()

            # Determine image format from file extension or use PNG as default
            file_format = self._get_image_format(file_path)
            image.save(image_buffer, format=file_format)
            image_bytes = image_buffer.getvalue()

            # Upload to S3
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=s3_key,
                Body=image_bytes,
                ContentType=f"image/{file_format.lower()}",
                Metadata={
                    'original_filename': file_path.name,
                    'uploaded_by': 'nano-api-client'
                }
            )

            # Return S3 URL as Path object for compatibility
            s3_url = build_s3_url(self.bucket_name, s3_key)
            logging.info("Image saved to S3: %s", s3_url)
            return Path(s3_url)

        except Exception as e:
            raise FileOperationError(
                f"Failed to save image to S3: {str(e)}",
                file_path=str(file_path),
                operation="save_image_s3"
            ) from e

    def load_image(self, file_path: Path) -> Image.Image:
        """
        Load an image from S3 storage.

        Args:
            file_path: S3 URL or key path for the image

        Returns:
            PIL Image object

        Raises:
            FileOperationError: If image load fails
            ValidationError: If file path is invalid
        """
        try:
            # Handle both S3 URLs and direct keys
            s3_key = self._extract_s3_key(file_path)

            # Download from S3
            response = self.s3_client.get_object(Bucket=self.bucket_name, Key=s3_key)
            image_data = response['Body'].read()

            # Convert bytes to PIL Image
            image_buffer = io.BytesIO(image_data)
            image = Image.open(image_buffer)

            logging.info("Image loaded from S3: %s", s3_key)
            return image

        except self.s3_client.exceptions.NoSuchKey as e:
            raise FileOperationError(
                f"Image not found in S3: {file_path}",
                file_path=str(file_path),
                operation="load_image_s3"
            ) from e
        except Exception as e:
            raise FileOperationError(
                f"Failed to load image from S3: {str(e)}",
                file_path=str(file_path),
                operation="load_image_s3"
            ) from e

    def validate_image_file(self, file_path: Path) -> bool:
        """
        Validate that an image file exists in S3.

        Args:
            file_path: S3 URL or key path to validate

        Returns:
            True if file exists and is valid, False otherwise
        """
        try:
            s3_key = self._extract_s3_key(file_path)

            # Check if object exists using head_object
            self.s3_client.head_object(Bucket=self.bucket_name, Key=s3_key)
            logging.debug("S3 image file validated: %s", s3_key)
            return True

        except self.s3_client.exceptions.NoSuchKey:
            return False
        except (self.s3_client.exceptions.ClientError, OSError, ValueError) as e:
            logging.warning("Failed to validate S3 image file %s: %s", file_path, e)
            return False

    def generate_image_path(self, base_name: str, output_dir: Path) -> Path:
        """
        Generate an S3 key path for a new image.

        Args:
            base_name: Base filename for the image
            output_dir: S3 key prefix (used as directory-like structure)

        Returns:
            S3 key path as Path object
        """
        # For S3, output_dir becomes part of the key prefix
        key_prefix = (f"{self.key_prefix}{output_dir}/" if output_dir != Path('.')
                      else self.key_prefix)
        s3_key = generate_s3_key(base_name, key_prefix.rstrip('/'))

        # Return as S3 URL for consistency
        s3_url = build_s3_url(self.bucket_name, s3_key)
        return Path(s3_url)

    def _get_image_format(self, file_path: Path) -> str:
        """
        Determine image format from file extension.

        Args:
            file_path: Path to extract format from

        Returns:
            Image format string (PNG, JPEG, etc.)
        """
        extension = file_path.suffix.lower()
        format_mapping = {
            '.png': 'PNG',
            '.jpg': 'JPEG',
            '.jpeg': 'JPEG',
            '.gif': 'GIF',
            '.bmp': 'BMP',
            '.webp': 'WEBP'
        }
        return format_mapping.get(extension, 'PNG')  # Default to PNG

    def _extract_s3_key(self, file_path: Path) -> str:
        """
        Extract S3 key from file path (handles both URLs and keys).

        Args:
            file_path: S3 URL or key path

        Returns:
            S3 object key

        Raises:
            ValidationError: If path format is invalid
        """
        path_str = str(file_path)

        # Handle S3 URLs
        if path_str.startswith('s3://'):
            from nano_api.repositories.s3_client import parse_s3_url
            try:
                bucket, key = parse_s3_url(path_str)
                if bucket != self.bucket_name:
                    raise ValidationError(
                        f"S3 bucket mismatch: expected {self.bucket_name}, got {bucket}",
                        field="file_path",
                        value=path_str
                    )
                return key
            except ValueError as e:
                raise ValidationError(
                    f"Invalid S3 URL format: {path_str}",
                    field="file_path",
                    value=path_str
                ) from e

        # Handle direct keys (remove leading slash if present)
        return path_str.lstrip('/')
