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

from stable_delusion.config import Config
from stable_delusion.exceptions import FileOperationError, ValidationError
from stable_delusion.repositories.interfaces import ImageRepository
from stable_delusion.repositories.s3_client import S3ClientManager, generate_s3_key, build_s3_url

if TYPE_CHECKING:
    from mypy_boto3_s3 import S3Client


class S3ImageRepository(ImageRepository):
    """S3-based implementation of ImageRepository interface."""

    def __init__(self, config: Config):
        self.config = config
        self.s3_client: "S3Client" = S3ClientManager.create_s3_client(config)
        # S3ClientManager validation ensures bucket_name is not None
        self.bucket_name: str = config.s3_bucket  # type: ignore[assignment]
        self.key_prefix = "images/"

    def _convert_image_to_bytes(self, image: Image.Image, file_path: Path) -> bytes:
        image_buffer = io.BytesIO()
        file_format = self._get_image_format(file_path)
        image.save(image_buffer, format=file_format)
        return image_buffer.getvalue()

    def _upload_to_s3(
        self, s3_key: str, image_bytes: bytes, file_format: str, file_path: Path
    ) -> None:
        self.s3_client.put_object(
            Bucket=self.bucket_name,
            Key=s3_key,
            Body=image_bytes,
            ContentType=f"image/{file_format.lower()}",
            Metadata={"original_filename": file_path.name, "uploaded_by": "stable-delusion"},
        )

    def _build_result_path(self, s3_key: str) -> Path:
        https_url = f"https://{self.bucket_name}.s3.{self.config.s3_region}.amazonaws.com/{s3_key}"
        logging.debug("Image saved to S3: %s", https_url)
        result_path = Path(https_url)
        # Fix URL normalization issue with Path objects
        result_str = str(result_path)
        if result_str.startswith("https:/") and not result_str.startswith("https://"):
            result_str = result_str.replace("https:/", "https://", 1)
            result_path = Path(result_str)
        return result_path

    def save_image(self, image: Image.Image, file_path: Path) -> Path:
        try:
            s3_key = generate_s3_key(str(file_path), self.key_prefix)
            image_bytes = self._convert_image_to_bytes(image, file_path)
            file_format = self._get_image_format(file_path)
            self._upload_to_s3(s3_key, image_bytes, file_format, file_path)
            return self._build_result_path(s3_key)
        except Exception as e:
            raise FileOperationError(
                f"Failed to save image to S3: {str(e)}",
                file_path=str(file_path),
                operation="save_image_s3",
            ) from e

    def load_image(self, file_path: Path) -> Image.Image:
        try:
            s3_key = self._extract_s3_key(file_path)
            image_data = self._download_image_from_s3(s3_key)
            image = self._convert_bytes_to_image(image_data)

            logging.debug("Image loaded from S3: %s", s3_key)
            return image

        except ValidationError:
            # Re-raise ValidationError without wrapping
            raise
        except Exception as e:  # pylint: disable=broad-exception-caught
            self._handle_load_image_error(e, file_path)
            # The above method always raises, so this is unreachable
            raise  # pragma: no cover

    def _download_image_from_s3(self, s3_key: str) -> bytes:
        """Download image data from S3."""
        response = self.s3_client.get_object(Bucket=self.bucket_name, Key=s3_key)
        return response["Body"].read()

    def _convert_bytes_to_image(self, image_data: bytes) -> Image.Image:
        """Convert bytes data to PIL Image."""
        image_buffer = io.BytesIO(image_data)
        return Image.open(image_buffer)

    def _handle_load_image_error(self, error: Exception, file_path: Path) -> None:
        """Handle errors during image loading."""
        error_code = getattr(error, "response", {}).get("Error", {}).get("Code", None)
        if error_code == "NoSuchKey" or (
            hasattr(error, "__class__") and "NoSuchKey" in str(error.__class__)
        ):
            raise FileOperationError(
                f"Image not found in S3: {file_path}",
                file_path=str(file_path),
                operation="load_image_s3",
            ) from error
        raise FileOperationError(
            f"Failed to load image from S3: {str(error)}",
            file_path=str(file_path),
            operation="load_image_s3",
        ) from error

    def validate_image_file(self, file_path: Path) -> bool:
        try:
            s3_key = self._extract_s3_key(file_path)

            # Check if object exists using head_object
            self.s3_client.head_object(Bucket=self.bucket_name, Key=s3_key)
            logging.debug("S3 image file validated: %s", s3_key)
            return True

        except ValidationError:
            # Re-raise validation errors (e.g., bucket mismatch)
            raise
        except Exception as e:  # pylint: disable=broad-exception-caught
            # Handle boto3 ClientError and other S3 exceptions
            error_code = getattr(e, "response", {}).get("Error", {}).get("Code", None)
            if error_code == "NoSuchKey" or (
                hasattr(e, "__class__") and "NoSuchKey" in str(e.__class__)
            ):
                return False
            logging.warning("Failed to validate S3 image file %s: %s", file_path, e)
            return False

    def generate_image_path(self, base_name: str, output_dir: Path) -> Path:
        # For S3, output_dir becomes part of the key prefix
        key_prefix = (
            f"{self.key_prefix}{output_dir}/" if output_dir != Path(".") else self.key_prefix
        )
        s3_key = generate_s3_key(base_name, key_prefix.rstrip("/"))

        # Return as S3 URL for consistency
        s3_url = build_s3_url(self.bucket_name, s3_key)
        return Path(s3_url)

    def _get_image_format(self, file_path: Path) -> str:
        extension = file_path.suffix.lower()
        format_mapping = {
            ".png": "PNG",
            ".jpg": "JPEG",
            ".jpeg": "JPEG",
            ".gif": "GIF",
            ".bmp": "BMP",
            ".webp": "WEBP",
        }
        return format_mapping.get(extension, "PNG")  # Default to PNG

    def _parse_s3_url_and_validate_bucket(self, path_str: str) -> str:
        from stable_delusion.repositories.s3_client import parse_s3_url

        try:
            bucket, key = parse_s3_url(path_str)
            if bucket != self.bucket_name:
                raise ValidationError(
                    f"S3 bucket mismatch: expected {self.bucket_name}, got {bucket}",
                    field="file_path",
                    value=path_str,
                )
            return key
        except ValueError as e:
            raise ValidationError(
                f"Invalid S3 URL format: {path_str}", field="file_path", value=path_str
            ) from e

    def _parse_https_s3_url_and_validate_bucket(self, path_str: str) -> str:
        from stable_delusion.repositories.s3_client import parse_https_s3_url

        try:
            bucket, key = parse_https_s3_url(path_str)
            if bucket != self.bucket_name:
                raise ValidationError(
                    f"S3 bucket mismatch: expected {self.bucket_name}, got {bucket}",
                    field="file_path",
                    value=path_str,
                )
            return key
        except ValueError as e:
            raise ValidationError(
                f"Invalid HTTPS S3 URL format: {path_str}", field="file_path", value=path_str
            ) from e

    def _extract_s3_key(self, file_path: Path) -> str:
        path_str = str(file_path)

        # Handle S3 URLs (s3:// format)
        if path_str.startswith("s3://"):
            return self._parse_s3_url_and_validate_bucket(path_str)

        # Handle HTTPS S3 URLs (including Path-normalized ones like https:/)
        if path_str.startswith(("https://", "http://", "https:/", "http:/")):
            return self._parse_https_s3_url_and_validate_bucket(path_str)

        # Handle direct keys (remove leading slash if present)
        return path_str.lstrip("/")
