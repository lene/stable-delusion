"""
Shared utility functions for the NanoAPIClient project.
Provides common functionality for date formatting, error handling, and file operations.
"""

__author__ = "Lene Preuss <lene.preuss@gmail.com>"

from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, Any, Union, Callable
from flask import jsonify, Response
from werkzeug.utils import secure_filename

from stable_delusion.exceptions import FileOperationError


# Date/time format constants
STANDARD_DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"
FILENAME_DATETIME_FORMAT = "%Y-%m-%d-%H:%M:%S"
COMPACT_DATETIME_FORMAT = "%y%m%d-%H:%M:%S"


def format_timestamp(dt: Optional[datetime], format_type: str = "standard") -> str:
    if not dt:
        return "Unknown"

    formats = {
        "standard": STANDARD_DATETIME_FORMAT,
        "filename": FILENAME_DATETIME_FORMAT,
        "compact": COMPACT_DATETIME_FORMAT,
    }
    return dt.strftime(formats.get(format_type, STANDARD_DATETIME_FORMAT))


def get_current_timestamp(format_type: str = "filename") -> str:
    return format_timestamp(datetime.now(), format_type)


def create_error_response(message: str, status_code: int = 400) -> Tuple[Response, int]:
    return jsonify({"error": message}), status_code


def safe_format_timestamps(
    create_time: Optional[datetime], expiration_time: Optional[datetime]
) -> Tuple[str, str]:
    create_time_str = format_timestamp(create_time, "standard")
    expiration_time_str = format_timestamp(expiration_time, "standard")
    return create_time_str, expiration_time_str


def log_upload_info(image_path: Any, uploaded_file: Any) -> None:
    import logging  # pylint: disable=import-outside-toplevel

    create_time_str, expiration_time_str = safe_format_timestamps(
        uploaded_file.create_time, uploaded_file.expiration_time
    )

    logging.info(
        "Uploaded file: %s -> name=%s, mime_type=%s, size_bytes=%d, "
        "create_time=%s, expiration_time=%s, uri=%s",
        image_path,
        uploaded_file.name,
        uploaded_file.mime_type,
        uploaded_file.size_bytes,
        create_time_str,
        expiration_time_str,
        uploaded_file.uri,
    )


def generate_timestamped_filename(
    base_name: str, extension: str = "png", format_type: str = "filename", secure: bool = False
) -> str:
    timestamp = get_current_timestamp(format_type)
    filename = f"{base_name}_{timestamp}.{extension}"

    if secure:
        filename = secure_filename(filename)

    return filename


def validate_image_file(path: Path) -> None:
    if not path.is_file():
        raise FileOperationError(
            f"Image file not found: {path}", file_path=str(path), operation="read"
        )


def ensure_directory_exists(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def handle_file_operation_error(
    operation: str,
    resource: str,
    file_path: Union[str, Path],
    storage_type: str = "local"
) -> Callable:
    """
    Decorator factory for consistent file operation error handling.

    Args:
        operation: Operation being performed (e.g., "save", "load", "delete")
        resource: Resource being operated on (e.g., "image", "file", "metadata")
        file_path: Path to the file being operated on
        storage_type: Storage type (e.g., "local", "s3")

    Returns:
        Decorator function that wraps operations with consistent error handling
    """
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                raise FileOperationError(
                    f"Failed to {operation} {resource}: {str(e)}",
                    file_path=str(file_path),
                    operation=f"{operation}_{storage_type}",
                ) from e
        return wrapper
    return decorator


def is_s3_not_found_error(exception: Exception) -> bool:
    """
    Check if an exception indicates S3 object not found.

    Args:
        exception: Exception to check

    Returns:
        True if the exception indicates object not found
    """
    # Check for direct NoSuchKey exception
    if hasattr(exception, '__class__') and 'NoSuchKey' in str(exception.__class__):
        return True

    # Check for ClientError with NoSuchKey code
    error_code = getattr(exception, "response", {}).get("Error", {}).get("Code", None)
    if error_code == "NoSuchKey":
        return True

    return False


def ensure_path_object(value: Union[str, Path]) -> Path:
    """
    Ensure a value is converted to a Path object.

    Args:
        value: String or Path object

    Returns:
        Path object
    """
    if isinstance(value, str):
        return Path(value)
    return value


def normalize_https_url(url: str) -> str:
    """
    Normalize HTTPS URLs to ensure proper protocol format.

    Args:
        url: URL to normalize

    Returns:
        Normalized URL with proper https:// prefix
    """
    if url.startswith("https:/") and not url.startswith("https://"):
        return url.replace("https:/", "https://", 1)
    return url
