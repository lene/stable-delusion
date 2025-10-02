"""
Shared utility functions for the NanoAPIClient project.
Provides common functionality for date formatting, error handling, and file operations.
"""

__author__ = "Lene Preuss <lene.preuss@gmail.com>"

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, Any
from flask import jsonify, Response
from werkzeug.utils import secure_filename

import coloredlogs  # type: ignore[import-untyped]

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


# Logging utilities for consistent service and operation logging
def log_service_creation(service_name: str, model: str = "", **kwargs) -> None:

    if model:
        logging.info("🏗️ Creating %s for model: %s", service_name, model)
    else:
        logging.info("🏗️ Creating %s", service_name)

    for key, value in kwargs.items():
        if value is not None:
            logging.info("   %s: %s", key, value)


def log_operation_start(operation: str, **details) -> None:

    logging.info("🚀 Starting %s", operation)
    for key, value in details.items():
        if value is not None:
            logging.info("   %s: %s", key, value)


def log_operation_success(operation: str, result_count: Optional[int] = None, **details) -> None:

    if result_count is not None:
        logging.info("✅ %s completed: %d items", operation, result_count)
    else:
        logging.info("✅ %s completed", operation)

    for key, value in details.items():
        if value is not None:
            logging.info("   %s: %s", key, value)


def log_operation_failure(operation: str, error: Exception) -> None:

    logging.error("❌ %s failed: %s", operation, str(error))


# Error handling utilities
def handle_file_operation_error(operation: str, file_path: str, error: Exception) -> None:

    logging.error("File operation '%s' failed for %s: %s", operation, file_path, str(error))
    raise FileOperationError(
        f"Failed to {operation}: {str(error)}",
        file_path=file_path,
        operation=operation,
    ) from error


def safe_file_operation(operation_name: str, file_path: str, operation_func):
    try:
        return operation_func()
    except (OSError, IOError) as e:
        handle_file_operation_error(operation_name, file_path, e)
        return None  # This line will never be reached due to exception, but satisfies pylint


# Path and URL utilities
def normalize_path_for_key(path: str) -> str:
    return str(path).strip("/")


def is_s3_url(url: str) -> bool:
    return url.startswith("s3://")


def is_https_s3_url(url: str) -> bool:
    return url.startswith("https://") and (".s3." in url or ".s3-" in url)


def is_any_s3_url(url: str) -> bool:
    return is_s3_url(url) or is_https_s3_url(url)


def setup_logging(quiet: bool = False, debug: bool = False) -> None:
    if debug:
        log_level = logging.DEBUG
    elif quiet:
        log_level = logging.WARNING
    else:
        log_level = logging.INFO

    coloredlogs.install(
        level=log_level, fmt="%(asctime)s %(levelname)s: %(message)s", datefmt="%H:%M:%S"
    )
    if quiet and debug:
        logging.warning("Both --quiet and --debug specified. Using --debug mode.")
