"""
Shared utility functions for the NanoAPIClient project.
Provides common functionality for date formatting, error handling, and file operations.
"""

__author__ = "Lene Preuss <lene.preuss@gmail.com>"

from datetime import datetime
from typing import Optional, Tuple, Any
from flask import jsonify, Response


# Date/time format constants
STANDARD_DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"
FILENAME_DATETIME_FORMAT = "%Y-%m-%d-%H:%M:%S"
COMPACT_DATETIME_FORMAT = "%y%m%d-%H:%M:%S"


def format_timestamp(dt: Optional[datetime], format_type: str = "standard") -> str:
    """
    Format timestamp with fallback to 'Unknown'.

    Args:
        dt: DateTime object to format, can be None
        format_type: Type of format ('standard', 'filename', 'compact')

    Returns:
        Formatted timestamp string or 'Unknown' if dt is None
    """
    if not dt:
        return "Unknown"

    formats = {
        "standard": STANDARD_DATETIME_FORMAT,
        "filename": FILENAME_DATETIME_FORMAT,
        "compact": COMPACT_DATETIME_FORMAT
    }
    return dt.strftime(formats.get(format_type, STANDARD_DATETIME_FORMAT))


def get_current_timestamp(format_type: str = "filename") -> str:
    """
    Get current timestamp in specified format.

    Args:
        format_type: Type of format ('standard', 'filename', 'compact')

    Returns:
        Current timestamp as formatted string
    """
    # Import datetime here to allow for easier mocking in tests
    from datetime import datetime as dt
    return format_timestamp(dt.now(), format_type)


def create_error_response(message: str, status_code: int = 400) -> Tuple[Response, int]:
    """
    Create standardized error response for Flask endpoints.

    Args:
        message: Error message to include in response
        status_code: HTTP status code (default: 400)

    Returns:
        Tuple of (JSON response, status code)
    """
    return jsonify({"error": message}), status_code


def safe_format_timestamps(create_time: Optional[datetime],
                           expiration_time: Optional[datetime]) -> Tuple[str, str]:
    """
    Safely format both create_time and expiration_time timestamps.

    Args:
        create_time: Creation timestamp, can be None
        expiration_time: Expiration timestamp, can be None

    Returns:
        Tuple of (formatted_create_time, formatted_expiration_time)
    """
    create_time_str = format_timestamp(create_time, "standard")
    expiration_time_str = format_timestamp(expiration_time, "standard")
    return create_time_str, expiration_time_str


def log_upload_info(image_path: Any, uploaded_file: Any) -> None:
    """
    Log upload information with formatted timestamps.

    Args:
        image_path: Path to the uploaded image
        uploaded_file: Uploaded file object with metadata
    """
    import logging

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
        uploaded_file.uri
    )
