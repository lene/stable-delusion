"""
Shared utility functions for the NanoAPIClient project.
Provides common functionality for date formatting, error handling, and file operations.
"""

__author__ = "Lene Preuss <lene.preuss@gmail.com>"

from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, Any, Dict, Union
from flask import jsonify, Response
from werkzeug.utils import secure_filename

from nano_api.exceptions import ValidationError, FileOperationError
from nano_api.conf import VALID_SCALE_FACTORS, DEFAULT_PROJECT_ID, DEFAULT_LOCATION


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
        "compact": COMPACT_DATETIME_FORMAT
    }
    return dt.strftime(formats.get(format_type, STANDARD_DATETIME_FORMAT))


def get_current_timestamp(format_type: str = "filename") -> str:
    return format_timestamp(datetime.now(), format_type)


def create_error_response(message: str, status_code: int = 400) -> Tuple[Response, int]:
    return jsonify({"error": message}), status_code


def safe_format_timestamps(create_time: Optional[datetime],
                           expiration_time: Optional[datetime]) -> Tuple[str, str]:
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
        uploaded_file.uri
    )


def validate_scale_parameter(scale_value: Union[str, int, None]) -> Optional[int]:
    if scale_value is None:
        return None

    if isinstance(scale_value, int):
        scale = scale_value
    else:
        try:
            scale = int(scale_value)
        except (ValueError, TypeError) as e:
            raise ValidationError(
                "Scale must be an integer",
                field="scale",
                value=str(scale_value)
            ) from e

    if scale not in VALID_SCALE_FACTORS:
        raise ValidationError(
            f"Scale must be one of {VALID_SCALE_FACTORS}",
            field="scale",
            value=str(scale)
        )

    return scale


def get_project_config(source_dict: Dict[str, Any],
                       key_project: str = "project_id",
                       key_location: str = "location") -> Tuple[str, str]:
    if hasattr(source_dict, 'get'):  # Dict-like object (request.form)
        project_id = source_dict.get(key_project) or DEFAULT_PROJECT_ID
        location = source_dict.get(key_location) or DEFAULT_LOCATION
    else:  # Object with attributes (argparse Namespace)
        project_id = getattr(source_dict, key_project, None) or DEFAULT_PROJECT_ID
        location = getattr(source_dict, key_location, None) or DEFAULT_LOCATION

    return project_id, location


def generate_timestamped_filename(base_name: str,
                                  extension: str = "png",
                                  format_type: str = "filename",
                                  secure: bool = False) -> str:
    timestamp = get_current_timestamp(format_type)
    filename = f"{base_name}_{timestamp}.{extension}"

    if secure:
        filename = secure_filename(filename)

    return filename


def validate_image_file(path: Path) -> None:
    if not path.is_file():
        raise FileOperationError(
            f"Image file not found: {path}",
            file_path=str(path),
            operation="read"
        )


def ensure_directory_exists(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
