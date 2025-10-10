"""
Configuration dataclass for application settings.
"""

__author__ = "Lene Preuss <lene.preuss@gmail.com>"

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from stable_delusion.exceptions import ConfigurationError


@dataclass
class Config:
    """Configuration class containing all application settings."""

    project_id: str
    location: str
    gemini_api_key: str
    upload_folder: Path
    default_output_dir: Path
    flask_debug: bool

    # Storage configuration
    storage_type: str
    s3_bucket: Optional[str]
    s3_region: Optional[str]
    aws_access_key_id: Optional[str]
    aws_secret_access_key: Optional[str]

    def __post_init__(self) -> None:
        # GEMINI_API_KEY validation is now done only when needed in GeminiClient

        # Validate S3 configuration if S3 storage is enabled
        if self.storage_type == "s3":
            if not self.s3_bucket:
                raise ConfigurationError(
                    "AWS_S3_BUCKET environment variable is required when storage_type is 's3'",
                    config_key="AWS_S3_BUCKET",
                )
            if not self.s3_region:
                raise ConfigurationError(
                    "AWS_S3_REGION environment variable is required when storage_type is 's3'",
                    config_key="AWS_S3_REGION",
                )

        # Ensure local directories exist only for local storage
        if self.storage_type == "local":
            self.upload_folder.mkdir(parents=True, exist_ok=True)
            self.default_output_dir.mkdir(parents=True, exist_ok=True)
