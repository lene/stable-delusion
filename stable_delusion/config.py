"""
Centralized configuration management for NanoAPIClient.
Provides environment-based configuration with validation and defaults.
"""

__author__ = "Lene Preuss <lene.preuss@gmail.com>"

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

from stable_delusion.exceptions import ConfigurationError

# Configuration constants
DEFAULT_PROJECT_ID = "gen-lang-client-0216779332"
DEFAULT_LOCATION = "us-central1"

# Model configuration
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash-image-preview"
DEFAULT_SEEDREAM_MODEL = "seedream-4-0-250828"

# Supported models for image generation
SUPPORTED_MODELS = ["gemini", "seedream"]


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


class ConfigManager:
    """Manages application configuration from environment variables."""

    _instance: Optional[Config] = None

    @classmethod
    def get_config(cls) -> Config:
        if cls._instance is None:
            cls._instance = cls._create_config()
        return cls._instance

    @classmethod
    def reset_config(cls) -> None:
        cls._instance = None

    @classmethod
    def _create_config(cls) -> Config:
        # Load .env file if it exists (environment variables take precedence)
        load_dotenv(override=False)

        return Config(
            project_id=os.getenv("GCP_PROJECT_ID") or DEFAULT_PROJECT_ID,
            location=os.getenv("GCP_LOCATION") or DEFAULT_LOCATION,
            gemini_api_key=os.getenv("GEMINI_API_KEY", ""),
            upload_folder=Path(os.getenv("UPLOAD_FOLDER", "uploads")),
            default_output_dir=Path(os.getenv("DEFAULT_OUTPUT_DIR", ".")),
            flask_debug=os.getenv("FLASK_DEBUG", "False").lower() in ("true", "1", "yes"),
            # Storage configuration
            storage_type=os.getenv("STORAGE_TYPE", "local").lower(),
            s3_bucket=os.getenv("AWS_S3_BUCKET"),
            s3_region=os.getenv("AWS_S3_REGION"),
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        )
