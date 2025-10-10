"""
Configuration manager for loading and managing application configuration.
"""

__author__ = "Lene Preuss <lene.preuss@gmail.com>"

import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

from stable_delusion.config.config import Config
from stable_delusion.config.constants import DEFAULT_PROJECT_ID, DEFAULT_LOCATION


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
