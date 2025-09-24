"""
Centralized configuration management for NanoAPIClient.
Provides environment-based configuration with validation and defaults.
"""

__author__ = "Lene Preuss <lene.preuss@gmail.com>"

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from nano_api.conf import DEFAULT_PROJECT_ID, DEFAULT_LOCATION


@dataclass
class Config:
    """Configuration class containing all application settings."""

    project_id: str
    location: str
    gemini_api_key: str
    upload_folder: Path
    default_output_dir: Path
    flask_debug: bool

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        if not self.gemini_api_key:
            raise ValueError(
                "GEMINI_API_KEY environment variable is required but not set"
            )

        # Ensure directories exist
        self.upload_folder.mkdir(parents=True, exist_ok=True)
        self.default_output_dir.mkdir(parents=True, exist_ok=True)


class ConfigManager:
    """Manages application configuration from environment variables."""

    _instance: Optional[Config] = None

    @classmethod
    def get_config(cls) -> Config:
        """
        Get the application configuration.
        Uses singleton pattern to ensure consistent configuration across the app.
        """
        if cls._instance is None:
            cls._instance = cls._create_config()
        return cls._instance

    @classmethod
    def reset_config(cls) -> None:
        """Reset the configuration instance (useful for testing)."""
        cls._instance = None

    @classmethod
    def _create_config(cls) -> Config:
        """Create configuration from environment variables."""
        return Config(
            project_id=os.getenv("GCP_PROJECT_ID") or DEFAULT_PROJECT_ID,
            location=os.getenv("GCP_LOCATION") or DEFAULT_LOCATION,
            gemini_api_key=os.getenv("GEMINI_API_KEY", ""),
            upload_folder=Path(os.getenv("UPLOAD_FOLDER", "uploads")),
            default_output_dir=Path(os.getenv("DEFAULT_OUTPUT_DIR", ".")),
            flask_debug=os.getenv("FLASK_DEBUG", "False").lower() in (
                "true", "1", "yes"
            )
        )
