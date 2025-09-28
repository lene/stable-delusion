"""
Configuration models for GeminiClient.
Provides type-safe configuration groupings for different service areas.
"""

__author__ = "Lene Preuss <lene.preuss@gmail.com>"

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List


@dataclass
class GCPConfig:
    """Google Cloud Platform configuration."""

    project_id: Optional[str] = None
    location: Optional[str] = None
    gemini_api_key: Optional[str] = None


@dataclass
class AWSConfig:
    """Amazon Web Services configuration."""

    s3_bucket: Optional[str] = None
    s3_region: Optional[str] = None
    aws_access_key_id: Optional[str] = None
    aws_secret_access_key: Optional[str] = None


@dataclass
class SeedreamConfig:
    """SeeEdit Seedream API configuration."""

    api_key: Optional[str] = None


@dataclass
class StorageConfig:
    """Storage and file system configuration."""

    storage_type: Optional[str] = None
    upload_folder: Optional[Path] = None
    default_output_dir: Optional[Path] = None
    output_dir: Optional[Path] = None  # Runtime override


@dataclass
class AppConfig:
    """Application-level configuration."""

    flask_debug: Optional[bool] = None


@dataclass
class ImageGenerationConfig:
    """Configuration for image generation parameters."""

    generated_file: Optional[Path] = None
    prompt: str = ""
    scale: Optional[int] = None
    image_size: Optional[str] = None
    saved_files: Optional[List[Path]] = None
    output_dir: Optional[Path] = None


@dataclass
class GeminiClientConfig:
    """Complete configuration for GeminiClient."""

    gcp: Optional[GCPConfig] = None
    aws: Optional[AWSConfig] = None
    storage: Optional[StorageConfig] = None
    app: Optional[AppConfig] = None
    seedream: Optional[SeedreamConfig] = None

    def __post_init__(self):
        if self.gcp is None:
            self.gcp = GCPConfig()
        if self.aws is None:
            self.aws = AWSConfig()
        if self.storage is None:
            self.storage = StorageConfig()
        if self.app is None:
            self.app = AppConfig()
        if self.seedream is None:
            self.seedream = SeedreamConfig()
