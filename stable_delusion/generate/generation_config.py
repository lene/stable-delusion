"""Configuration for image generation parameters."""

__author__ = "Lene Preuss <lene.preuss@gmail.com>"

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from stable_delusion.config import DEFAULT_PROJECT_ID, DEFAULT_LOCATION


@dataclass
class GenerationConfig:
    """Configuration for image generation parameters."""

    project_id: str = DEFAULT_PROJECT_ID
    location: str = DEFAULT_LOCATION
    output_dir: Path = Path(".")
    storage_type: Optional[str] = None
