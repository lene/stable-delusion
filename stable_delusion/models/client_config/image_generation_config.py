"""Configuration for image generation parameters."""

__author__ = "Lene Preuss <lene.preuss@gmail.com>"

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List


@dataclass
class ImageGenerationConfig:
    """Configuration for image generation parameters."""

    generated_file: Optional[Path] = None
    prompt: str = ""
    scale: Optional[int] = None
    image_size: Optional[str] = None
    saved_files: Optional[List[Path]] = None
    output_dir: Optional[Path] = None
