"""Storage and file system configuration."""

__author__ = "Lene Preuss <lene.preuss@gmail.com>"

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class StorageConfig:
    """Storage and file system configuration."""

    storage_type: Optional[str] = None
    upload_folder: Optional[Path] = None
    default_output_dir: Optional[Path] = None
    output_dir: Optional[Path] = None  # Runtime override
