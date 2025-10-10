"""Application-level configuration."""

__author__ = "Lene Preuss <lene.preuss@gmail.com>"

from dataclasses import dataclass
from typing import Optional


@dataclass
class AppConfig:
    """Application-level configuration."""

    flask_debug: Optional[bool] = None
