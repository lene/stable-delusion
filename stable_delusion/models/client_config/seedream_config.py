"""SeeEdit Seedream API configuration."""

__author__ = "Lene Preuss <lene.preuss@gmail.com>"

from dataclasses import dataclass
from typing import Optional


@dataclass
class SeedreamConfig:
    """SeeEdit Seedream API configuration."""

    api_key: Optional[str] = None
