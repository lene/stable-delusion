"""Google Cloud Platform configuration."""

__author__ = "Lene Preuss <lene.preuss@gmail.com>"

from dataclasses import dataclass
from typing import Optional


@dataclass
class GCPConfig:
    """Google Cloud Platform configuration."""

    project_id: Optional[str] = None
    location: Optional[str] = None
    gemini_api_key: Optional[str] = None
