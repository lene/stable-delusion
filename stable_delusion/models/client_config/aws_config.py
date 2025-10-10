"""Amazon Web Services configuration."""

__author__ = "Lene Preuss <lene.preuss@gmail.com>"

from dataclasses import dataclass
from typing import Optional


@dataclass
class AWSConfig:
    """Amazon Web Services configuration."""

    s3_bucket: Optional[str] = None
    s3_region: Optional[str] = None
    aws_access_key_id: Optional[str] = None
    aws_secret_access_key: Optional[str] = None
