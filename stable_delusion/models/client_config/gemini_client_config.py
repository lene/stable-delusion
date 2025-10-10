"""Complete configuration for GeminiClient."""

__author__ = "Lene Preuss <lene.preuss@gmail.com>"

from dataclasses import dataclass
from typing import Optional

from stable_delusion.models.client_config.gcp_config import GCPConfig
from stable_delusion.models.client_config.aws_config import AWSConfig
from stable_delusion.models.client_config.storage_config import StorageConfig
from stable_delusion.models.client_config.app_config import AppConfig
from stable_delusion.models.client_config.seedream_config import SeedreamConfig


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
