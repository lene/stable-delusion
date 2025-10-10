"""Abstract repository interface for metadata storage operations."""

__author__ = "Lene Preuss <lene.preuss@gmail.com>"

from abc import ABC, abstractmethod
from typing import List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from stable_delusion.models.metadata import GenerationMetadata


class MetadataRepository(ABC):
    """Abstract repository interface for metadata storage and retrieval operations."""

    @abstractmethod
    def save_metadata(self, metadata: "GenerationMetadata") -> str:
        pass

    @abstractmethod
    def load_metadata(self, metadata_key: str) -> "GenerationMetadata":
        pass

    @abstractmethod
    def metadata_exists(self, content_hash: str) -> Optional[str]:
        pass

    @abstractmethod
    def list_metadata_by_hash_prefix(self, hash_prefix: str) -> List[str]:
        pass
