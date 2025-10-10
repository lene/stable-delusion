"""
Repository interface definitions for NanoAPIClient.
Defines abstract base classes for data persistence operations.
"""

__author__ = "Lene Preuss <lene.preuss@gmail.com>"

from stable_delusion.repositories.interfaces.image_repository import ImageRepository
from stable_delusion.repositories.interfaces.file_repository import FileRepository
from stable_delusion.repositories.interfaces.metadata_repository import MetadataRepository

__all__ = [
    "ImageRepository",
    "FileRepository",
    "MetadataRepository",
]
