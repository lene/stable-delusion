"""
Repositories package for NanoAPIClient.
Contains repository interfaces and implementations for data persistence operations.
"""

__author__ = "Lene Preuss <lene.preuss@gmail.com>"

from nano_api.repositories.interfaces import ImageRepository, FileRepository, UploadRepository
from nano_api.repositories.local_file_repository import LocalFileRepository
from nano_api.repositories.local_image_repository import LocalImageRepository
from nano_api.repositories.upload_repository import LocalUploadRepository

__all__ = [
    "ImageRepository",
    "FileRepository",
    "UploadRepository",
    "LocalImageRepository",
    "LocalFileRepository",
    "LocalUploadRepository",
]
