"""
Local filesystem-based metadata repository implementation.
Provides local storage for generation metadata using JSON files.
"""

__author__ = "Lene Preuss <lene.preuss@gmail.com>"

import logging
from pathlib import Path
from typing import Optional, List

from stable_delusion.config import Config
from stable_delusion.exceptions import FileOperationError
from stable_delusion.models.metadata import GenerationMetadata
from stable_delusion.repositories.interfaces import MetadataRepository
from stable_delusion.utils import ensure_directory_exists


class LocalMetadataRepository(MetadataRepository):
    """Local filesystem-based implementation of MetadataRepository interface."""

    def __init__(self, config: Config):
        """
        Initialize local metadata repository.

        Args:
            config: Application configuration
        """
        self.config = config
        self.metadata_dir = config.default_output_dir / "metadata"
        # Ensure metadata directory exists
        ensure_directory_exists(self.metadata_dir)

    def save_metadata(self, metadata: GenerationMetadata) -> str:
        """
        Save generation metadata to local filesystem.

        Args:
            metadata: GenerationMetadata object to save

        Returns:
            Local file path where metadata was saved

        Raises:
            FileOperationError: If save operation fails
        """
        try:
            # Generate filename using metadata
            filename = metadata.get_metadata_filename()
            file_path = self.metadata_dir / filename

            # Write JSON to file
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(metadata.to_json())

            logging.info("Metadata saved locally: %s", file_path)
            return str(file_path)

        except (OSError, IOError, PermissionError) as e:
            raise FileOperationError(
                f"Failed to save metadata locally: {str(e)}",
                operation="save_metadata",
                file_path=str(file_path) if "file_path" in locals() else "unknown",
            ) from e

    def load_metadata(self, metadata_key: str) -> GenerationMetadata:
        """
        Load metadata from local filesystem.

        Args:
            metadata_key: Local file path for metadata

        Returns:
            GenerationMetadata object

        Raises:
            FileOperationError: If load operation fails
        """
        try:
            file_path = Path(metadata_key)

            if not file_path.exists():
                raise FileOperationError(
                    f"Metadata file not found: {metadata_key}",
                    operation="load_metadata",
                    file_path=metadata_key,
                )

            # Read JSON content
            with open(file_path, "r", encoding="utf-8") as f:
                json_content = f.read()

            # Parse metadata
            metadata = GenerationMetadata.from_json(json_content)

            logging.info("Metadata loaded locally: %s", metadata_key)
            return metadata

        except (OSError, IOError, PermissionError) as e:
            raise FileOperationError(
                f"Failed to load metadata locally: {str(e)}",
                operation="load_metadata",
                file_path=metadata_key,
            ) from e

    def metadata_exists(self, content_hash: str) -> Optional[str]:
        """
        Check if metadata exists for given content hash.

        Args:
            content_hash: SHA256 hash of generation inputs

        Returns:
            File path if metadata exists, None otherwise
        """
        try:
            # Search for files with matching hash prefix
            hash_prefix = content_hash[:8]
            pattern = f"metadata_{hash_prefix}_*.json"

            matching_files = list(self.metadata_dir.glob(pattern))

            for file_path in matching_files:
                try:
                    metadata = self.load_metadata(str(file_path))
                    if metadata.content_hash == content_hash:
                        return str(file_path)
                except FileOperationError:
                    # Skip corrupted or inaccessible metadata files
                    continue

            return None

        except (OSError, IOError, PermissionError) as e:
            logging.warning("Error checking metadata existence: %s", e)
            return None

    def list_metadata_by_hash_prefix(self, hash_prefix: str) -> List[str]:
        """
        List metadata files by content hash prefix.

        Args:
            hash_prefix: Hash prefix to search for

        Returns:
            List of file paths matching prefix
        """
        try:
            pattern = f"metadata_{hash_prefix}*.json"
            matching_files = list(self.metadata_dir.glob(pattern))
            return [str(f) for f in matching_files]

        except (OSError, IOError, PermissionError) as e:
            logging.warning("Error listing metadata by hash prefix: %s", e)
            return []
