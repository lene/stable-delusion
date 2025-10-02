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
            file_path = self._prepare_metadata_file_path(metadata)
            self._write_metadata_to_file(file_path, metadata)

            logging.debug("Metadata saved locally: %s", file_path)
            return str(file_path)

        except (OSError, IOError, PermissionError) as e:
            raise FileOperationError(
                f"Failed to save metadata locally: {str(e)}",
                operation="save_metadata",
                file_path=str(file_path) if "file_path" in locals() else "unknown",
            ) from e

    def _prepare_metadata_file_path(self, metadata: GenerationMetadata) -> Path:
        """Prepare file path for metadata storage."""
        filename = metadata.get_metadata_filename()
        return self.metadata_dir / filename

    def _write_metadata_to_file(self, file_path: Path, metadata: GenerationMetadata) -> None:
        """Write metadata JSON to file."""
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(metadata.to_json())

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
            self._validate_metadata_file_exists(file_path, metadata_key)
            json_content = self._read_metadata_file(file_path)
            metadata = self._parse_metadata_content(json_content)

            logging.debug("Metadata loaded locally: %s", metadata_key)
            return metadata

        except (OSError, IOError, PermissionError) as e:
            raise FileOperationError(
                f"Failed to load metadata locally: {str(e)}",
                operation="load_metadata",
                file_path=metadata_key,
            ) from e

    def _validate_metadata_file_exists(self, file_path: Path, metadata_key: str) -> None:
        """Validate that metadata file exists."""
        if not file_path.exists():
            raise FileOperationError(
                f"Metadata file not found: {metadata_key}",
                operation="load_metadata",
                file_path=metadata_key,
            )

    def _read_metadata_file(self, file_path: Path) -> str:
        """Read JSON content from metadata file."""
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()

    def _parse_metadata_content(self, json_content: str) -> GenerationMetadata:
        """Parse JSON content into GenerationMetadata object."""
        return GenerationMetadata.from_json(json_content)

    def metadata_exists(self, content_hash: str) -> Optional[str]:
        """
        Check if metadata exists for given content hash.

        Args:
            content_hash: SHA256 hash of generation inputs

        Returns:
            File path if metadata exists, None otherwise
        """
        try:
            matching_files = self._find_potential_metadata_files(content_hash)
            return self._verify_content_hash_match(matching_files, content_hash)
        except (OSError, IOError, PermissionError) as e:
            logging.warning("Error checking metadata existence: %s", e)
            return None

    def _find_potential_metadata_files(self, content_hash: str) -> List[Path]:
        """Find metadata files with matching hash prefix."""
        hash_prefix = content_hash[:8]
        pattern = f"metadata_{hash_prefix}_*.json"
        return list(self.metadata_dir.glob(pattern))

    def _verify_content_hash_match(
        self, file_paths: List[Path], content_hash: str
    ) -> Optional[str]:
        """Verify which file has matching content hash."""
        for file_path in file_paths:
            if self._check_file_hash_match(file_path, content_hash):
                return str(file_path)
        return None

    def _check_file_hash_match(self, file_path: Path, content_hash: str) -> bool:
        """Check if a single file has matching content hash."""
        try:
            metadata = self.load_metadata(str(file_path))
            return metadata.content_hash == content_hash
        except FileOperationError:
            # Skip corrupted or inaccessible metadata files
            return False

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
