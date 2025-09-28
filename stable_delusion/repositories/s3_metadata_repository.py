"""
S3-based metadata repository implementation.
Provides cloud storage for generation metadata using Amazon S3.
"""

__author__ = "Lene Preuss <lene.preuss@gmail.com>"

import logging
from typing import Optional, List, TYPE_CHECKING

from botocore.exceptions import ClientError

from stable_delusion.config import Config
from stable_delusion.exceptions import FileOperationError
from stable_delusion.models.metadata import GenerationMetadata
from stable_delusion.repositories.interfaces import MetadataRepository
from stable_delusion.repositories.s3_client import S3ClientManager, generate_s3_key

if TYPE_CHECKING:
    from mypy_boto3_s3 import S3Client


class S3MetadataRepository(MetadataRepository):
    """S3-based implementation of MetadataRepository interface."""

    def __init__(self, config: Config):
        self.config = config
        self.s3_client: "S3Client" = S3ClientManager.create_s3_client(config)
        # S3ClientManager validation ensures bucket_name is not None
        self.bucket_name: str = config.s3_bucket  # type: ignore[assignment]
        self.key_prefix = "metadata/"

    def save_metadata(self, metadata: GenerationMetadata) -> str:
        """
        Save generation metadata to S3 storage.

        Args:
            metadata: GenerationMetadata object to save

        Returns:
            S3 key where metadata was saved

        Raises:
            FileOperationError: If save operation fails
        """
        try:
            # Generate S3 key using metadata filename
            filename = metadata.get_metadata_filename()
            s3_key = generate_s3_key(filename, self.key_prefix)

            # Convert metadata to JSON
            json_content = metadata.to_json()

            # Upload to S3 with public read permissions
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=s3_key,
                Body=json_content.encode("utf-8"),
                ContentType="application/json",
                ACL="public-read",  # Make publicly accessible
                Metadata={
                    "content-hash": metadata.content_hash or "",
                    "generation-timestamp": metadata.timestamp or "",
                    "prompt-preview": (
                        metadata.prompt[:100] + "..."
                        if len(metadata.prompt) > 100
                        else metadata.prompt
                    ),
                },
            )

            logging.info("Metadata saved to S3: %s", s3_key)
            return s3_key

        except ClientError as e:
            error_code = e.response["Error"]["Code"]
            raise FileOperationError(
                f"Failed to save metadata to S3: {error_code}",
                operation="save_metadata",
                file_path=s3_key if "s3_key" in locals() else "unknown",
            ) from e

        except Exception as e:
            raise FileOperationError(
                f"Unexpected error saving metadata: {str(e)}",
                operation="save_metadata",
                file_path="unknown",
            ) from e

    def load_metadata(self, metadata_key: str) -> GenerationMetadata:
        """
        Load metadata from S3 storage by key.

        Args:
            metadata_key: S3 key for metadata

        Returns:
            GenerationMetadata object

        Raises:
            FileOperationError: If load operation fails
        """
        try:
            response = self.s3_client.get_object(Bucket=self.bucket_name, Key=metadata_key)

            # Read JSON content
            json_content = response["Body"].read().decode("utf-8")

            # Parse metadata
            metadata = GenerationMetadata.from_json(json_content)

            logging.info("Metadata loaded from S3: %s", metadata_key)
            return metadata

        except ClientError as e:
            error_code = e.response["Error"]["Code"]
            if error_code == "NoSuchKey":
                raise FileOperationError(
                    f"Metadata not found: {metadata_key}",
                    operation="load_metadata",
                    file_path=metadata_key,
                ) from e
            raise FileOperationError(
                f"Failed to load metadata from S3: {error_code}",
                operation="load_metadata",
                file_path=metadata_key,
            ) from e

        except Exception as e:
            raise FileOperationError(
                f"Unexpected error loading metadata: {str(e)}",
                operation="load_metadata",
                file_path=metadata_key,
            ) from e

    def metadata_exists(self, content_hash: str) -> Optional[str]:
        """
        Check if metadata exists for given content hash.

        Args:
            content_hash: SHA256 hash of generation inputs

        Returns:
            S3 key if metadata exists, None otherwise
        """
        try:
            # List objects with metadata prefix and filter by hash
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=self.key_prefix,
                MaxKeys=1000,  # Should be sufficient for deduplication checks
            )

            if "Contents" not in response:
                return None

            # Look for metadata files with matching content hash
            for obj in response["Contents"]:
                key = obj["Key"]
                if f"metadata_{content_hash[:8]}" in key:
                    # Verify by loading the metadata and checking full hash
                    try:
                        metadata = self.load_metadata(key)
                        if metadata.content_hash == content_hash:
                            return key
                    except FileOperationError:
                        # Skip corrupted or inaccessible metadata files
                        continue

            return None

        except ClientError as e:
            logging.warning("Error checking metadata existence: %s", e)
            return None

    def list_metadata_by_hash_prefix(self, hash_prefix: str) -> List[str]:
        """
        List metadata keys by content hash prefix.

        Args:
            hash_prefix: Hash prefix to search for

        Returns:
            List of S3 keys matching prefix
        """
        try:
            matching_keys = []

            # List objects with metadata prefix
            paginator = self.s3_client.get_paginator("list_objects_v2")
            pages = paginator.paginate(Bucket=self.bucket_name, Prefix=self.key_prefix)

            for page in pages:
                if "Contents" not in page:
                    continue

                for obj in page["Contents"]:
                    key = obj["Key"]
                    if f"metadata_{hash_prefix}" in key:
                        matching_keys.append(key)

            return matching_keys

        except ClientError as e:
            logging.warning("Error listing metadata by hash prefix: %s", e)
            return []
