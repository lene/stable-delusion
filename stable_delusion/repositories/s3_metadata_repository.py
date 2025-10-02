"""
S3-based metadata repository implementation.
Provides cloud storage for generation metadata using Amazon S3.
"""

__author__ = "Lene Preuss <lene.preuss@gmail.com>"

import logging
from typing import Optional, List, Tuple, Dict, TYPE_CHECKING

from botocore.exceptions import ClientError

from stable_delusion.config import Config
from stable_delusion.exceptions import FileOperationError
from stable_delusion.models.metadata import GenerationMetadata
from stable_delusion.repositories.interfaces import MetadataRepository
from stable_delusion.repositories.s3_client import S3ClientManager, generate_s3_key

if TYPE_CHECKING:
    from mypy_boto3_s3 import S3Client
    from mypy_boto3_s3.type_defs import ListObjectsV2OutputTypeDef


class S3MetadataRepository(MetadataRepository):
    """S3-based implementation of MetadataRepository interface."""

    def __init__(self, config: Config):
        self.config = config
        self.s3_client: "S3Client" = S3ClientManager.create_s3_client(config)
        # S3ClientManager validation ensures bucket_name is not None
        self.bucket_name: str = config.s3_bucket  # type: ignore[assignment]
        self.key_prefix = "metadata/"

    def _prepare_metadata_for_upload(self, metadata: GenerationMetadata) -> Tuple[str, str]:
        filename = metadata.get_metadata_filename()
        s3_key = generate_s3_key(filename, self.key_prefix)
        json_content = metadata.to_json()
        return s3_key, json_content

    def _create_s3_metadata(self, metadata: GenerationMetadata) -> Dict[str, str]:
        prompt_preview = (
            metadata.prompt[:100] + "..." if len(metadata.prompt) > 100 else metadata.prompt
        )
        return {
            "content-hash": metadata.content_hash or "",
            "generation-timestamp": metadata.timestamp or "",
            "prompt-preview": prompt_preview,
        }

    def _upload_metadata_to_s3(
        self, s3_key: str, json_content: str, s3_metadata: Dict[str, str]
    ) -> None:
        self.s3_client.put_object(
            Bucket=self.bucket_name,
            Key=s3_key,
            Body=json_content.encode("utf-8"),
            ContentType="application/json",
            ACL="public-read",  # Make publicly accessible
            Metadata=s3_metadata,
        )
        logging.debug("Metadata saved to S3: %s", s3_key)

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
            s3_key, json_content = self._prepare_metadata_for_upload(metadata)
            s3_metadata = self._create_s3_metadata(metadata)
            self._upload_metadata_to_s3(s3_key, json_content, s3_metadata)
            return s3_key

        except ClientError as e:
            error_code = e.response["Error"]["Code"]
            raise FileOperationError(
                f"Failed to save metadata to S3: {error_code}",
                operation="save_metadata",
                file_path=locals().get("s3_key", "unknown"),
            ) from e

        except Exception as e:
            raise FileOperationError(
                f"Unexpected error saving metadata: {str(e)}",
                operation="save_metadata",
                file_path="unknown",
            ) from e

    def _handle_load_metadata_error(self, error: Exception, metadata_key: str) -> None:
        if isinstance(error, ClientError):
            error_code = error.response["Error"]["Code"]
            if error_code == "NoSuchKey":
                raise FileOperationError(
                    f"Metadata not found: {metadata_key}",
                    operation="load_metadata",
                    file_path=metadata_key,
                ) from error
            raise FileOperationError(
                f"Failed to load metadata from S3: {error_code}",
                operation="load_metadata",
                file_path=metadata_key,
            ) from error
        raise FileOperationError(
            f"Unexpected error loading metadata: {str(error)}",
            operation="load_metadata",
            file_path=metadata_key,
        ) from error

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
            json_content = response["Body"].read().decode("utf-8")
            metadata = GenerationMetadata.from_json(json_content)
            logging.debug("Metadata loaded from S3: %s", metadata_key)
            return metadata
        except Exception as e:  # pylint: disable=broad-exception-caught
            self._handle_load_metadata_error(e, metadata_key)
            # The above method always raises, so this return is unreachable
            raise  # pragma: no cover

    def metadata_exists(self, content_hash: str) -> Optional[str]:
        """
        Check if metadata exists for given content hash.

        Args:
            content_hash: SHA256 hash of generation inputs

        Returns:
            S3 key if metadata exists, None otherwise
        """
        try:
            response = self._list_metadata_objects()
            if response is None:
                return None

            return self._find_matching_metadata_key(response, content_hash)

        except ClientError as e:
            logging.warning("Error checking metadata existence: %s", e)
            return None

    def _list_metadata_objects(self) -> Optional["ListObjectsV2OutputTypeDef"]:
        """List S3 objects with metadata prefix."""
        response = self.s3_client.list_objects_v2(
            Bucket=self.bucket_name,
            Prefix=self.key_prefix,
            MaxKeys=1000,  # Should be sufficient for deduplication checks
        )

        if "Contents" not in response:
            return None

        return response

    def _find_matching_metadata_key(
        self, response: "ListObjectsV2OutputTypeDef", content_hash: str
    ) -> Optional[str]:
        """Find metadata key with matching content hash."""
        hash_prefix = content_hash[:8]

        for obj in response["Contents"]:
            key = obj["Key"]
            if f"metadata_{hash_prefix}" in key:
                if self._verify_metadata_hash_match(key, content_hash):
                    return key

        return None

    def _verify_metadata_hash_match(self, key: str, content_hash: str) -> bool:
        """Verify metadata file has matching content hash."""
        try:
            metadata = self.load_metadata(key)
            return metadata.content_hash == content_hash
        except FileOperationError:
            # Skip corrupted or inaccessible metadata files
            return False

    def list_metadata_by_hash_prefix(self, hash_prefix: str) -> List[str]:
        """
        List metadata keys by content hash prefix.

        Args:
            hash_prefix: Hash prefix to search for

        Returns:
            List of S3 keys matching prefix
        """
        try:
            return self._collect_matching_metadata_keys(hash_prefix)
        except ClientError as e:
            logging.warning("Error listing metadata by hash prefix: %s", e)
            return []

    def _collect_matching_metadata_keys(self, hash_prefix: str) -> List[str]:
        """Collect all metadata keys matching the hash prefix."""
        matching_keys = []
        paginator = self.s3_client.get_paginator("list_objects_v2")
        pages = paginator.paginate(Bucket=self.bucket_name, Prefix=self.key_prefix)

        for page in pages:
            if "Contents" in page:
                page_keys = self._extract_matching_keys_from_page(page, hash_prefix)
                matching_keys.extend(page_keys)

        return matching_keys

    def _extract_matching_keys_from_page(
        self, page: "ListObjectsV2OutputTypeDef", hash_prefix: str
    ) -> List[str]:
        """Extract keys matching hash prefix from a single page."""
        matching_keys = []
        target_pattern = f"metadata_{hash_prefix}"

        for obj in page["Contents"]:
            key = obj["Key"]
            if target_pattern in key:
                matching_keys.append(key)

        return matching_keys
