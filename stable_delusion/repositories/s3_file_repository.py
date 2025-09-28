"""
S3-based file repository implementation.
Provides cloud storage for general files using Amazon S3.
"""

__author__ = "Lene Preuss <lene.preuss@gmail.com>"

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional

from stable_delusion.config import Config
from stable_delusion.exceptions import FileOperationError, ValidationError
from stable_delusion.repositories.interfaces import FileRepository
from stable_delusion.repositories.s3_client import (
    S3ClientManager,
    generate_s3_key,
    build_s3_url,
    parse_s3_url,
)

if TYPE_CHECKING:
    from mypy_boto3_s3 import S3Client


class S3FileRepository(FileRepository):
    """S3-based implementation of FileRepository interface."""

    def __init__(self, config: Config):
        """
        Initialize S3 file repository.

        Args:
            config: Application configuration containing S3 settings
        """
        self.config = config
        self.s3_client: "S3Client" = S3ClientManager.create_s3_client(config)
        # S3ClientManager validation ensures bucket_name is not None
        self.bucket_name: str = config.s3_bucket  # type: ignore[assignment]
        self.key_prefix = "files/"

    def exists(self, file_path: Path) -> bool:
        """
        Check if a file exists in S3.

        Args:
            file_path: S3 URL or key path to check

        Returns:
            True if file exists, False otherwise
        """
        try:
            s3_key = self._extract_s3_key(file_path)
            self.s3_client.head_object(Bucket=self.bucket_name, Key=s3_key)
            return True
        except self.s3_client.exceptions.NoSuchKey:
            return False
        except (self.s3_client.exceptions.ClientError, OSError, ValueError) as e:
            logging.warning("Error checking S3 file existence for %s: %s", file_path, e)
            return False

    def create_directory(self, dir_path: Path) -> Path:
        """
        Create a directory-like structure in S3.

        Note: S3 doesn't have real directories, but we can create a marker object
        to represent directory structure.

        Args:
            dir_path: Directory path to create in S3

        Returns:
            The directory path that was created
        """
        try:
            # Create a directory marker object
            s3_key = generate_s3_key(f"{str(dir_path).strip('/')}/", self.key_prefix)

            # Ensure the key ends with / to indicate directory
            if not s3_key.endswith("/"):
                s3_key += "/"

            # Create empty object as directory marker
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=s3_key,
                Body=b"",
                ContentType="application/x-directory",
                Metadata={"type": "directory_marker", "created_by": "nano-api-client"},
            )

            logging.info("S3 directory marker created: %s", s3_key)
            return dir_path

        except Exception as e:
            raise FileOperationError(
                f"Failed to create S3 directory marker: {str(e)}",
                file_path=str(dir_path),
                operation="create_directory_s3",
            ) from e

    def delete_file(self, file_path: Path) -> bool:
        """
        Delete a file from S3.

        Args:
            file_path: S3 URL or key path to delete

        Returns:
            True if file was deleted successfully, False otherwise
        """
        try:
            s3_key = self._extract_s3_key(file_path)

            self.s3_client.delete_object(Bucket=self.bucket_name, Key=s3_key)
            logging.info("File deleted from S3: %s", s3_key)
            return True

        except (self.s3_client.exceptions.ClientError, OSError, ValueError) as e:
            logging.warning("Failed to delete S3 file %s: %s", file_path, e)
            return False

    def move_file(self, source: Path, destination: Path) -> Path:
        """
        Move a file within S3 (copy then delete).

        Args:
            source: Source S3 URL or key path
            destination: Destination S3 URL or key path

        Returns:
            Destination path where file was moved

        Raises:
            FileOperationError: If move operation fails
        """
        try:
            source_key = self._extract_s3_key(source)
            dest_key = self._extract_s3_key(destination)

            # Copy object to new location
            copy_source = f"{self.bucket_name}/{source_key}"
            self.s3_client.copy_object(
                CopySource=copy_source,
                Bucket=self.bucket_name,
                Key=dest_key,
                MetadataDirective="COPY",
            )

            # Delete original object
            self.s3_client.delete_object(Bucket=self.bucket_name, Key=source_key)

            logging.info("File moved in S3: %s -> %s", source_key, dest_key)
            return destination

        except Exception as e:
            raise FileOperationError(
                f"Failed to move S3 file: {str(e)}",
                file_path=f"{source} -> {destination}",
                operation="move_file_s3",
            ) from e

    def list_files(self, directory_path: Path, pattern: Optional[str] = None) -> List[Path]:
        """
        List files in an S3 directory.

        Args:
            directory_path: S3 directory path to list
            pattern: Optional file pattern to filter (basic wildcard support)

        Returns:
            List of S3 URLs for matching files

        Raises:
            FileOperationError: If listing fails
        """
        try:
            # Generate S3 prefix
            dir_prefix = generate_s3_key(str(directory_path).strip("/") + "/", self.key_prefix)

            # List objects with prefix
            paginator = self.s3_client.get_paginator("list_objects_v2")
            pages = paginator.paginate(Bucket=self.bucket_name, Prefix=dir_prefix)

            file_paths = []
            for page in pages:
                contents = page.get("Contents", [])
                for obj in contents:
                    key = obj["Key"]

                    # Skip directory markers
                    if key.endswith("/"):
                        continue

                    filename = Path(key).name
                    if pattern is None or self._matches_pattern(filename, pattern):
                        s3_url = build_s3_url(self.bucket_name, key)
                        file_paths.append(Path(s3_url))

            return file_paths

        except Exception as e:
            raise FileOperationError(
                f"Failed to list S3 files: {str(e)}",
                file_path=str(directory_path),
                operation="list_files_s3",
            ) from e

    def get_file_size(self, file_path: Path) -> int:
        """
        Get the size of a file in S3.

        Args:
            file_path: S3 URL or key path

        Returns:
            File size in bytes

        Raises:
            FileOperationError: If operation fails
        """
        try:
            s3_key = self._extract_s3_key(file_path)
            response = self.s3_client.head_object(Bucket=self.bucket_name, Key=s3_key)
            return response["ContentLength"]

        except Exception as e:
            raise FileOperationError(
                f"Failed to get S3 file size: {str(e)}",
                file_path=str(file_path),
                operation="get_file_size_s3",
            ) from e

    def cleanup_old_files(self, directory_path: Path, max_age_hours: int = 24) -> int:
        """
        Clean up old files in S3 directory based on age.

        Args:
            directory_path: S3 directory path to clean
            max_age_hours: Maximum age in hours for files to keep

        Returns:
            Number of files deleted

        Raises:
            FileOperationError: If cleanup fails
        """
        try:
            cutoff_time = datetime.now() - timedelta(hours=max_age_hours)

            # List all files in directory
            files_to_delete = []
            dir_prefix = generate_s3_key(str(directory_path).strip("/") + "/", self.key_prefix)

            paginator = self.s3_client.get_paginator("list_objects_v2")
            pages = paginator.paginate(Bucket=self.bucket_name, Prefix=dir_prefix)

            for page in pages:
                if "Contents" in page:
                    for obj in page["Contents"]:
                        key = obj["Key"]
                        last_modified = obj["LastModified"].replace(tzinfo=None)

                        # Skip directory markers and check age
                        if not key.endswith("/") and last_modified < cutoff_time:
                            files_to_delete.append({"Key": key})

            # Delete old files in batches
            deleted_count = 0
            if files_to_delete:
                # S3 batch delete supports up to 1000 objects per request
                batch_size = 1000
                for i in range(0, len(files_to_delete), batch_size):
                    batch = files_to_delete[i:i + batch_size]

                    self.s3_client.delete_objects(
                        Bucket=self.bucket_name,
                        Delete={"Objects": batch},  # type: ignore[typeddict-item]
                    )
                    deleted_count += len(batch)

            logging.info("Cleaned up %d old files from S3: %s", deleted_count, dir_prefix)
            return deleted_count

        except Exception as e:
            raise FileOperationError(
                f"Failed to cleanup old S3 files: {str(e)}",
                file_path=str(directory_path),
                operation="cleanup_old_files_s3",
            ) from e

    def _extract_s3_key(self, file_path: Path) -> str:
        """
        Extract S3 key from file path (handles both URLs and keys).

        Args:
            file_path: S3 URL or key path

        Returns:
            S3 object key

        Raises:
            ValidationError: If path format is invalid
        """
        path_str = str(file_path)

        # Handle S3 URLs
        if path_str.startswith("s3://"):
            try:
                bucket, key = parse_s3_url(path_str)
                if bucket != self.bucket_name:
                    raise ValidationError(
                        f"S3 bucket mismatch: expected {self.bucket_name}, got {bucket}",
                        field="file_path",
                        value=path_str,
                    )
                return key
            except ValueError as e:
                raise ValidationError(
                    f"Invalid S3 URL format: {path_str}", field="file_path", value=path_str
                ) from e

        # Handle direct keys (remove leading slash if present)
        return path_str.lstrip("/")

    def _matches_pattern(self, filename: str, pattern: str) -> bool:
        """
        Simple wildcard pattern matching.

        Args:
            filename: Filename to check
            pattern: Pattern with * wildcard support

        Returns:
            True if filename matches pattern
        """
        import fnmatch

        return fnmatch.fnmatch(filename, pattern)
