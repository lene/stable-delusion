"""
S3-based file repository implementation.
Provides cloud storage for files including uploads using Amazon S3.
"""

__author__ = "Lene Preuss <lene.preuss@gmail.com>"

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional

from werkzeug.datastructures import FileStorage
from werkzeug.utils import secure_filename

from stable_delusion.config import Config
from stable_delusion.exceptions import FileOperationError, ValidationError
from stable_delusion.repositories.interfaces import FileRepository
from stable_delusion.repositories.s3_client import (
    S3ClientManager,
    generate_s3_key,
    build_s3_url,
    parse_s3_url,
)
from stable_delusion.utils import get_current_timestamp

if TYPE_CHECKING:
    from mypy_boto3_s3 import S3Client


class S3FileRepository(FileRepository):
    """S3-based implementation of FileRepository interface."""

    def __init__(self, config: Config):
        self.config = config
        self.s3_client: "S3Client" = S3ClientManager.create_s3_client(config)
        # S3ClientManager validation ensures bucket_name is not None
        self.bucket_name: str = config.s3_bucket  # type: ignore[assignment]
        self.key_prefix = "files/"

    def exists(self, file_path: Path) -> bool:
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
                Metadata={"type": "directory_marker", "created_by": "stable-delusion"},
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
        try:
            s3_key = self._extract_s3_key(file_path)

            self.s3_client.delete_object(Bucket=self.bucket_name, Key=s3_key)
            logging.info("File deleted from S3: %s", s3_key)
            return True

        except (self.s3_client.exceptions.ClientError, OSError, ValueError) as e:
            logging.warning("Failed to delete S3 file %s: %s", file_path, e)
            return False

    def move_file(self, source: Path, destination: Path) -> Path:
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

    def _process_s3_objects(self, pages, pattern: Optional[str]) -> List[Path]:
        file_paths = []
        for page in pages:
            contents = page.get("Contents", [])
            for obj in contents:
                key = obj["Key"]
                if key.endswith("/"):  # Skip directory markers
                    continue
                filename = Path(key).name
                if pattern is None or self._matches_pattern(filename, pattern):
                    s3_url = build_s3_url(self.bucket_name, key)
                    file_paths.append(Path(s3_url))
        return file_paths

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
            dir_prefix = generate_s3_key(str(directory_path).strip("/") + "/", self.key_prefix)
            paginator = self.s3_client.get_paginator("list_objects_v2")
            pages = paginator.paginate(Bucket=self.bucket_name, Prefix=dir_prefix)
            return self._process_s3_objects(pages, pattern)
        except Exception as e:
            raise FileOperationError(
                f"Failed to list S3 files: {str(e)}",
                file_path=str(directory_path),
                operation="list_files_s3",
            ) from e

    def get_file_size(self, file_path: Path) -> int:
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

    def _collect_old_files(self, pages, cutoff_time) -> List[Dict[str, str]]:
        files_to_delete = []
        for page in pages:
            if "Contents" in page:
                for obj in page["Contents"]:
                    key = obj["Key"]
                    last_modified = obj["LastModified"].replace(tzinfo=None)
                    if not key.endswith("/") and last_modified < cutoff_time:
                        files_to_delete.append({"Key": key})
        return files_to_delete

    def _delete_files_in_batches(self, files_to_delete: List[Dict[str, str]]) -> int:
        deleted_count = 0
        batch_size = 1000  # S3 batch delete supports up to 1000 objects per request
        for i in range(0, len(files_to_delete), batch_size):
            batch = files_to_delete[i : i + batch_size]
            self.s3_client.delete_objects(
                Bucket=self.bucket_name,
                Delete={"Objects": batch},  # type: ignore[typeddict-item]
            )
            deleted_count += len(batch)
        return deleted_count

    def cleanup_old_files(self, directory_path: Path, max_age_hours: int = 24) -> int:
        try:
            cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
            dir_prefix = generate_s3_key(str(directory_path).strip("/") + "/", self.key_prefix)

            paginator = self.s3_client.get_paginator("list_objects_v2")
            pages = paginator.paginate(Bucket=self.bucket_name, Prefix=dir_prefix)

            files_to_delete = self._collect_old_files(pages, cutoff_time)
            deleted_count = self._delete_files_in_batches(files_to_delete) if files_to_delete else 0

            logging.info("Cleaned up %d old files from S3: %s", deleted_count, dir_prefix)
            return deleted_count
        except Exception as e:
            raise FileOperationError(
                f"Failed to cleanup old S3 files: {str(e)}",
                file_path=str(directory_path),
                operation="cleanup_old_files_s3",
            ) from e

    def _extract_s3_key(self, file_path: Path) -> str:
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
        import fnmatch

        return fnmatch.fnmatch(filename, pattern)

    def save_uploaded_files(self, files: List[FileStorage], upload_dir: Path) -> List[Path]:
        try:
            # Create upload directory marker if needed
            self.create_directory(upload_dir)

            saved_files = []
            for file in files:
                # Validate the uploaded file
                if not self.validate_uploaded_file(file):
                    continue

                # Generate secure filename
                timestamp = get_current_timestamp("compact")
                filename = self.generate_secure_filename(file.filename, timestamp)

                # Create S3 key
                upload_path = f"{str(upload_dir).strip('/')}/{filename}"
                s3_key = generate_s3_key(upload_path, self.key_prefix)

                # Determine content type
                content_type = file.content_type or "application/octet-stream"

                # Upload to S3
                file.stream.seek(0)  # Reset stream position
                self.s3_client.put_object(
                    Bucket=self.bucket_name,
                    Key=s3_key,
                    Body=file.stream.read(),
                    ContentType=content_type,
                    Metadata={
                        "original_filename": filename,
                        "uploaded_by": "stable-delusion",
                        "upload_timestamp": timestamp,
                    },
                )

                # Return S3 URL
                s3_url = build_s3_url(self.bucket_name, s3_key)
                saved_files.append(Path(s3_url))
                logging.info("File uploaded to S3: %s", s3_key)

            return saved_files
        except Exception as e:
            raise FileOperationError(
                f"Failed to save uploaded files to S3: {str(e)}",
                file_path=str(upload_dir),
                operation="save_uploads_s3",
            ) from e

    def generate_secure_filename(
        self, filename: Optional[str], timestamp: Optional[str] = None
    ) -> str:
        if not filename:
            timestamp = timestamp or get_current_timestamp("compact")
            return f"uploaded_file_{timestamp}.bin"

        # Use werkzeug's secure_filename to sanitize
        secure_name = secure_filename(filename)

        # If secure_filename returns empty string, generate a fallback
        if not secure_name:
            timestamp = timestamp or get_current_timestamp("compact")
            return f"uploaded_file_{timestamp}.bin"

        return secure_name

    def cleanup_old_uploads(self, upload_dir: Path, max_age_hours: int = 24) -> int:
        # Reuse the existing cleanup_old_files method
        return self.cleanup_old_files(upload_dir, max_age_hours)

    def validate_uploaded_file(self, file: FileStorage) -> bool:
        if file is None:
            raise ValidationError("No file provided")

        if not file.filename:
            raise ValidationError("No filename provided")

        # Check if file has content
        if not hasattr(file, "stream") or not file.stream:
            raise ValidationError("File has no content")

        # Basic content type validation for images
        if file.content_type and not file.content_type.startswith("image/"):
            raise ValidationError(
                f"Invalid file type: {file.content_type}. Only images are allowed."
            )

        return True
