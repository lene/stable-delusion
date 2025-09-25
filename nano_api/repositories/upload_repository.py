"""
Implementation of upload repository for handling file uploads.
Manages uploaded files with security and cleanup features.
"""

__author__ = "Lene Preuss <lene.preuss@gmail.com>"

import time
from pathlib import Path
from typing import List, Optional

from werkzeug.datastructures import FileStorage
from werkzeug.utils import secure_filename

from nano_api.exceptions import FileOperationError, ValidationError
from nano_api.repositories.interfaces import UploadRepository
from nano_api.utils import get_current_timestamp


class LocalUploadRepository(UploadRepository):
    """Local filesystem implementation of upload repository."""

    def save_uploaded_files(self, files: List[FileStorage],
                            upload_dir: Path) -> List[Path]:
        """
        Save uploaded files to the specified directory.

        Args:
            files: List of uploaded FileStorage objects
            upload_dir: Directory to save files

        Returns:
            List of saved file paths

        Raises:
            FileOperationError: If save operation fails
        """
        try:
            # Ensure upload directory exists
            upload_dir.mkdir(parents=True, exist_ok=True)

            saved_files = []
            for file in files:
                # Validate the uploaded file
                if not self.validate_uploaded_file(file):
                    continue

                # Generate secure filename
                timestamp = get_current_timestamp("compact")
                filename = self.generate_secure_filename(file.filename, timestamp)
                filepath = upload_dir / filename

                # Save the file
                file.save(str(filepath))
                saved_files.append(filepath)

            return saved_files
        except (OSError, IOError) as e:
            raise FileOperationError(
                f"Failed to save uploaded files to {upload_dir}",
                file_path=str(upload_dir),
                operation="save_uploads"
            ) from e

    def generate_secure_filename(self, filename: Optional[str],
                                 timestamp: Optional[str] = None) -> str:
        """
        Generate a secure filename for upload.

        Args:
            filename: Original filename (can be None)
            timestamp: Optional timestamp string

        Returns:
            Secure filename string
        """
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
        """
        Clean up old uploaded files.

        Args:
            upload_dir: Upload directory to clean
            max_age_hours: Maximum age of files to keep

        Returns:
            Number of files cleaned up

        Raises:
            FileOperationError: If cleanup fails
        """
        if not upload_dir.exists():
            return 0

        try:
            current_time = time.time()
            max_age_seconds = max_age_hours * 3600
            cleanup_count = 0

            for file_path in upload_dir.iterdir():
                if file_path.is_file():
                    file_age = current_time - file_path.stat().st_mtime
                    if file_age > max_age_seconds:
                        file_path.unlink()
                        cleanup_count += 1

            return cleanup_count
        except (OSError, IOError) as e:
            raise FileOperationError(
                f"Failed to cleanup uploads in {upload_dir}",
                file_path=str(upload_dir),
                operation="cleanup"
            ) from e

    def validate_uploaded_file(self, file: FileStorage) -> bool:
        """
        Validate an uploaded file.

        Args:
            file: FileStorage object to validate

        Returns:
            True if file is valid

        Raises:
            ValidationError: If file is invalid
        """
        if file is None:
            raise ValidationError("No file provided")

        if not file.filename:
            raise ValidationError("No filename provided")

        # Check if file has content
        if not hasattr(file, 'stream') or not file.stream:
            raise ValidationError("File has no content")

        # Basic content type validation for images
        if file.content_type and not file.content_type.startswith('image/'):
            raise ValidationError(
                f"Invalid file type: {file.content_type}. Only images are allowed."
            )

        return True
