"""
Local filesystem implementation of file repository.
Handles file operations including uploads on local filesystem.
"""

__author__ = "Lene Preuss <lene.preuss@gmail.com>"

import time
from pathlib import Path
from typing import List, Optional

from werkzeug.datastructures import FileStorage
from werkzeug.utils import secure_filename

from stable_delusion.exceptions import ValidationError
from stable_delusion.repositories.interfaces import FileRepository
from stable_delusion.utils import get_current_timestamp, safe_file_operation


class LocalFileRepository(FileRepository):
    """Local filesystem implementation of file repository with upload support."""

    def exists(self, file_path: Path) -> bool:
        return file_path.exists()

    def create_directory(self, dir_path: Path) -> Path:
        def _create_operation():
            dir_path.mkdir(parents=True, exist_ok=True)
            return dir_path

        return safe_file_operation("create_directory", str(dir_path), _create_operation)

    def delete_file(self, file_path: Path) -> bool:
        if not file_path.exists():
            return False

        def _delete_operation():
            file_path.unlink()
            return True

        return safe_file_operation("delete", str(file_path), _delete_operation)

    def move_file(self, source: Path, destination: Path) -> Path:
        def _move_operation():
            destination.parent.mkdir(parents=True, exist_ok=True)
            source.rename(destination)
            return destination

        return safe_file_operation(
            "move",
            f"{source} -> {destination}",
            _move_operation,
        )

    def save_uploaded_files(self, files: List[FileStorage], upload_dir: Path) -> List[Path]:
        def _save_operation():
            upload_dir.mkdir(parents=True, exist_ok=True)
            saved_files = []
            for file in files:
                if not self.validate_uploaded_file(file):
                    continue

                timestamp = get_current_timestamp("compact")
                filename = self.generate_secure_filename(file.filename, timestamp)
                filepath = upload_dir / filename
                file.save(str(filepath))
                saved_files.append(filepath)
            return saved_files

        return safe_file_operation("save_uploads", str(upload_dir), _save_operation)

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
        if not upload_dir.exists():
            return 0

        def _cleanup_operation():
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

        return safe_file_operation("cleanup", str(upload_dir), _cleanup_operation)

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
