"""Unit tests for repository implementations."""

import tempfile
import time
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

import pytest
from PIL import Image
from werkzeug.datastructures import FileStorage

from stable_delusion.exceptions import FileOperationError, ValidationError
from stable_delusion.repositories.local_file_repository import LocalFileRepository
from stable_delusion.repositories.local_image_repository import LocalImageRepository
from stable_delusion.repositories.upload_repository import LocalUploadRepository


class TestLocalImageRepository:
    """Test cases for LocalImageRepository."""

    @pytest.fixture
    def repository(self):
        return LocalImageRepository()

    @pytest.fixture
    def test_image(self):
        image = Image.new("RGB", (100, 100), color="red")
        return image

    @pytest.fixture
    def temp_dir(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    def test_save_image_success(self, repository, test_image, temp_dir):
        file_path = temp_dir / "test_image.png"

        result = repository.save_image(test_image, file_path)

        assert result == file_path
        assert file_path.exists()

        # Verify the image was saved correctly
        loaded_image = Image.open(file_path)
        assert loaded_image.size == (100, 100)

    def test_save_image_creates_directories(self, repository, test_image, temp_dir):
        file_path = temp_dir / "subdir" / "test_image.png"

        result = repository.save_image(test_image, file_path)

        assert result == file_path
        assert file_path.exists()
        assert file_path.parent.exists()

    def test_save_image_failure(self, repository, test_image):
        # Try to save to an invalid path
        invalid_path = Path("/dev/null/test.png")

        with pytest.raises(FileOperationError) as excinfo:
            repository.save_image(test_image, invalid_path)

        assert "Failed to save image" in str(excinfo.value)

    def test_load_image_success(self, repository, test_image, temp_dir):
        file_path = temp_dir / "test_image.png"
        test_image.save(str(file_path))

        loaded_image = repository.load_image(file_path)

        assert loaded_image.size == (100, 100)

    def test_load_image_not_found(self, repository, temp_dir):
        file_path = temp_dir / "nonexistent.png"

        with pytest.raises(FileOperationError) as excinfo:
            repository.load_image(file_path)

        assert "Failed to load image" in str(excinfo.value)

    def test_validate_image_file_success(self, repository, test_image, temp_dir):
        file_path = temp_dir / "test_image.png"
        test_image.save(str(file_path))

        result = repository.validate_image_file(file_path)

        assert result is True

    def test_validate_image_file_not_exists(self, repository, temp_dir):
        file_path = temp_dir / "nonexistent.png"

        with pytest.raises(FileOperationError) as excinfo:
            repository.validate_image_file(file_path)

        assert "File does not exist" in str(excinfo.value)

    def test_validate_image_file_not_file(self, repository, temp_dir):
        with pytest.raises(FileOperationError) as excinfo:
            repository.validate_image_file(temp_dir)

        assert "Path is not a file" in str(excinfo.value)

    def test_validate_image_file_invalid_image(self, repository, temp_dir):
        file_path = temp_dir / "not_image.txt"
        file_path.write_text("This is not an image")

        with pytest.raises(FileOperationError) as excinfo:
            repository.validate_image_file(file_path)

        assert "File is not a valid image" in str(excinfo.value)

    @patch("stable_delusion.repositories.local_image_repository.generate_timestamped_filename")
    def test_generate_image_path(self, mock_generate, repository, temp_dir):
        mock_generate.return_value = "generated_test_123.png"

        result = repository.generate_image_path("test", temp_dir)

        expected = temp_dir / "generated_test_123.png"
        assert result == expected
        mock_generate.assert_called_once_with("test")


class TestLocalFileRepository:
    """Test cases for LocalFileRepository."""

    @pytest.fixture
    def repository(self):
        return LocalFileRepository()

    @pytest.fixture
    def temp_dir(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    def test_exists_true(self, repository, temp_dir):
        file_path = temp_dir / "test.txt"
        file_path.write_text("test content")

        result = repository.exists(file_path)

        assert result is True

    def test_exists_false(self, repository, temp_dir):
        file_path = temp_dir / "nonexistent.txt"

        result = repository.exists(file_path)

        assert result is False

    def test_create_directory_success(self, repository, temp_dir):
        new_dir = temp_dir / "new_directory"

        result = repository.create_directory(new_dir)

        assert result == new_dir
        assert new_dir.exists()
        assert new_dir.is_dir()

    def test_create_directory_nested(self, repository, temp_dir):
        nested_dir = temp_dir / "level1" / "level2" / "level3"

        result = repository.create_directory(nested_dir)

        assert result == nested_dir
        assert nested_dir.exists()
        assert nested_dir.is_dir()

    def test_delete_file_success(self, repository, temp_dir):
        file_path = temp_dir / "test.txt"
        file_path.write_text("test content")

        result = repository.delete_file(file_path)

        assert result is True
        assert not file_path.exists()

    def test_delete_file_not_exists(self, repository, temp_dir):
        file_path = temp_dir / "nonexistent.txt"

        result = repository.delete_file(file_path)

        assert result is False

    def test_move_file_success(self, repository, temp_dir):
        source = temp_dir / "source.txt"
        destination = temp_dir / "destination.txt"
        source.write_text("test content")

        result = repository.move_file(source, destination)

        assert result == destination
        assert not source.exists()
        assert destination.exists()
        assert destination.read_text() == "test content"

    def test_move_file_with_directory_creation(self, repository, temp_dir):
        source = temp_dir / "source.txt"
        destination = temp_dir / "subdir" / "destination.txt"
        source.write_text("test content")

        result = repository.move_file(source, destination)

        assert result == destination
        assert destination.exists()
        assert destination.parent.exists()


class TestLocalUploadRepository:
    """Test cases for LocalUploadRepository."""

    @pytest.fixture
    def repository(self):
        return LocalUploadRepository()

    @pytest.fixture
    def temp_dir(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    @pytest.fixture
    def mock_file(self):
        file_storage = FileStorage(
            stream=BytesIO(b"fake image data"), filename="test_image.png", content_type="image/png"
        )
        return file_storage

    def test_save_uploaded_files_success(self, repository, temp_dir, mock_file):
        files = [mock_file]

        result = repository.save_uploaded_files(files, temp_dir)

        assert len(result) == 1
        assert result[0].exists()
        assert result[0].parent == temp_dir

    def test_save_uploaded_files_multiple(self, repository, temp_dir):
        files = [
            FileStorage(stream=BytesIO(b"file 1"), filename="file1.png", content_type="image/png"),
            FileStorage(stream=BytesIO(b"file 2"), filename="file2.png", content_type="image/png"),
        ]

        result = repository.save_uploaded_files(files, temp_dir)

        assert len(result) == 2
        assert all(f.exists() for f in result)

    def test_generate_secure_filename_with_filename(self, repository):
        result = repository.generate_secure_filename("test_image.png")

        assert result == "test_image.png"

    def test_generate_secure_filename_with_malicious_filename(self, repository):
        result = repository.generate_secure_filename("../../../evil.png")

        # werkzeug.secure_filename should sanitize this
        assert ".." not in result
        assert "/" not in result

    def test_generate_secure_filename_no_filename(self, repository):
        result = repository.generate_secure_filename(None, "123456")

        assert result == "uploaded_file_123456.bin"

    def test_generate_secure_filename_empty_filename(self, repository):
        result = repository.generate_secure_filename("", "123456")

        assert result == "uploaded_file_123456.bin"

    def test_cleanup_old_uploads(self, repository, temp_dir):
        # Create old and new files
        old_file = temp_dir / "old_file.txt"
        new_file = temp_dir / "new_file.txt"

        old_file.write_text("old content")
        new_file.write_text("new content")

        # Make old file actually old
        import os

        old_time = time.time() - (25 * 3600)  # 25 hours ago
        os.utime(old_file, (old_time, old_time))

        result = repository.cleanup_old_uploads(temp_dir, max_age_hours=24)

        assert result == 1
        assert not old_file.exists()
        assert new_file.exists()

    def test_cleanup_old_uploads_no_directory(self, repository, temp_dir):
        nonexistent_dir = temp_dir / "nonexistent"

        result = repository.cleanup_old_uploads(nonexistent_dir)

        assert result == 0

    def test_validate_uploaded_file_success(self, repository, mock_file):
        result = repository.validate_uploaded_file(mock_file)

        assert result is True

    def test_validate_uploaded_file_no_file(self, repository):
        with pytest.raises(ValidationError) as excinfo:
            repository.validate_uploaded_file(None)

        assert "No file provided" in str(excinfo.value)

    def test_validate_uploaded_file_no_filename(self, repository):
        file_storage = FileStorage(
            stream=BytesIO(b"content"),
            filename="",  # Empty string instead of None
            content_type="image/png",
        )

        with pytest.raises(ValidationError) as excinfo:
            repository.validate_uploaded_file(file_storage)

        assert "No filename provided" in str(excinfo.value)

    def test_validate_uploaded_file_invalid_content_type(self, repository):
        file_storage = FileStorage(
            stream=BytesIO(b"content"), filename="test.txt", content_type="text/plain"
        )

        with pytest.raises(ValidationError) as excinfo:
            repository.validate_uploaded_file(file_storage)

        assert "Invalid file type" in str(excinfo.value)
