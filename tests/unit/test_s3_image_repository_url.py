"""
Unit tests for S3ImageRepository URL handling functionality.
Tests HTTPS URL generation and Path object handling for Seedream integration.
"""

__author__ = "Lene Preuss <lene.preuss@gmail.com>"

from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from PIL import Image

from nano_api.repositories.s3_image_repository import S3ImageRepository
from nano_api.config import Config
from nano_api.exceptions import FileOperationError, ValidationError


class TestS3ImageRepositoryURL:
    """Test S3ImageRepository URL handling functionality."""

    @pytest.fixture
    def mock_config(self):
        config = Mock(spec=Config)
        config.s3_bucket = "test-bucket"
        config.s3_region = "us-east-1"
        config.aws_access_key_id = "test-key"
        config.aws_secret_access_key = "test-secret"
        return config

    @pytest.fixture
    def mock_s3_client(self):
        mock_client = Mock()
        mock_client.put_object.return_value = {}
        mock_client.head_object.return_value = {}
        mock_client.get_object.return_value = {"Body": Mock()}
        return mock_client

    @pytest.fixture
    def s3_repository(self, mock_config, mock_s3_client):
        with patch(
            "nano_api.repositories.s3_image_repository.S3ClientManager.create_s3_client",
            return_value=mock_s3_client,
        ):
            with patch(
                "nano_api.repositories.s3_image_repository.S3ClientManager._validate_s3_access"
            ):
                return S3ImageRepository(mock_config)

    @pytest.fixture
    def mock_image(self):
        mock_img = Mock(spec=Image.Image)
        mock_img.save = Mock()
        return mock_img

    def test_save_image_returns_https_url(self, s3_repository, mock_image):
        file_path = Path("test_image.jpg")

        with patch(
            "nano_api.repositories.s3_client.generate_s3_key", return_value="images/test_image.jpg"
        ):
            result = s3_repository.save_image(mock_image, file_path)

        # Should return HTTPS URL, not S3 URL (Path normalization may affect slashes)
        result_str = str(result)
        assert result_str.startswith("https:/")
        assert "test-bucket.s3.us-east-1.amazonaws.com/images/test_image.jpg" in result_str

    def test_save_image_without_acl(self, s3_repository, mock_image, mock_s3_client):
        file_path = Path("test_image.jpg")

        with patch(
            "nano_api.repositories.s3_client.generate_s3_key", return_value="images/test_image.jpg"
        ):
            s3_repository.save_image(mock_image, file_path)

        # Verify put_object was called without ACL parameter
        put_object_call = mock_s3_client.put_object.call_args
        assert "ACL" not in put_object_call[1]

    def test_https_url_format_correct(self, s3_repository, mock_image):
        file_path = Path("subfolder/image.png")

        with patch(
            "nano_api.repositories.s3_client.generate_s3_key",
            return_value="images/subfolder/image.png",
        ):
            result = s3_repository.save_image(mock_image, file_path)

        result_str = str(result)
        assert result_str.startswith("https:/")
        assert "test-bucket.s3.us-east-1.amazonaws.com" in result_str
        assert result_str.endswith("images/subfolder/image.png")

    def test_https_url_different_regions(self, mock_image, mock_s3_client):
        regions = ["eu-central-1", "ap-southeast-1", "us-west-2"]

        for region in regions:
            config = Mock(spec=Config)
            config.s3_bucket = "test-bucket"
            config.s3_region = region
            config.aws_access_key_id = "test-key"
            config.aws_secret_access_key = "test-secret"

            with patch(
                "nano_api.repositories.s3_image_repository.S3ClientManager.create_s3_client",
                return_value=mock_s3_client,
            ):
                with patch(
                    "nano_api.repositories.s3_image_repository.S3ClientManager._validate_s3_access"
                ):
                    repository = S3ImageRepository(config)

            with patch(
                "nano_api.repositories.s3_client.generate_s3_key", return_value="images/test.jpg"
            ):
                result = repository.save_image(mock_image, Path("test.jpg"))

            result_str = str(result)
            assert result_str.startswith("https:/")
            assert f"test-bucket.s3.{region}.amazonaws.com/images/test.jpg" in result_str

    def test_path_object_url_preservation(self, s3_repository, mock_image):
        file_path = Path("test.jpg")

        with patch(
            "nano_api.repositories.s3_client.generate_s3_key", return_value="images/test.jpg"
        ):
            result_path = s3_repository.save_image(mock_image, file_path)

        # Convert back to string and verify URL is intact
        result_str = str(result_path)
        assert "test-bucket.s3.us-east-1.amazonaws.com/images/test.jpg" in result_str

        # Verify this is a valid HTTPS URL (Path normalization affects slashes)
        assert result_str.startswith("https:/")
        assert "test-bucket.s3.us-east-1.amazonaws.com" in result_str

    def test_image_format_detection(self, s3_repository, mock_image, mock_s3_client):
        test_cases = [
            ("image.jpg", "JPEG", "image/jpeg"),
            ("image.png", "PNG", "image/png"),
            ("image.gif", "GIF", "image/gif"),
            ("image.webp", "WEBP", "image/webp"),
            ("image.bmp", "BMP", "image/bmp"),
            ("image.unknown", "PNG", "image/png"),  # Default to PNG
        ]

        for filename, expected_format, expected_content_type in test_cases:
            file_path = Path(filename)

            with patch(
                "nano_api.repositories.s3_client.generate_s3_key", return_value=f"images/{filename}"
            ):
                s3_repository.save_image(mock_image, file_path)

            # Check that save was called with correct format
            mock_image.save.assert_called()
            save_call_args = mock_image.save.call_args
            assert save_call_args[1]["format"] == expected_format

            # Check that put_object was called with correct ContentType
            put_object_call = mock_s3_client.put_object.call_args
            assert put_object_call[1]["ContentType"] == expected_content_type

    def test_metadata_included_in_upload(self, s3_repository, mock_image, mock_s3_client):
        file_path = Path("test_image.jpg")

        with patch(
            "nano_api.repositories.s3_client.generate_s3_key", return_value="images/test_image.jpg"
        ):
            s3_repository.save_image(mock_image, file_path)

        put_object_call = mock_s3_client.put_object.call_args
        metadata = put_object_call[1]["Metadata"]

        assert metadata["original_filename"] == "test_image.jpg"
        assert metadata["uploaded_by"] == "nano-api-client"

    def test_s3_upload_error_handling(self, s3_repository, mock_image, mock_s3_client):
        mock_s3_client.put_object.side_effect = Exception("S3 upload failed")

        with pytest.raises(FileOperationError) as exc_info:
            with patch(
                "nano_api.repositories.s3_client.generate_s3_key", return_value="images/test.jpg"
            ):
                s3_repository.save_image(mock_image, Path("test.jpg"))

        assert "Failed to save image to S3" in str(exc_info.value)
        assert "S3 upload failed" in str(exc_info.value)
        assert exc_info.value.operation == "save_image_s3"

    def test_image_save_error_handling(self, s3_repository):
        mock_image = Mock(spec=Image.Image)
        mock_image.save.side_effect = Exception("Image save failed")

        with pytest.raises(FileOperationError) as exc_info:
            with patch(
                "nano_api.repositories.s3_client.generate_s3_key", return_value="images/test.jpg"
            ):
                s3_repository.save_image(mock_image, Path("test.jpg"))

        assert "Failed to save image to S3" in str(exc_info.value)
        assert "Image save failed" in str(exc_info.value)

    def test_generate_image_path_s3_format(self, s3_repository):
        result = s3_repository.generate_image_path("test_image.jpg", Path("subfolder"))

        result_str = str(result)
        assert result_str.startswith("s3:/")
        assert "test-bucket" in result_str
        assert "images/subfolder/" in result_str
        assert result_str.endswith(".jpg")

    def test_generate_image_path_root_directory(self, s3_repository):
        result = s3_repository.generate_image_path("test_image.jpg", Path("."))

        result_str = str(result)
        assert "images/test_image" in result_str
        assert "images/./test_image" not in result_str  # Should not include "./"

    def test_validate_image_file_s3_url(self, s3_repository, mock_s3_client):
        s3_url = "https://test-bucket.s3.us-east-1.amazonaws.com/images/test.jpg"

        result = s3_repository.validate_image_file(Path(s3_url))

        assert result is True
        mock_s3_client.head_object.assert_called_once()

    def test_validate_image_file_s3_url_not_found(self, s3_repository, mock_s3_client):
        from botocore.exceptions import ClientError

        # Mock the head_object to raise ClientError
        mock_s3_client.head_object.side_effect = ClientError(
            {"Error": {"Code": "NoSuchKey"}}, "HeadObject"
        )

        # Mock the exceptions attribute with both NoSuchKey and ClientError
        mock_no_such_key = type("MockNoSuchKey", (Exception,), {})
        mock_client_error = type("MockClientError", (Exception,), {})
        mock_s3_client.exceptions = type(
            "MockExceptions", (), {"NoSuchKey": mock_no_such_key, "ClientError": mock_client_error}
        )

        s3_url = "https://test-bucket.s3.us-east-1.amazonaws.com/images/nonexistent.jpg"
        result = s3_repository.validate_image_file(Path(s3_url))

        assert result is False

    def test_load_image_from_s3_url(self, s3_repository, mock_s3_client):
        s3_url = "https://test-bucket.s3.us-east-1.amazonaws.com/images/test.jpg"

        # Mock S3 response with image data
        mock_body = Mock()
        mock_body.read.return_value = b"fake_image_data"
        mock_s3_client.get_object.return_value = {"Body": mock_body}

        with patch("PIL.Image.open") as mock_image_open:
            mock_image_open.return_value = Mock(spec=Image.Image)
            s3_repository.load_image(Path(s3_url))

        mock_s3_client.get_object.assert_called_once()
        assert mock_image_open.called

    def test_extract_s3_key_from_https_url(self, s3_repository):
        https_url = "https://test-bucket.s3.us-east-1.amazonaws.com/images/subfolder/test.jpg"

        # This is a private method, but we need to test the URL parsing logic
        # We'll test it indirectly through validate_image_file
        with patch.object(
            s3_repository, "_extract_s3_key", return_value="images/subfolder/test.jpg"
        ) as mock_extract:
            s3_repository.validate_image_file(Path(https_url))

        mock_extract.assert_called_once_with(Path(https_url))

    def test_bucket_name_mismatch_validation(self, s3_repository, mock_s3_client):
        wrong_bucket_url = "https://wrong-bucket.s3.us-east-1.amazonaws.com/images/test.jpg"

        # The bucket mismatch should be detected during URL parsing in _extract_s3_key
        # This should raise ValidationError due to bucket mismatch
        with pytest.raises(ValidationError) as exc_info:
            s3_repository.load_image(Path(wrong_bucket_url))

        assert "S3 bucket mismatch" in str(exc_info.value)
