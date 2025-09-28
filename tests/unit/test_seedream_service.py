"""
Unit tests for SeedreamImageGenerationService.
Tests S3 upload functionality, image URL handling, and service integration.
"""

__author__ = "Lene Preuss <lene.preuss@gmail.com>"

from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from PIL import Image

from stable_delusion.services.seedream_service import SeedreamImageGenerationService
from stable_delusion.models.requests import GenerateImageRequest
from stable_delusion.models.responses import GenerateImageResponse
from stable_delusion.exceptions import ConfigurationError
from stable_delusion.repositories.s3_image_repository import S3ImageRepository


class TestSeedreamImageGenerationService:  # pylint: disable=too-many-public-methods
    """Test SeedreamImageGenerationService functionality."""

    @pytest.fixture
    def mock_seedream_client(self):
        mock_client = Mock()
        mock_client.generate_and_save.return_value = Path("/tmp/test_output.png")
        return mock_client

    @pytest.fixture
    def mock_s3_repository(self):
        mock_repo = Mock(spec=S3ImageRepository)
        mock_repo.save_image.return_value = Path(
            "https://test-bucket.s3.us-east-1.amazonaws.com/test.jpg"
        )
        return mock_repo

    @pytest.fixture
    def mock_local_repository(self):
        mock_repo = Mock()
        # Not an S3ImageRepository instance
        return mock_repo

    @pytest.fixture
    def service_with_s3_repo(self, mock_seedream_client, mock_s3_repository):
        return SeedreamImageGenerationService(
            seedream_client=mock_seedream_client, image_repository=mock_s3_repository
        )

    @pytest.fixture
    def service_with_local_repo(self, mock_seedream_client, mock_local_repository):
        return SeedreamImageGenerationService(
            seedream_client=mock_seedream_client, image_repository=mock_local_repository
        )

    @pytest.fixture
    def service_no_repo(self, mock_seedream_client):
        return SeedreamImageGenerationService(
            seedream_client=mock_seedream_client, image_repository=None
        )

    @pytest.fixture
    def mock_image(self):
        mock_img = Mock(spec=Image.Image)
        mock_img.__enter__ = Mock(return_value=mock_img)
        mock_img.__exit__ = Mock(return_value=None)
        return mock_img

    def test_upload_images_to_s3_success(self, service_with_s3_repo, mock_image):
        test_images = [Path("/tmp/test1.jpg"), Path("/tmp/test2.jpg")]

        with patch("PIL.Image.open", return_value=mock_image):
            with patch(
                "stable_delusion.utils.generate_timestamped_filename",
                side_effect=["file1.jpg", "file2.jpg"],
            ):
                urls = service_with_s3_repo.upload_images_to_s3(test_images)

        assert len(urls) == 2
        assert all(url.startswith("https://") for url in urls)
        # Verify save_image was called for each image
        assert service_with_s3_repo.image_repository.save_image.call_count == 2

    def test_upload_images_to_s3_url_normalization_fix(self, service_with_s3_repo, mock_image):
        test_images = [Path("/tmp/test.jpg")]

        # Mock repository to return malformed URL (missing one slash)
        service_with_s3_repo.image_repository.save_image.return_value = Path(
            "https:/test-bucket.s3.us-east-1.amazonaws.com/test.jpg"
        )

        with patch("PIL.Image.open", return_value=mock_image):
            with patch(
                "stable_delusion.utils.generate_timestamped_filename", return_value="file.jpg"
            ):
                urls = service_with_s3_repo.upload_images_to_s3(test_images)

        assert len(urls) == 1
        assert urls[0] == "https://test-bucket.s3.us-east-1.amazonaws.com/test.jpg"  # Fixed URL

    def test_upload_images_to_s3_no_repository_fails(self, service_no_repo):
        test_images = [Path("/tmp/test.jpg")]

        with pytest.raises(ConfigurationError) as exc_info:
            service_no_repo.upload_images_to_s3(test_images)

        assert "Image repository not configured for S3 uploads" in str(exc_info.value)
        assert exc_info.value.config_key == "image_repository"

    def test_upload_images_to_s3_non_s3_repository_fails(self, service_with_local_repo):
        test_images = [Path("/tmp/test.jpg")]

        with pytest.raises(ConfigurationError) as exc_info:
            service_with_local_repo.upload_images_to_s3(test_images)

        assert "S3 storage required for Seedream image uploads" in str(exc_info.value)
        assert exc_info.value.config_key == "storage_type"

    def test_upload_images_to_s3_file_error_handling(self, service_with_s3_repo):
        test_images = [Path("/nonexistent/test.jpg")]

        with patch("PIL.Image.open", side_effect=FileNotFoundError("File not found")):
            with pytest.raises(ConfigurationError) as exc_info:
                service_with_s3_repo.upload_images_to_s3(test_images)

        assert "Failed to upload image" in str(exc_info.value)
        assert exc_info.value.config_key == "s3_upload"

    def test_upload_images_to_s3_s3_upload_error_handling(self, service_with_s3_repo, mock_image):
        test_images = [Path("/tmp/test.jpg")]

        service_with_s3_repo.image_repository.save_image.side_effect = Exception("S3 upload failed")

        with patch("PIL.Image.open", return_value=mock_image):
            with patch(
                "stable_delusion.utils.generate_timestamped_filename", return_value="file.jpg"
            ):
                with pytest.raises(ConfigurationError) as exc_info:
                    service_with_s3_repo.upload_images_to_s3(test_images)

        assert "Failed to upload image" in str(exc_info.value)
        assert "S3 upload failed" in str(exc_info.value)

    def test_upload_files_interface_method(self, service_with_s3_repo, mock_image):
        test_images = [Path("/tmp/test.jpg")]

        with patch.object(
            service_with_s3_repo, "upload_images_to_s3", return_value=["https://test.com/image.jpg"]
        ) as mock_upload:
            urls = service_with_s3_repo.upload_files(test_images)

        mock_upload.assert_called_once_with(test_images)
        assert urls == ["https://test.com/image.jpg"]

    @patch("stable_delusion.config.ConfigManager.get_config")
    def test_generate_image_with_images_uploads_to_s3(
        self, mock_config, service_with_s3_repo, mock_image
    ):
        mock_config.return_value.default_output_dir = Path("/tmp")

        request = GenerateImageRequest(
            prompt="Edit this image",
            images=[Path("/tmp/test.jpg")],
            model="seedream",
            storage_type="s3",
        )

        with patch.object(
            service_with_s3_repo, "upload_images_to_s3", return_value=["https://test.com/image.jpg"]
        ) as mock_upload:
            service_with_s3_repo.generate_image(request)

        mock_upload.assert_called_once_with([Path("/tmp/test.jpg")])
        service_with_s3_repo.client.generate_and_save.assert_called_once()

        # Check that image_urls were passed to the client
        call_args = service_with_s3_repo.client.generate_and_save.call_args
        assert call_args[1]["image_urls"] == ["https://test.com/image.jpg"]

    @patch("stable_delusion.config.ConfigManager.get_config")
    def test_generate_image_without_images_skips_upload(self, mock_config, service_with_s3_repo):
        mock_config.return_value.default_output_dir = Path("/tmp")

        request = GenerateImageRequest(
            prompt="Generate a beautiful sunset",
            images=[],  # No input images
            model="seedream",
            storage_type="s3",
        )

        with patch.object(service_with_s3_repo, "upload_images_to_s3") as mock_upload:
            service_with_s3_repo.generate_image(request)

        mock_upload.assert_not_called()
        service_with_s3_repo.client.generate_and_save.assert_called_once()

        # Check that empty image_urls were passed to the client
        call_args = service_with_s3_repo.client.generate_and_save.call_args
        assert call_args[1]["image_urls"] == []

    @patch("stable_delusion.config.ConfigManager.get_config")
    def test_generate_image_response_structure(self, mock_config, service_with_s3_repo):
        mock_config.return_value.default_output_dir = Path("/tmp")

        request = GenerateImageRequest(
            prompt="Test prompt", images=[], model="seedream", storage_type="s3"
        )

        response = service_with_s3_repo.generate_image(request)

        assert isinstance(response, GenerateImageResponse)
        assert response.image_config.prompt == "Test prompt"
        assert response.image_config.generated_file == Path("/tmp/test_output.png")

    @patch("stable_delusion.config.ConfigManager.get_config")
    def test_generate_image_error_handling(self, mock_config, service_with_s3_repo):
        mock_config.return_value.default_output_dir = Path("/tmp")

        service_with_s3_repo.client.generate_and_save.side_effect = Exception("Generation failed")

        request = GenerateImageRequest(
            prompt="Test prompt", images=[], model="seedream", storage_type="s3"
        )

        response = service_with_s3_repo.generate_image(request)

        assert isinstance(response, GenerateImageResponse)
        assert response.image_config.generated_file is None  # Failed generation

    @patch("stable_delusion.services.seedream_service.SeedreamClient")
    def test_create_class_method_with_api_key(self, mock_client_class):
        mock_client_instance = Mock()
        mock_client_class.return_value = mock_client_instance

        service = SeedreamImageGenerationService.create(api_key="test-key")

        mock_client_class.assert_called_once_with("test-key")
        assert service.client == mock_client_instance

    @patch("stable_delusion.services.seedream_service.SeedreamClient")
    def test_create_class_method_without_api_key(self, mock_client_class):
        mock_client_instance = Mock()
        mock_client_class.create_with_env_key.return_value = mock_client_instance

        service = SeedreamImageGenerationService.create()

        mock_client_class.create_with_env_key.assert_called_once()
        assert service.client == mock_client_instance

    @patch("stable_delusion.services.seedream_service.SeedreamClient")
    def test_create_class_method_error_handling(self, mock_client_class):
        mock_client_class.create_with_env_key.side_effect = Exception("API key not found")

        with pytest.raises(ConfigurationError) as exc_info:
            SeedreamImageGenerationService.create()

        assert "Failed to create Seedream client" in str(exc_info.value)
        assert exc_info.value.config_key == "SEEDREAM_API_KEY"

    def test_upload_images_to_s3_timestamped_filenames(self, service_with_s3_repo, mock_image):
        test_images = [Path("/tmp/base.jpg")]

        with patch("PIL.Image.open", return_value=mock_image):
            with patch(
                "stable_delusion.utils.generate_timestamped_filename",
                return_value="seedream_input_base_2025-09-27-12:34:56.jpg",
            ) as mock_timestamp:
                service_with_s3_repo.upload_images_to_s3(test_images)

        mock_timestamp.assert_called_once_with("seedream_input_base", "jpg")

        # Verify the S3 path includes the timestamped filename
        call_args = service_with_s3_repo.image_repository.save_image.call_args
        s3_path = call_args[0][1]  # Second argument is the path
        assert "seedream_input_base_2025-09-27-12:34:56.jpg" in str(s3_path)
        assert "seedream/inputs" in str(s3_path)

    def test_upload_images_to_s3_path_structure(self, service_with_s3_repo, mock_image):
        test_images = [Path("/tmp/test.jpg")]

        with patch("PIL.Image.open", return_value=mock_image):
            with patch(
                "stable_delusion.utils.generate_timestamped_filename", return_value="file.jpg"
            ):
                service_with_s3_repo.upload_images_to_s3(test_images)

        # Verify the S3 path structure
        call_args = service_with_s3_repo.image_repository.save_image.call_args
        s3_path = call_args[0][1]  # Second argument is the path
        assert str(s3_path) == "seedream/inputs/file.jpg"
