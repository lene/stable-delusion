"""
Unit tests for error handling in Seedream S3 integration.
Tests various failure scenarios and error propagation.
"""

__author__ = "Lene Preuss <lene.preuss@gmail.com>"

from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from PIL import Image

from nano_api.services.seedream_service import SeedreamImageGenerationService
from nano_api.seedream import SeedreamClient
from nano_api.models.requests import GenerateImageRequest
from nano_api.exceptions import (
    ConfigurationError,
    ImageGenerationError,
    AuthenticationError,
    ValidationError,
)


class TestErrorHandling:  # pylint: disable=too-many-public-methods
    """Test error handling across the Seedream S3 integration."""

    @pytest.fixture
    def mock_seedream_client(self):
        return Mock(spec=SeedreamClient)

    @pytest.fixture
    def mock_s3_repository(self):
        from nano_api.repositories.s3_image_repository import S3ImageRepository

        return Mock(spec=S3ImageRepository)

    @pytest.fixture
    def mock_local_repository(self):
        return Mock()

    def test_seedream_s3_upload_failure(self, mock_seedream_client, mock_s3_repository):
        service = SeedreamImageGenerationService(
            seedream_client=mock_seedream_client, image_repository=mock_s3_repository
        )

        # Mock S3 upload failure
        mock_s3_repository.save_image.side_effect = Exception("S3 connection timeout")

        test_images = [Path("/tmp/test.jpg")]

        mock_img = Mock(spec=Image.Image)
        mock_img.__enter__ = Mock(return_value=mock_img)
        mock_img.__exit__ = Mock(return_value=None)
        with patch("PIL.Image.open", return_value=mock_img):
            with pytest.raises(ConfigurationError) as exc_info:
                service.upload_images_to_s3(test_images)

        assert "Failed to upload image" in str(exc_info.value)
        assert "S3 connection timeout" in str(exc_info.value)
        assert exc_info.value.config_key == "s3_upload"

    def test_seedream_api_403_error(self, mock_seedream_client):
        # Mock 403 error response
        mock_seedream_client.generate_image.side_effect = ImageGenerationError(
            "Error code: 403 - Access denied to image URL"
        )

        with pytest.raises(ImageGenerationError) as exc_info:
            mock_seedream_client.generate_image(
                "Test prompt", ["https://private-bucket.com/image.jpg"]
            )

        assert "Access denied to image URL" in str(exc_info.value)

    def test_seedream_api_400_error_invalid_parameters(self, mock_seedream_client):
        mock_seedream_client.generate_image.side_effect = ImageGenerationError(
            "Error code: 400 - Invalid parameter: size"
        )

        with pytest.raises(ImageGenerationError) as exc_info:
            mock_seedream_client.generate_image("Test prompt")

        assert "Invalid parameter" in str(exc_info.value)

    def test_configuration_error_no_s3_repository(self, mock_seedream_client):
        service = SeedreamImageGenerationService(
            seedream_client=mock_seedream_client, image_repository=None  # No repository configured
        )

        test_images = [Path("/tmp/test.jpg")]

        with pytest.raises(ConfigurationError) as exc_info:
            service.upload_images_to_s3(test_images)

        assert "Image repository not configured for S3 uploads" in str(exc_info.value)
        assert exc_info.value.config_key == "image_repository"

    def test_configuration_error_wrong_repository_type(
        self, mock_seedream_client, mock_local_repository
    ):
        service = SeedreamImageGenerationService(
            seedream_client=mock_seedream_client,
            image_repository=mock_local_repository,  # Not S3 repository
        )

        test_images = [Path("/tmp/test.jpg")]

        with pytest.raises(ConfigurationError) as exc_info:
            service.upload_images_to_s3(test_images)

        assert "S3 storage required for Seedream image uploads" in str(exc_info.value)
        assert exc_info.value.config_key == "storage_type"

    def test_file_operation_error_invalid_image(self, mock_seedream_client, mock_s3_repository):
        service = SeedreamImageGenerationService(
            seedream_client=mock_seedream_client, image_repository=mock_s3_repository
        )

        test_images = [Path("/tmp/corrupted.jpg")]

        # Mock PIL Image.open failure
        with patch("PIL.Image.open", side_effect=Exception("Image file is corrupted")):
            with pytest.raises(ConfigurationError) as exc_info:
                service.upload_images_to_s3(test_images)

        assert "Failed to upload image" in str(exc_info.value)
        assert "Image file is corrupted" in str(exc_info.value)

    def test_file_operation_error_nonexistent_file(self, mock_seedream_client, mock_s3_repository):
        service = SeedreamImageGenerationService(
            seedream_client=mock_seedream_client, image_repository=mock_s3_repository
        )

        test_images = [Path("/nonexistent/file.jpg")]

        with patch("PIL.Image.open", side_effect=FileNotFoundError("No such file or directory")):
            with pytest.raises(ConfigurationError) as exc_info:
                service.upload_images_to_s3(test_images)

        assert "Failed to upload image" in str(exc_info.value)
        assert "No such file or directory" in str(exc_info.value)

    def test_authentication_error_invalid_api_key(self):
        with patch("nano_api.seedream.Ark") as mock_ark:
            mock_ark.side_effect = Exception("401 Unauthorized")

            with pytest.raises(Exception):  # Constructor error
                SeedreamClient("invalid-api-key")

    def test_authentication_error_missing_env_key(self):
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(AuthenticationError) as exc_info:
                SeedreamClient.create_with_env_key()

        assert "BytePlus ARK API key not found in environment variable" in str(exc_info.value)

    def test_image_generation_error_no_response_data(self):
        with patch("nano_api.seedream.Ark") as mock_ark:
            mock_client = Mock()
            mock_response = Mock()
            mock_response.data = []  # Empty response
            mock_client.images.generate.return_value = mock_response
            mock_ark.return_value = mock_client

            client = SeedreamClient("test-key")

            with pytest.raises(ImageGenerationError) as exc_info:
                client.generate_image("Test prompt")

        assert "No images were generated by Seedream API" in str(exc_info.value)

    def test_image_generation_error_malformed_response(self):
        with patch("nano_api.seedream.Ark") as mock_ark:
            mock_client = Mock()
            mock_response = Mock()
            del mock_response.data  # No data attribute
            mock_client.images.generate.return_value = mock_response
            mock_ark.return_value = mock_client

            client = SeedreamClient("test-key")

            with pytest.raises(ImageGenerationError) as exc_info:
                client.generate_image("Test prompt")

        assert "No images were generated by Seedream API" in str(exc_info.value)

    def test_validation_error_invalid_model_and_storage_combination(self):
        with pytest.raises(ValidationError) as exc_info:
            GenerateImageRequest(
                prompt="Edit this image",
                images=[Path("test.jpg")],
                model="seedream",
                storage_type="local",
            )

        assert "Seedream model with input images requires S3 storage type" in str(exc_info.value)
        assert exc_info.value.field == "storage_type"

    def test_validation_error_invalid_storage_type(self):
        with pytest.raises(ValidationError) as exc_info:
            GenerateImageRequest(
                prompt="Test prompt",
                images=[Path("test.jpg")],  # Provide images to avoid other validation error
                model="gemini",
                storage_type="invalid_storage",
            )

        assert "Storage type must be 'local' or 's3'" in str(exc_info.value)
        assert exc_info.value.field == "storage_type"

    def test_validation_error_invalid_model(self):
        with pytest.raises(ValidationError) as exc_info:
            GenerateImageRequest(
                prompt="Test prompt",
                images=[Path("test.jpg")],  # Provide images to avoid other validation error
                model="invalid_model",
                storage_type="local",
            )

        assert "Model must be one of" in str(exc_info.value)
        assert exc_info.value.field == "model"

    def test_validation_error_empty_prompt(self):
        with pytest.raises(ValidationError) as exc_info:
            GenerateImageRequest(
                prompt="", images=[], model="seedream", storage_type="local"  # Empty prompt
            )

        assert "Prompt cannot be empty" in str(exc_info.value)
        assert exc_info.value.field == "prompt"

    @patch("requests.get")
    def test_download_image_network_error(self, mock_requests):
        mock_requests.side_effect = Exception("Network timeout")

        with patch("nano_api.seedream.Ark") as mock_ark:
            mock_ark.return_value = Mock()
            client = SeedreamClient("test-key")

            with pytest.raises(ImageGenerationError) as exc_info:
                client.download_image("https://test.com/image.jpg", Path("/tmp/output.png"))

        assert "Failed to download image from" in str(exc_info.value)
        assert "Network timeout" in str(exc_info.value)

    @patch("requests.get")
    def test_download_image_http_error(self, mock_requests):
        mock_response = Mock()
        mock_response.raise_for_status.side_effect = Exception("404 Not Found")
        mock_requests.return_value = mock_response

        with patch("nano_api.seedream.Ark") as mock_ark:
            mock_ark.return_value = Mock()
            client = SeedreamClient("test-key")

            with pytest.raises(ImageGenerationError) as exc_info:
                client.download_image("https://test.com/image.jpg", Path("/tmp/output.png"))

        assert "Failed to download image from" in str(exc_info.value)
        assert "404 Not Found" in str(exc_info.value)

    def test_service_creation_error_invalid_api_key(self):
        with patch("nano_api.services.seedream_service.SeedreamClient") as mock_client_class:
            mock_client_class.create_with_env_key.side_effect = Exception("Invalid API key")

            with pytest.raises(ConfigurationError) as exc_info:
                SeedreamImageGenerationService.create()

        assert "Failed to create Seedream client" in str(exc_info.value)
        assert exc_info.value.config_key == "SEEDREAM_API_KEY"

    def test_generate_image_service_error_recovery(self, mock_seedream_client, mock_s3_repository):
        service = SeedreamImageGenerationService(
            seedream_client=mock_seedream_client, image_repository=mock_s3_repository
        )

        # Mock client failure
        mock_seedream_client.generate_and_save.side_effect = Exception("Generation failed")

        request = GenerateImageRequest(
            prompt="Test prompt", images=[], model="seedream", storage_type="s3"
        )

        with patch("nano_api.config.ConfigManager.get_config") as mock_config:
            mock_config.return_value.default_output_dir = Path("/tmp")
            response = service.generate_image(request)

        # Should return response with None generated_file (indicating failure)
        assert response.image_config.generated_file is None

    def test_error_message_details_preserved(self, mock_seedream_client, mock_s3_repository):
        service = SeedreamImageGenerationService(
            seedream_client=mock_seedream_client, image_repository=mock_s3_repository
        )

        original_error = "S3 bucket 'test-bucket' access denied: insufficient permissions"
        mock_s3_repository.save_image.side_effect = Exception(original_error)

        test_images = [Path("/tmp/test.jpg")]

        mock_img = Mock(spec=Image.Image)
        mock_img.__enter__ = Mock(return_value=mock_img)
        mock_img.__exit__ = Mock(return_value=None)
        with patch("PIL.Image.open", return_value=mock_img):
            with pytest.raises(ConfigurationError) as exc_info:
                service.upload_images_to_s3(test_images)

        # Error should preserve the original error message
        error_str = str(exc_info.value)
        assert "Failed to upload image /tmp/test.jpg to S3" in error_str
        assert original_error in error_str

    def test_chained_exception_preservation(self, mock_seedream_client, mock_s3_repository):
        service = SeedreamImageGenerationService(
            seedream_client=mock_seedream_client, image_repository=mock_s3_repository
        )

        original_exception = FileNotFoundError("File not found")
        mock_s3_repository.save_image.side_effect = original_exception

        test_images = [Path("/tmp/test.jpg")]

        mock_img = Mock(spec=Image.Image)
        mock_img.__enter__ = Mock(return_value=mock_img)
        mock_img.__exit__ = Mock(return_value=None)
        with patch("PIL.Image.open", return_value=mock_img):
            with pytest.raises(ConfigurationError) as exc_info:
                service.upload_images_to_s3(test_images)

        # Original exception should be preserved in the chain
        assert exc_info.value.__cause__ == original_exception
