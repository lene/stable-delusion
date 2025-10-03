"""
Unit tests for SeedreamImageGenerationService.
Tests S3 upload functionality, image URL handling, and service integration.
"""

__author__ = "Lene Preuss <lene.preuss@gmail.com>"

from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from stable_delusion.services.seedream_service import SeedreamImageGenerationService
from stable_delusion.models.requests import GenerateImageRequest
from stable_delusion.models.responses import GenerateImageResponse
from stable_delusion.exceptions import ConfigurationError


class TestSeedreamImageGenerationService:  # pylint: disable=too-many-public-methods
    """Test SeedreamImageGenerationService functionality."""

    @pytest.fixture
    def mock_local_repository(self):
        mock_repo = Mock()
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

    def test_upload_images_to_s3_success(
        self, service_with_s3_repo, mock_pil_image_context_manager, mock_s3_file_repository
    ):
        test_images = [Path("/tmp/test1.jpg"), Path("/tmp/test2.jpg")]

        with patch("PIL.Image.open", return_value=mock_pil_image_context_manager):
            with patch(
                "stable_delusion.utils.generate_timestamped_filename",
                side_effect=["file1.jpg", "file2.jpg"],
            ):
                with patch(
                    "stable_delusion.repositories.s3_file_repository.S3FileRepository",
                    return_value=mock_s3_file_repository,
                ):
                    urls = service_with_s3_repo.upload_images_to_s3(test_images)

        assert len(urls) == 2
        assert all(url.startswith("https://") or url.startswith("s3://") for url in urls)

    def test_upload_images_to_s3_url_normalization_fix(
        self, service_with_s3_repo, mock_pil_image_context_manager, mock_s3_file_repository
    ):
        test_images = [Path("/tmp/test.jpg")]

        with patch("PIL.Image.open", return_value=mock_pil_image_context_manager):
            with patch(
                "stable_delusion.utils.generate_timestamped_filename", return_value="file.jpg"
            ):
                with patch(
                    "stable_delusion.repositories.s3_file_repository.S3FileRepository",
                    return_value=mock_s3_file_repository,
                ):
                    urls = service_with_s3_repo.upload_images_to_s3(test_images)

        assert len(urls) == 1
        assert urls[0].startswith("https://") or urls[0].startswith("s3://")

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

    def test_upload_images_to_s3_s3_upload_error_handling(
        self,
        service_with_s3_repo,
        mock_pil_image_context_manager,
        mock_config_with_s3,
        mock_s3_file_repository,
    ):
        test_images = [Path("/tmp/test.jpg")]
        mock_pil_image_context_manager.format = "JPEG"

        mock_s3_file_repository.s3_client.put_object.side_effect = Exception("S3 upload failed")

        with patch("PIL.Image.open", return_value=mock_pil_image_context_manager):
            with patch(
                "stable_delusion.utils.generate_timestamped_filename", return_value="file.jpg"
            ):
                with patch(
                    "stable_delusion.services.seedream_service.ConfigManager.get_config",
                    return_value=mock_config_with_s3,
                ):
                    with patch(
                        "stable_delusion.repositories.s3_file_repository.S3FileRepository",
                        return_value=mock_s3_file_repository,
                    ):
                        with pytest.raises(ConfigurationError) as exc_info:
                            service_with_s3_repo.upload_images_to_s3(test_images)

        assert "Failed to upload image" in str(exc_info.value)
        assert "S3 upload failed" in str(exc_info.value)

    def test_upload_files_interface_method(self, service_with_s3_repo):
        test_images = [Path("/tmp/test.jpg")]

        with patch.object(
            service_with_s3_repo, "upload_images_to_s3", return_value=["https://test.com/image.jpg"]
        ) as mock_upload:
            urls = service_with_s3_repo.upload_files(test_images)

        mock_upload.assert_called_once_with(test_images)
        assert urls == ["https://test.com/image.jpg"]

    @patch("stable_delusion.config.ConfigManager.get_config")
    def test_generate_image_with_images_uploads_to_s3(self, mock_config, service_with_s3_repo):
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

        call_args = service_with_s3_repo.client.generate_and_save.call_args
        assert call_args[1]["image_urls"] == ["https://test.com/image.jpg"]

    @patch("stable_delusion.config.ConfigManager.get_config")
    def test_generate_image_without_images_skips_upload(self, mock_config, service_with_s3_repo):
        mock_config.return_value.default_output_dir = Path("/tmp")

        request = GenerateImageRequest(
            prompt="Generate a beautiful sunset",
            images=[],
            model="seedream",
            storage_type="s3",
        )

        with patch.object(service_with_s3_repo, "upload_images_to_s3") as mock_upload:
            service_with_s3_repo.generate_image(request)

        mock_upload.assert_not_called()
        service_with_s3_repo.client.generate_and_save.assert_called_once()

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
        assert response.image_config.generated_file == Path("/tmp/generated_image.png")

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

    def test_upload_images_to_s3_timestamped_filenames(
        self, service_with_s3_repo, mock_pil_image_context_manager, mock_s3_file_repository
    ):
        test_images = [Path("/tmp/base.jpg")]

        with patch("PIL.Image.open", return_value=mock_pil_image_context_manager):
            with patch(
                "stable_delusion.utils.generate_timestamped_filename",
                return_value="base_2025-09-27-12:34:56.jpg",
            ) as mock_timestamp:
                with patch(
                    "stable_delusion.repositories.s3_file_repository.S3FileRepository",
                    return_value=mock_s3_file_repository,
                ):
                    service_with_s3_repo.upload_images_to_s3(test_images)

        mock_timestamp.assert_called_once_with("base", "jpg")

    def test_upload_images_to_s3_path_structure(
        self, service_with_s3_repo, mock_pil_image_context_manager, mock_s3_file_repository
    ):
        test_images = [Path("/tmp/test.jpg")]

        with patch("PIL.Image.open", return_value=mock_pil_image_context_manager):
            with patch(
                "stable_delusion.utils.generate_timestamped_filename", return_value="file.jpg"
            ):
                with patch(
                    "stable_delusion.repositories.s3_file_repository.S3FileRepository",
                    return_value=mock_s3_file_repository,
                ):
                    urls = service_with_s3_repo.upload_images_to_s3(test_images)

        assert len(urls) == 1
