"""
Unit tests for SeedreamClient.
Tests URL detection, API parameter handling, and image generation logic.
"""

__author__ = "Lene Preuss <lene.preuss@gmail.com>"

from pathlib import Path
from unittest.mock import Mock, patch, mock_open

import pytest

from stable_delusion.seedream import SeedreamClient
from stable_delusion.exceptions import ImageGenerationError, AuthenticationError


class TestSeedreamClient:  # pylint: disable=too-many-public-methods
    """Test SeedreamClient functionality."""

    @pytest.fixture
    def mock_ark_client(self):
        mock_client = Mock()

        # Mock successful response
        mock_response = Mock()
        mock_response.data = [Mock()]
        mock_response.data[0].url = "https://generated-image.com/result.jpg"
        mock_client.images.generate.return_value = mock_response

        return mock_client

    @pytest.fixture
    def seedream_client(self, mock_ark_client):
        with patch("stable_delusion.seedream.Ark", return_value=mock_ark_client):
            client = SeedreamClient("test-api-key")
            client.client = mock_ark_client  # Ensure the mock is used
        return client

    def test_generate_image_with_https_urls(self, seedream_client, mock_ark_client):
        https_urls = [
            "https://test-bucket.s3.us-east-1.amazonaws.com/image1.jpg",
            "https://test-bucket.s3.us-east-1.amazonaws.com/image2.jpg",
        ]

        result = seedream_client.generate_image("Edit these images", https_urls)

        assert len(result) == 1
        assert result[0] == "https://generated-image.com/result.jpg"

        # Verify API was called with image URLs
        mock_ark_client.images.generate.assert_called_once()
        call_kwargs = mock_ark_client.images.generate.call_args[1]
        assert "image" in call_kwargs
        assert call_kwargs["image"] == https_urls

    def test_generate_image_with_http_urls(self, seedream_client, mock_ark_client):
        http_urls = ["http://test-bucket.s3.us-east-1.amazonaws.com/image1.jpg"]

        result = seedream_client.generate_image("Edit this image", http_urls)

        assert len(result) == 1
        # Verify API was called with image URLs
        call_kwargs = mock_ark_client.images.generate.call_args[1]
        assert "image" in call_kwargs
        assert call_kwargs["image"] == http_urls

    def test_generate_image_with_invalid_urls_skips_them(self, seedream_client, mock_ark_client):
        invalid_urls = ["/tmp/local_image.jpg", "not-a-url", ""]

        with patch("stable_delusion.seedream.logging") as mock_logging:
            result = seedream_client.generate_image("Edit these images", invalid_urls)

        # Should still generate an image (text-to-image mode since no valid URLs)
        assert len(result) == 1

        # Verify warning was logged
        mock_logging.warning.assert_called()
        warning_calls = [call[0][0] for call in mock_logging.warning.call_args_list]
        assert any("Skipping invalid URL" in call for call in warning_calls)

        # Verify no images were passed to API
        call_kwargs = mock_ark_client.images.generate.call_args[1]
        assert "image" not in call_kwargs or not call_kwargs.get("image")

    def test_generate_image_url_normalization(self, seedream_client, mock_ark_client):
        malformed_urls = [
            "https:/test-bucket.s3.us-east-1.amazonaws.com/image1.jpg",  # Missing slash
            "http:/test-bucket.s3.us-east-1.amazonaws.com/image2.jpg",  # Missing slash
        ]

        result = seedream_client.generate_image("Edit these images", malformed_urls)

        assert len(result) == 1

        # Verify API was called with corrected URLs
        call_kwargs = mock_ark_client.images.generate.call_args[1]
        assert "image" in call_kwargs
        # URLs should be fixed
        expected_urls = [
            "https://test-bucket.s3.us-east-1.amazonaws.com/image1.jpg",
            "http://test-bucket.s3.us-east-1.amazonaws.com/image2.jpg",
        ]
        assert call_kwargs["image"] == expected_urls

    def test_generate_image_url_validation_edge_cases(self, seedream_client, mock_ark_client):
        valid_urls = [
            "https://test-bucket.s3.us-east-1.amazonaws.com/image1.jpg",
            "http://example.com/image2.jpg",
        ]

        seedream_client.generate_image("Edit these images", valid_urls)

        # Verify all valid URLs were passed to API
        call_kwargs = mock_ark_client.images.generate.call_args[1]
        assert "image" in call_kwargs
        assert call_kwargs["image"] == valid_urls

    def test_generate_image_max_10_images(self, seedream_client, mock_ark_client):
        many_urls = [f"https://test.com/image{i}.jpg" for i in range(15)]

        seedream_client.generate_image("Edit these images", many_urls)

        # Verify only first 10 URLs were passed to API
        call_kwargs = mock_ark_client.images.generate.call_args[1]
        assert "image" in call_kwargs
        assert len(call_kwargs["image"]) == 10
        assert call_kwargs["image"] == many_urls[:10]

    def test_generate_and_save_with_image_urls(self, seedream_client):
        image_urls = ["https://test.com/remote.jpg"]

        mock_result_path = Path("/tmp/result.png")
        with patch.object(
            seedream_client, "generate_image", return_value=["https://result.jpg"]
        ) as mock_generate:
            with patch.object(seedream_client, "download_image", return_value=mock_result_path):
                # Mock the Path.exists and Path.stat methods globally
                with patch("pathlib.Path.exists", return_value=True):
                    with patch("pathlib.Path.stat") as mock_stat:
                        mock_stat.return_value.st_size = 1024
                        seedream_client.generate_and_save(
                            prompt="Test prompt", output_dir=Path("/tmp"), image_urls=image_urls
                        )

        # Should use image_urls with default image_size
        mock_generate.assert_called_once_with("Test prompt", image_urls, "2K")

    def test_generate_and_save_without_image_urls(self, seedream_client):
        mock_result_path = Path("/tmp/result.png")
        with patch.object(
            seedream_client, "generate_image", return_value=["https://result.jpg"]
        ) as mock_generate:
            with patch.object(seedream_client, "download_image", return_value=mock_result_path):
                # Mock the Path.exists and Path.stat methods globally
                with patch("pathlib.Path.exists", return_value=True):
                    with patch("pathlib.Path.stat") as mock_stat:
                        mock_stat.return_value.st_size = 1024
                        seedream_client.generate_and_save(
                            prompt="Test prompt", output_dir=Path("/tmp")
                        )

        mock_generate.assert_called_once_with("Test prompt", None, "2K")

    def test_generate_and_save_none_inputs(self, seedream_client):
        mock_result_path = Path("/tmp/result.png")
        with patch.object(
            seedream_client, "generate_image", return_value=["https://result.jpg"]
        ) as mock_generate:
            with patch.object(seedream_client, "download_image", return_value=mock_result_path):
                # Mock the Path.exists and Path.stat methods globally
                with patch("pathlib.Path.exists", return_value=True):
                    with patch("pathlib.Path.stat") as mock_stat:
                        mock_stat.return_value.st_size = 1024
                        seedream_client.generate_and_save(
                            prompt="Test prompt", output_dir=Path("/tmp")
                        )

        mock_generate.assert_called_once_with("Test prompt", None, "2K")

    def test_generate_and_save_with_custom_image_size(self, seedream_client):
        mock_result_path = Path("/tmp/result.png")
        with patch.object(
            seedream_client, "generate_image", return_value=["https://result.jpg"]
        ) as mock_generate:
            with patch.object(seedream_client, "download_image", return_value=mock_result_path):
                with patch("pathlib.Path.exists", return_value=True):
                    with patch("pathlib.Path.stat") as mock_stat:
                        mock_stat.return_value.st_size = 1024
                        seedream_client.generate_and_save(
                            prompt="Test prompt", output_dir=Path("/tmp"), image_size="4K"
                        )

        mock_generate.assert_called_once_with("Test prompt", None, "4K")

    def test_api_parameters_structure(self, seedream_client, mock_ark_client):
        seedream_client.generate_image(
            prompt="Test prompt",
            image_urls=["https://test.com/image.jpg"],
            image_size="4K",
            seed=12345,
        )

        call_kwargs = mock_ark_client.images.generate.call_args[1]

        # Check core parameters
        assert call_kwargs["model"] == "seedream-4-0-250828"
        assert call_kwargs["prompt"] == "Test prompt"
        assert call_kwargs["size"] == "4K"  # Should use provided image_size
        assert call_kwargs["sequential_image_generation"] == "disabled"
        assert call_kwargs["response_format"] == "url"
        assert call_kwargs["watermark"] is True
        assert call_kwargs["seed"] == 12345
        assert call_kwargs["image"] == ["https://test.com/image.jpg"]

    def test_api_parameters_without_seed(self, seedream_client, mock_ark_client):
        seedream_client.generate_image("Test prompt")

        call_kwargs = mock_ark_client.images.generate.call_args[1]
        assert "seed" not in call_kwargs

    def test_api_parameters_default_size(self, seedream_client, mock_ark_client):
        seedream_client.generate_image("Test prompt")

        call_kwargs = mock_ark_client.images.generate.call_args[1]
        assert call_kwargs["size"] == "2K"  # Should use default size

    def test_api_parameters_without_images(self, seedream_client, mock_ark_client):
        seedream_client.generate_image("Test prompt")

        call_kwargs = mock_ark_client.images.generate.call_args[1]
        assert "image" not in call_kwargs

    def test_response_parsing_success(self, seedream_client, mock_ark_client):
        # Mock response with multiple images
        mock_response = Mock()
        mock_response.data = [Mock(), Mock()]
        mock_response.data[0].url = "https://result1.jpg"
        mock_response.data[1].url = "https://result2.jpg"
        mock_ark_client.images.generate.return_value = mock_response

        result = seedream_client.generate_image("Test prompt")

        assert len(result) == 2
        assert result[0] == "https://result1.jpg"
        assert result[1] == "https://result2.jpg"

    def test_response_parsing_no_images(self, seedream_client, mock_ark_client):
        mock_response = Mock()
        mock_response.data = []
        mock_ark_client.images.generate.return_value = mock_response

        with pytest.raises(ImageGenerationError) as exc_info:
            seedream_client.generate_image("Test prompt")

        assert "No images were generated by Seedream API" in str(exc_info.value)

    def test_response_parsing_no_data_attribute(self, seedream_client, mock_ark_client):
        mock_response = Mock()
        del mock_response.data  # No data attribute
        mock_ark_client.images.generate.return_value = mock_response

        with pytest.raises(ImageGenerationError) as exc_info:
            seedream_client.generate_image("Test prompt")

        assert "No images were generated by Seedream API" in str(exc_info.value)

    def test_authentication_error_handling(self, seedream_client, mock_ark_client):
        mock_ark_client.images.generate.side_effect = Exception("401 Unauthorized")

        with pytest.raises(AuthenticationError) as exc_info:
            seedream_client.generate_image("Test prompt")

        assert "Invalid API key for BytePlus Seedream API" in str(exc_info.value)

    def test_general_error_handling(self, seedream_client, mock_ark_client):
        mock_ark_client.images.generate.side_effect = Exception("Network error")

        with pytest.raises(ImageGenerationError) as exc_info:
            seedream_client.generate_image("Test prompt")

        assert "Seedream image generation failed: Network error" in str(exc_info.value)

    @patch("requests.get")
    def test_download_image_success(self, mock_requests_get, seedream_client):
        mock_response = Mock()
        mock_response.content = b"fake_image_data"
        mock_requests_get.return_value = mock_response

        output_path = Path("/tmp/test_image.png")

        with patch("builtins.open", mock_open()) as mock_file:
            with patch("pathlib.Path.mkdir") as mock_mkdir:
                result = seedream_client.download_image("https://test.com/image.jpg", output_path)

        assert result == output_path
        mock_requests_get.assert_called_once_with("https://test.com/image.jpg", timeout=60)
        mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)
        mock_file.assert_called_once_with(output_path, "wb")

    @patch("requests.get")
    def test_download_image_failure(self, mock_requests_get, seedream_client):
        mock_requests_get.side_effect = Exception("Download failed")

        with pytest.raises(ImageGenerationError) as exc_info:
            seedream_client.download_image("https://test.com/image.jpg", Path("/tmp/test.png"))

        assert "Failed to download image from" in str(exc_info.value)
        assert "Download failed" in str(exc_info.value)

    @patch("stable_delusion.seedream.generate_timestamped_filename")
    def test_generate_and_save_filename_generation(self, mock_timestamp, seedream_client):
        mock_timestamp.return_value = "seedream_generated_2025-09-27-12:34:56.png"

        mock_result_path = Path("/tmp/result.png")
        with patch.object(seedream_client, "generate_image", return_value=["https://result.jpg"]):
            with patch.object(seedream_client, "download_image", return_value=mock_result_path):
                # Mock the Path.exists and Path.stat methods globally
                with patch("pathlib.Path.exists", return_value=True):
                    with patch("pathlib.Path.stat") as mock_stat:
                        mock_stat.return_value.st_size = 1024
                        seedream_client.generate_and_save(
                            prompt="Test prompt", output_dir=Path("/tmp"), base_name="custom_base"
                        )

        mock_timestamp.assert_called_once_with("custom_base", "png")

    def test_create_with_env_key_success(self):
        with patch.dict("os.environ", {"ARK_API_KEY": "env-api-key"}):
            with patch("stable_delusion.seedream.Ark") as mock_ark:
                SeedreamClient.create_with_env_key()

        mock_ark.assert_called_once_with(api_key="env-api-key")

    def test_create_with_env_key_missing(self):
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(AuthenticationError) as exc_info:
                SeedreamClient.create_with_env_key()

        assert "BytePlus ARK API key not found in environment variable: ARK_API_KEY" in str(
            exc_info.value
        )

    def test_create_with_custom_env_var(self):
        with patch.dict("os.environ", {"CUSTOM_KEY": "custom-api-key"}):
            with patch("stable_delusion.seedream.Ark") as mock_ark:
                SeedreamClient.create_with_env_key("CUSTOM_KEY")

        mock_ark.assert_called_once_with(api_key="custom-api-key")

    def test_url_detection_edge_cases(self, seedream_client, mock_ark_client):
        edge_cases = [
            "https://",  # Incomplete URL
            "http://",  # Incomplete URL
            "ftp://test.com/file.jpg",  # Non-HTTP protocol
            "",  # Empty string
        ]

        # These should be treated as invalid URLs and skipped
        with patch("stable_delusion.seedream.logging"):
            seedream_client.generate_image("Test prompt", edge_cases)

        # Should not pass any images to API since all URLs are invalid
        call_kwargs = mock_ark_client.images.generate.call_args[1]
        assert "image" not in call_kwargs or not call_kwargs.get("image")

    def test_url_detection_valid_domain_without_file(self, seedream_client, mock_ark_client):
        valid_domain_urls = ["https://bucket.s3.amazonaws.com/"]

        seedream_client.generate_image("Test prompt", valid_domain_urls)

        # Should pass the valid domain URL to API
        call_kwargs = mock_ark_client.images.generate.call_args[1]
        assert "image" in call_kwargs
        assert call_kwargs["image"] == valid_domain_urls
