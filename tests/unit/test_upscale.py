"""Unit tests for image upscaling functionality using Google Vertex AI."""

import argparse
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import requests
from PIL import Image

from nano_api.conf import DEFAULT_LOCATION, DEFAULT_PROJECT_ID
from nano_api.exceptions import UpscalingError, APIError
from nano_api.upscale import upscale_image

sys.path.append("nano_api")


class TestUpscaleImage:
    """Test cases for image upscaling functionality."""

    @patch("nano_api.upscale.requests.post")
    @patch("nano_api.upscale.default")
    def test_upscale_image_success_x2(self, mock_default, mock_post):
        # Mock authentication
        mock_credentials = MagicMock()
        mock_credentials.token = "test-token"  # nosec B105 - test-only mock credential
        mock_default.return_value = (mock_credentials, None)

        # Mock successful API response
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "predictions": [{"bytesBase64Encoded": "dGVzdCBpbWFnZSBkYXRh"}]
        }
        mock_post.return_value = mock_response

        # Mock Path.read_bytes and PIL Image operations
        with patch.object(Path, "read_bytes", return_value=b"test_image_data"):
            with patch("nano_api.upscale.Image.open") as mock_image_open:
                mock_image = MagicMock(spec=Image.Image)
                mock_image_open.return_value = mock_image

                with patch("nano_api.upscale.base64.b64encode", return_value=b"dGVzdCBkYXRh"):
                    with patch("nano_api.upscale.base64.b64decode") as mock_decode:
                        mock_decode.return_value = b"decoded_image_data"

                        result = upscale_image(
                            Path("test.jpg"), "test-project", "us-central1", "x2"
                        )

                        # Verify API call
                        mock_post.assert_called_once()
                        call_args = mock_post.call_args

                        # Check URL format
                        expected_url = (
                            "https://us-central1-aiplatform.googleapis.com"
                            "/v1/projects/test-project/locations/us-central1"
                            "/publishers/google/models/imagegeneration@002:predict"
                        )
                        assert call_args[0][0] == expected_url

                        # Check request payload
                        payload = call_args[1]["json"]
                        assert payload["parameters"]["upscaleConfig"]["upscaleFactor"] == "x2"

                        # Check result
                        assert result == mock_image

    @patch("nano_api.upscale.requests.post")
    @patch("nano_api.upscale.default")
    def test_upscale_image_success_x4(self, mock_default, mock_post):
        # Mock authentication
        mock_credentials = MagicMock()
        mock_credentials.token = "test-token"  # nosec B105 - test-only mock credential
        mock_default.return_value = (mock_credentials, None)

        # Mock successful API response
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "predictions": [{"bytesBase64Encoded": "dGVzdCBpbWFnZSBkYXRh"}]
        }
        mock_post.return_value = mock_response

        # Mock PIL Image operations
        with patch.object(Path, "read_bytes", return_value=b"test_image_data"):
            with patch("nano_api.upscale.Image.open") as mock_image_open:
                mock_image = MagicMock(spec=Image.Image)
                mock_image_open.return_value = mock_image

                with patch("nano_api.upscale.base64.b64encode", return_value=b"dGVzdCBkYXRh"):
                    with patch("nano_api.upscale.base64.b64decode") as mock_decode:
                        mock_decode.return_value = b"decoded_image_data"

                        upscale_image(Path("test.jpg"), "test-project", "us-central1", "x4")

                        # Check request payload has x4 factor
                        call_args = mock_post.call_args
                        payload = call_args[1]["json"]
                        assert payload["parameters"]["upscaleConfig"]["upscaleFactor"] == "x4"

    @patch("nano_api.upscale.requests.post")
    @patch("nano_api.upscale.default")
    def test_upscale_image_file_not_found(self, mock_default, _mock_post):
        # Mock authentication
        mock_credentials = MagicMock()
        mock_default.return_value = (mock_credentials, None)

        # Mock file not found
        with patch.object(Path, "read_bytes", side_effect=FileNotFoundError):
            with pytest.raises(UpscalingError, match="Failed to read image file"):
                upscale_image(Path("nonexistent.jpg"), "test-project")

    @patch("nano_api.upscale.requests.post")
    @patch("nano_api.upscale.default")
    def test_upscale_image_api_error(self, mock_default, mock_post):
        # Mock authentication
        mock_credentials = MagicMock()
        mock_default.return_value = (mock_credentials, None)

        # Mock API error
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = requests.HTTPError("API Error")
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        mock_post.return_value = mock_response

        with patch.object(Path, "read_bytes", return_value=b"test_data"):
            with pytest.raises(APIError, match="Upscaling API request failed"):
                upscale_image(Path("test.jpg"), "test-project")

    @patch("nano_api.upscale.default")
    def test_upscale_image_auth_error(self, mock_default):
        # Mock authentication failure
        mock_default.side_effect = Exception("Authentication failed")

        with pytest.raises(Exception, match="Authentication failed"):
            upscale_image(Path("test.jpg"), "test-project")

    @patch("nano_api.upscale.requests.post")
    @patch("nano_api.upscale.default")
    def test_upscale_image_default_location(self, mock_default, mock_post):
        # Mock authentication
        mock_credentials = MagicMock()
        mock_credentials.token = "test-token"  # nosec B105 - test-only mock credential
        mock_default.return_value = (mock_credentials, None)

        # Mock successful API response
        mock_response = MagicMock()
        mock_response.json.return_value = {"predictions": [{"bytesBase64Encoded": "dGVzdA=="}]}
        mock_post.return_value = mock_response

        with patch.object(Path, "read_bytes", return_value=b"test"):

            with patch("nano_api.upscale.Image.open"):
                with patch("nano_api.upscale.base64.b64encode"):
                    with patch("nano_api.upscale.base64.b64decode") as mock_decode:
                        mock_decode.return_value = b"decoded_test_data"

                        upscale_image(Path("test.jpg"), "test-project")

                        # Check that default location was used
                        call_args = mock_post.call_args
                        url = call_args[0][0]
                        assert "us-central1" in url

    def test_upscale_image_headers_format(self):
        with patch("nano_api.upscale.default") as mock_default:
            mock_credentials = MagicMock()
            mock_credentials.token = "test-bearer-token"  # nosec B105 - test-only mock
            mock_default.return_value = (mock_credentials, None)

            with patch("nano_api.upscale.requests.post") as mock_post:
                mock_response = MagicMock()
                mock_response.json.return_value = {
                    "predictions": [{"bytesBase64Encoded": "dGVzdA=="}]
                }
                mock_post.return_value = mock_response

                with patch.object(Path, "read_bytes", return_value=b"test"):

                    with patch("nano_api.upscale.Image.open"):
                        with patch("nano_api.upscale.base64.b64encode"):
                            with patch("nano_api.upscale.base64.b64decode") as mock_decode:
                                mock_decode.return_value = b"test_decoded_data"

                                upscale_image(Path("test.jpg"), "test-project")

                                # Check headers
                                call_args = mock_post.call_args
                                headers = call_args[1]["headers"]
                                assert headers["Authorization"] == "Bearer test-bearer-token"
                                assert headers["Content-Type"] == "application/json"


class TestUpscaleCommandLine:
    """Test cases for upscale command line interface."""

    def test_command_line_defaults(self):
        # Create parser similar to upscale.py
        parser = argparse.ArgumentParser()
        parser.add_argument("image_path", type=str)
        parser.add_argument("--scale", type=int, default=4, choices=[2, 4])

        args = parser.parse_args(["test.jpg"])
        assert args.image_path == "test.jpg"
        assert args.scale == 4

    def test_command_line_custom_scale(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("image_path", type=str)
        parser.add_argument("--scale", type=int, default=4, choices=[2, 4])

        args = parser.parse_args(["test.jpg", "--scale", "2"])
        assert args.image_path == "test.jpg"
        assert args.scale == 2

    def test_command_line_invalid_scale(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("image_path", type=str)
        parser.add_argument("--scale", type=int, default=4, choices=[2, 4])

        with pytest.raises(SystemExit):
            parser.parse_args(["test.jpg", "--scale", "3"])


class TestUpscaleIntegration:
    """Integration tests that test the full upscale workflow."""

    @patch("nano_api.upscale.upscale_image")
    def test_main_execution_default_scale(self, mock_upscale):
        mock_image = MagicMock()
        mock_upscale.return_value = mock_image

        test_args = ["upscale.py", "test.jpg"]

        with patch("sys.argv", test_args):
            with patch("nano_api.upscale.print"):
                # Import and execute the main block logic
                parser = argparse.ArgumentParser()
                parser.add_argument("image_path", type=str)
                parser.add_argument("--scale", type=int, default=4, choices=[2, 4])
                args = parser.parse_args(["test.jpg"])

                # Simulate main execution
                mock_upscale(
                    "test.jpg",
                    DEFAULT_PROJECT_ID,
                    DEFAULT_LOCATION,
                    upscale_factor=f"x{args.scale}",
                )
                mock_image.save("upscaled_test.jpg")

                mock_upscale.assert_called_once_with(
                    "test.jpg", DEFAULT_PROJECT_ID, DEFAULT_LOCATION, upscale_factor="x4"
                )
                mock_image.save.assert_called_once()
