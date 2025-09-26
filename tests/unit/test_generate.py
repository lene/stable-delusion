"""Unit tests for image generation functionality using Google Gemini API."""
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from nano_api.conf import DEFAULT_LOCATION, DEFAULT_PROJECT_ID
from nano_api.exceptions import ImageGenerationError, FileOperationError, ConfigurationError
from nano_api.generate import (
    GeminiClient,
    generate_from_images,
    parse_command_line,
    save_response_image,
)

from ..conftest import create_mock_gemini_response

sys.path.append("nano_api")


# Setup to prevent .env file loading in all tests
@pytest.fixture(autouse=True)
def prevent_dotenv_loading():
    """Prevent loading .env file during tests."""
    with patch('nano_api.config.load_dotenv'):
        yield


class TestGeminiClient:
    """Test cases for GeminiClient functionality."""
    def test_init_missing_api_key(self):
        """Test GeminiClient initialization without GEMINI_API_KEY."""
        from nano_api.config import ConfigManager
        with patch.dict(os.environ, {}, clear=True):
            ConfigManager.reset_config()  # Ensure clean config state
            with pytest.raises(
                ConfigurationError,
                match="GEMINI_API_KEY environment variable is required"
            ):
                GeminiClient()

    def test_init_successful(self):
        """Test successful GeminiClient initialization."""
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key", "STORAGE_TYPE": "local"}):
            with patch("nano_api.generate.genai.Client"):
                with patch("nano_api.generate.aiplatform.init"):
                    client = GeminiClient()
                    assert client.project_id == DEFAULT_PROJECT_ID
                    assert client.location == DEFAULT_LOCATION
                    assert client.output_dir == Path(".")

    def test_init_with_custom_params(self):
        """Test GeminiClient initialization with custom parameters."""
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key", "STORAGE_TYPE": "local"}):
            with patch("nano_api.generate.genai.Client"):
                with patch("nano_api.generate.aiplatform.init"):
                    custom_project = "custom-project"
                    custom_location = "custom-location"
                    custom_output_dir = Path("custom/output")

                    client = GeminiClient(
                        project_id=custom_project,
                        location=custom_location,
                        output_dir=custom_output_dir
                    )
                    assert client.project_id == custom_project
                    assert client.location == custom_location
                    assert client.output_dir == custom_output_dir

    def test_upload_files_success(self):
        """Test successful file upload."""
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key", "STORAGE_TYPE": "local"}):
            with patch("nano_api.generate.genai.Client") as mock_client_class:
                with patch("nano_api.generate.aiplatform.init"):
                    mock_client = MagicMock()
                    mock_client_class.return_value = mock_client

                    # Mock uploaded file
                    mock_uploaded_file = MagicMock()
                    mock_uploaded_file.name = "test_file"
                    mock_uploaded_file.mime_type = "image/png"
                    mock_uploaded_file.size_bytes = 1024
                    mock_uploaded_file.uri = "test_uri"

                    # Mock datetime objects for timestamps
                    from datetime import datetime
                    mock_uploaded_file.create_time = datetime.now()
                    mock_uploaded_file.expiration_time = datetime.now()

                    mock_client.files.upload.return_value = mock_uploaded_file

                    client = GeminiClient()

                    # Create temporary test files
                    with tempfile.TemporaryDirectory() as temp_dir:
                        test_file1 = Path(temp_dir) / "test1.png"
                        test_file2 = Path(temp_dir) / "test2.png"

                        # Create actual files
                        test_file1.write_bytes(b"test image data 1")
                        test_file2.write_bytes(b"test image data 2")

                        result = client.upload_files([test_file1, test_file2])

                        assert len(result) == 2
                        assert mock_client.files.upload.call_count == 2

    def test_upload_files_nonexistent(self):
        """Test upload_files with non-existent file."""
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}):
            with patch("nano_api.generate.genai.Client"):
                with patch("nano_api.generate.aiplatform.init"):
                    client = GeminiClient()

                    nonexistent_file = Path("nonexistent.png")

                    with pytest.raises(FileOperationError, match="Image file not found"):
                        client.upload_files([nonexistent_file])

    def test_generate_from_images_success(self):
        """Test successful image generation."""
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}):
            with patch("nano_api.generate.genai.Client") as mock_client_class:
                with patch("nano_api.generate.aiplatform.init"):
                    # Set up mocks
                    mock_client = MagicMock()
                    mock_client_class.return_value = mock_client

                    # Mock response using helper
                    mock_response = create_mock_gemini_response(b"fake_image_data")

                    mock_client.models.generate_content.return_value = mock_response
                    mock_client.files.upload.return_value = MagicMock()

                    # Mock PIL Image operations
                    with patch("nano_api.generate.Image.open") as mock_image_open:
                        mock_image = MagicMock()
                        mock_image_open.return_value = mock_image

                        with patch("nano_api.utils.get_current_timestamp") as mock_timestamp:
                            mock_timestamp.return_value = "2024-01-01-12:00:00"

                            client = GeminiClient()

                            # Create temporary test files
                            with tempfile.TemporaryDirectory() as temp_dir:
                                test_file = Path(temp_dir) / "test.png"
                                test_file.write_bytes(b"test image data")

                                result = client.generate_from_images("test prompt", [test_file])

                                expected_result = Path("./generated_2024-01-01-12:00:00.png")
                                assert result == expected_result
                                mock_image.save.assert_called_once()

    def test_generate_from_images_no_candidates(self):
        """Test image generation when API returns no candidates."""
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}):
            with patch("nano_api.generate.genai.Client") as mock_client_class:
                with patch("nano_api.generate.aiplatform.init"):
                    mock_client = MagicMock()
                    mock_client_class.return_value = mock_client

                    # Mock response with no candidates
                    mock_response = MagicMock()
                    mock_response.candidates = []

                    mock_client.models.generate_content.return_value = mock_response
                    mock_client.files.upload.return_value = MagicMock()

                    client = GeminiClient()

                    # Create temporary test files
                    with tempfile.TemporaryDirectory() as temp_dir:
                        test_file = Path(temp_dir) / "test.png"
                        test_file.write_bytes(b"test image data")

                        with pytest.raises(ImageGenerationError, match="Image generation failed"):
                            client.generate_from_images("test prompt", [test_file])

    def test_generate_hires_image_without_scale(self):
        """Test high-res image generation without upscaling."""
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}):
            with patch("nano_api.generate.genai.Client") as mock_client_class:
                with patch("nano_api.generate.aiplatform.init"):
                    mock_client = MagicMock()
                    mock_client_class.return_value = mock_client

                    # Mock successful generation
                    with patch.object(GeminiClient, "generate_from_images") as mock_generate:
                        mock_generate.return_value = Path("generated_image.png")

                        client = GeminiClient()

                        # Create temporary test files
                        with tempfile.TemporaryDirectory() as temp_dir:
                            test_file = Path(temp_dir) / "test.png"
                            test_file.write_bytes(b"test image data")

                            result = client.generate_hires_image_in_one_shot(
                                "test prompt", [test_file]
                            )

                            assert result == Path("generated_image.png")

    def test_generate_hires_image_with_scale(self):
        """Test high-res image generation with upscaling."""
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}):
            with patch("nano_api.generate.genai.Client") as mock_client_class:
                with patch("nano_api.generate.aiplatform.init"):
                    mock_client = MagicMock()
                    mock_client_class.return_value = mock_client

                    # Mock successful generation and upscaling
                    with patch.object(GeminiClient, "generate_from_images") as mock_generate:
                        with patch("nano_api.generate.upscale_image") as mock_upscale:
                            mock_generate.return_value = Path("generated_image.png")
                            mock_upscaled_image = MagicMock()
                            mock_upscale.return_value = mock_upscaled_image

                            client = GeminiClient()

                            # Create temporary test files
                            with tempfile.TemporaryDirectory() as temp_dir:
                                test_file = Path(temp_dir) / "test.png"
                                test_file.write_bytes(b"test image data")

                                result = client.generate_hires_image_in_one_shot(
                                    "test prompt", [test_file], scale=4
                                )

                                expected_result = Path("./upscaled_generated_image.png")
                                assert result == expected_result
                                mock_upscale.assert_called_once()
                                mock_upscaled_image.save.assert_called_once()


class TestSaveResponseImage:
    """Test cases for save_response_image function."""
    def test_save_response_image_success(self):
        """Test saving image from API response."""
        from google.genai import types

        # Mock response with image data
        mock_response = MagicMock(spec=types.GenerateContentResponse)
        mock_candidate = MagicMock()
        mock_part = MagicMock()
        mock_part.text = None
        mock_part.inline_data = MagicMock()
        mock_part.inline_data.data = b"fake_image_data"

        mock_candidate.content.parts = [mock_part]
        mock_response.candidates = [mock_candidate]

        # Mock PIL Image operations
        with patch("nano_api.generate.Image.open") as mock_image_open:
            mock_image = MagicMock()
            mock_image_open.return_value = mock_image

            with patch("nano_api.utils.get_current_timestamp") as mock_timestamp:
                mock_timestamp.return_value = "2024-01-01-12:00:00"

                with tempfile.TemporaryDirectory() as temp_dir:
                    output_dir = Path(temp_dir)

                    result = save_response_image(mock_response, output_dir)

                    expected_result = output_dir / "generated_2024-01-01-12:00:00.png"
                    assert result == expected_result
                    mock_image.save.assert_called_once()

    def test_save_response_image_text_only(self):
        """Test saving response with text only (no image)."""
        from google.genai import types

        # Mock response with text only
        mock_response = MagicMock(spec=types.GenerateContentResponse)
        mock_candidate = MagicMock()
        mock_part = MagicMock()
        mock_part.text = "Some text response"
        mock_part.inline_data = None

        mock_candidate.content.parts = [mock_part]
        mock_response.candidates = [mock_candidate]

        result = save_response_image(mock_response)
        assert result is None

    def test_save_response_image_no_parts(self):
        """Test saving response with no content parts."""
        from google.genai import types

        # Mock response with no parts
        mock_response = MagicMock(spec=types.GenerateContentResponse)
        mock_candidate = MagicMock()
        mock_candidate.content.parts = []
        mock_response.candidates = [mock_candidate]

        with pytest.raises(ImageGenerationError, match="No content parts in the candidate"):
            save_response_image(mock_response)


class TestParseCommandLine:
    """Test cases for command line argument parsing."""
    def test_parse_command_line_defaults(self):
        """Test command line parsing with default values."""
        with patch("sys.argv", ["generate.py"]):
            args = parse_command_line()
            assert args.prompt is None
            assert args.output == Path("generated_gemini_image.png")
            assert args.image is None
            assert args.project_id is None
            assert args.location is None
            assert args.scale is None
            assert args.output_dir == Path(".")

    def test_parse_command_line_all_args(self):
        """Test command line parsing with all arguments provided."""
        test_args = [
            "generate.py",
            "--prompt", "test prompt",
            "--output", "custom_output.png",
            "--image", "image1.png",
            "--image", "image2.png",
            "--project-id", "test-project",
            "--location", "test-location",
            "--scale", "4",
            "--output-dir", "/custom/output"
        ]

        with patch("sys.argv", test_args):
            args = parse_command_line()
            assert args.prompt == "test prompt"
            assert args.output == Path("custom_output.png")
            assert args.image == [Path("image1.png"), Path("image2.png")]
            assert args.project_id == "test-project"
            assert args.location == "test-location"
            assert args.scale == 4
            assert args.output_dir == Path("/custom/output")

    def test_parse_command_line_scale_validation(self):
        """Test command line parsing with invalid scale value."""
        with patch("sys.argv", ["generate.py", "--scale", "3"]):
            with pytest.raises(SystemExit):  # argparse raises SystemExit on invalid choice
                parse_command_line()


class TestGenerateFromImagesFunction:
    """Test cases for the standalone generate_from_images function."""
    def test_generate_from_images_function(self):
        """Test the standalone generate_from_images function."""
        with patch("nano_api.generate.GeminiClient") as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client
            mock_client.generate_from_images.return_value = Path("test_result.png")

            from nano_api.generate import GenerationConfig
            config = GenerationConfig(
                project_id="test-project",
                location="test-location",
                output_dir=Path("./test_output")
            )
            result = generate_from_images(
                "test prompt",
                [Path("test_image.png")],
                config=config
            )

            mock_client_class.assert_called_once_with(
                project_id="test-project",
                location="test-location",
                output_dir=Path("./test_output"),
                storage_type=None
            )
            mock_client.generate_from_images.assert_called_once_with(
                "test prompt", [Path("test_image.png")]
            )
            assert result == Path("test_result.png")

    def test_generate_from_images_function_defaults(self):
        """Test the standalone generate_from_images function with defaults."""
        with patch("nano_api.generate.GeminiClient") as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client
            mock_client.generate_from_images.return_value = "test_result.png"

            with tempfile.TemporaryDirectory() as temp_dir:
                custom_output_dir = Path(temp_dir) / "custom"

                from nano_api.generate import GenerationConfig
                config = GenerationConfig(output_dir=custom_output_dir)
                result = generate_from_images(
                    "test prompt",
                    [Path("test_image.png")],
                    config=config
                )

                mock_client_class.assert_called_once_with(
                    project_id=DEFAULT_PROJECT_ID,
                    location=DEFAULT_LOCATION,
                    output_dir=custom_output_dir,
                    storage_type=None
                )
            assert result == "test_result.png"
