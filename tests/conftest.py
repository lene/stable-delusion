"""
Pytest configuration and shared fixtures for the test suite.
"""
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Add the nano_api package to the Python path for testing
sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "..", "nano_api")
)


@pytest.fixture(scope="session")
def test_env_vars():
    """Set up test environment variables for the entire test session."""
    with patch.dict(os.environ, {
        "GEMINI_API_KEY": "test-api-key-12345",
    }, clear=False):
        yield


@pytest.fixture
def mock_gemini_response():
    """Create a mock Gemini API response."""
    mock_response = MagicMock()
    mock_candidate = MagicMock()
    mock_part = MagicMock()

    # Configure the mock response structure
    mock_part.text = None
    mock_part.inline_data = MagicMock()
    mock_part.inline_data.data = b"fake_generated_image_data"

    mock_candidate.content.parts = [mock_part]
    mock_candidate.finish_reason = "STOP"

    mock_response.candidates = [mock_candidate]
    mock_response.usage_metadata = MagicMock()
    mock_response.usage_metadata.total_token_count = 100

    return mock_response


@pytest.fixture
def mock_gemini_client():
    """Create a mock Gemini client."""
    with patch("nano_api.generate.genai.Client") as mock_client_class:
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        # Configure the mock client
        mock_client.models.generate_content.return_value = MagicMock()
        mock_client.files.upload.return_value = MagicMock()

        yield mock_client


@pytest.fixture
def mock_aiplatform_init():
    """Mock the aiplatform.init call."""
    with patch("nano_api.generate.aiplatform.init") as mock_init:
        yield mock_init


@pytest.fixture
def mock_upscale_function():
    """Mock the upscale_image function."""
    with patch("nano_api.generate.upscale_image") as mock_upscale:
        mock_upscaled_image = MagicMock()
        mock_upscaled_image.save.return_value = None
        mock_upscale.return_value = mock_upscaled_image
        yield mock_upscale


@pytest.fixture
def mock_main_gemini_service():
    """Mock GeminiImageGenerationService for main.py Flask tests."""
    with patch("nano_api.main.ServiceFactory."
               "create_image_generation_service") as mock_service_create:
        mock_service = MagicMock()
        mock_service_create.return_value = mock_service

        def create_mock_response(request_dto):
            """Create a dynamic mock response based on request."""
            mock_response = MagicMock()
            mock_response.generated_file = Path("generated_image.png")
            mock_response.prompt = request_dto.prompt
            mock_response.project_id = request_dto.project_id
            mock_response.location = request_dto.location
            mock_response.scale = request_dto.scale
            mock_response.saved_files = request_dto.images  # Use actual saved files
            mock_response.output_dir = request_dto.output_dir
            mock_response.upscaled = request_dto.scale is not None
            mock_response.to_dict.return_value = {
                "generated_file": "generated_image.png",
                "prompt": request_dto.prompt,
                "project_id": request_dto.project_id,
                "location": request_dto.location,
                "scale": request_dto.scale,
                "saved_files": [str(f) for f in request_dto.images],
                "output_dir": str(request_dto.output_dir),
                "upscaled": request_dto.scale is not None,
                "success": True,
                "message": "Image generated successfully"
            }
            return mock_response

        mock_service.generate_image.side_effect = create_mock_response
        yield mock_service


@pytest.fixture
def mock_generate_gemini_client():
    """Mock GeminiClient for generate.py tests."""
    with patch("nano_api.generate.genai.Client") as mock_client_class:
        with patch("nano_api.generate.aiplatform.init"):
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client
            mock_client.files.upload.return_value = MagicMock()
            yield mock_client


@pytest.fixture
def temp_image_file():
    """Create a temporary test image file with valid PNG data."""
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
        # Create a minimal valid PNG file (1x1 pixel, white)
        png_data = (
            b"\x89PNG\r\n\x1a\n"  # PNG signature
            b"\x00\x00\x00\rIHDR"  # IHDR chunk
            b"\x00\x00\x00\x01"    # Width: 1
            b"\x00\x00\x00\x01"    # Height: 1
            b"\x08\x02\x00\x00\x00"  # Bit depth: 8, Color type: 2 (RGB), etc.
            b"\x90wS\xde"          # IHDR CRC
            b"\x00\x00\x00\x0cIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01\r\n-\xdb"  # IDAT
            b"\x00\x00\x00\x00IEND\xaeB`\x82"  # IEND chunk
        )
        temp_file.write(png_data)
        temp_file.flush()
        yield temp_file.name

    # Cleanup
    if os.path.exists(temp_file.name):
        os.unlink(temp_file.name)


@pytest.fixture
def temp_images(temp_image_file):
    """Create multiple temporary test image files."""
    files = [temp_image_file]  # Start with the first one

    # Create additional files
    for i in range(1, 3):  # Create 2 more files (total of 3)
        with tempfile.NamedTemporaryFile(suffix=f"_test_{i}.png", delete=False) as temp_file:
            png_data = (
                b"\x89PNG\r\n\x1a\n"
                b"\x00\x00\x00\rIHDR"
                b"\x00\x00\x00\x01"
                b"\x00\x00\x00\x01"
                b"\x08\x02\x00\x00\x00"
                b"\x90wS\xde"
                b"\x00\x00\x00\x0cIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01\r\n-\xdb"
                b"\x00\x00\x00\x00IEND\xaeB`\x82"
            )
            temp_file.write(png_data)
            temp_file.flush()
            files.append(temp_file.name)

    yield files

    # Cleanup (except the first one, which is cleaned by temp_image_file)
    for file_path in files[1:]:
        if os.path.exists(file_path):
            os.unlink(file_path)


@pytest.fixture
def mock_pil_image():
    """Mock PIL Image operations."""
    with patch("nano_api.generate.Image.open") as mock_open:
        mock_image = MagicMock()
        mock_open.return_value = mock_image
        yield mock_image


@pytest.fixture
def mock_datetime():
    """Mock datetime for predictable timestamps."""
    with patch("nano_api.generate.datetime") as mock_dt:
        mock_dt.now.return_value.strftime.return_value = "2024-01-01-12:00:00"
        yield mock_dt


@pytest.fixture
def temp_upload_dir():
    """Create a temporary upload directory."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


@pytest.fixture
def flask_test_client():
    """Create a Flask test client with temporary upload directory."""
    from nano_api.main import app

    with tempfile.TemporaryDirectory() as temp_dir:
        app.config["TESTING"] = True
        app.config["UPLOAD_FOLDER"] = temp_dir

        with app.test_client() as client:
            yield client


@pytest.fixture
def sample_test_args():
    """Provide sample command line arguments for testing."""
    return {
        "prompt": "A beautiful sunset over mountains",
        "output": "test_output.png",
        "image": ["image1.png", "image2.png"],
        "project_id": "test-project-123",
        "location": "us-west1",
        "scale": 4
    }


@pytest.fixture
def mock_file_operations():
    """Mock common file operations."""
    mocks = {}

    with patch("builtins.open", create=True) as mock_open:
        mocks["open"] = mock_open

        with patch("os.path.exists") as mock_exists:
            mocks["exists"] = mock_exists
            mock_exists.return_value = True

            with patch("os.path.isfile") as mock_isfile:
                mocks["isfile"] = mock_isfile
                mock_isfile.return_value = True

                with patch("os.makedirs") as mock_makedirs:
                    mocks["makedirs"] = mock_makedirs

                    yield mocks


@pytest.fixture
def mock_logging():
    """Mock logging functions."""
    with patch("nano_api.generate.logging") as mock_log:
        yield mock_log


@pytest.fixture
def mock_base64_operations():
    """Mock base64 encoding/decoding operations."""
    mocks = {}

    with patch("nano_api.upscale.base64.b64encode") as mock_encode:
        mocks["encode"] = mock_encode
        mock_encode.return_value = b"bW9ja19lbmNvZGVkX2RhdGE="

        with patch("nano_api.upscale.base64.b64decode") as mock_decode:
            mocks["decode"] = mock_decode
            mock_decode.return_value = b"mock_decoded_image_data"

            yield mocks


@pytest.fixture
def mock_requests():
    """Mock requests library for API calls."""
    with patch("nano_api.upscale.requests.post") as mock_post:
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "predictions": [{"bytesBase64Encoded": "bW9ja19yZXNwb25zZQ=="}]
        }
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        yield mock_post


@pytest.fixture
def mock_google_auth():
    """Mock Google authentication."""
    with patch("nano_api.upscale.default") as mock_default:
        mock_credentials = MagicMock()
        # NOTE: This is a test-only mock token, not a real credential
        mock_credentials.token = "mock-access-token"  # nosec B105
        mock_default.return_value = (mock_credentials, None)

        yield mock_credentials


# Helper functions for reducing code duplication
def create_mock_gemini_response(image_data=b"fake_generated_image_data", finish_reason="STOP"):
    """Create a mock Gemini API response with consistent structure."""
    from unittest.mock import MagicMock

    mock_response = MagicMock()
    mock_candidate = MagicMock()
    mock_part = MagicMock()
    mock_part.text = None
    mock_part.inline_data = MagicMock()
    mock_part.inline_data.data = image_data

    mock_candidate.content.parts = [mock_part]
    mock_candidate.finish_reason = finish_reason
    mock_response.candidates = [mock_candidate]
    mock_response.usage_metadata.total_token_count = 100

    return mock_response


def assert_successful_flask_response(response, expected_message="Image generated successfully"):
    """Assert common Flask API response patterns."""
    import json

    assert response.status_code == 200
    response_data = json.loads(response.data)
    assert response_data["message"] == expected_message
    return response_data


def mock_image_operations():
    """Create mock context for PIL Image and datetime operations."""
    from unittest.mock import patch

    return patch("nano_api.generate.Image.open"), patch("nano_api.generate.datetime")


# Test configuration
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers",
        "integration: marks tests as integration tests (deselect with '-m \"not integration\"')"
    )
    config.addinivalue_line(
        "markers",
        "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "api: marks tests that make actual API calls"
    )


# Test collection hooks
def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on test names."""
    for item in items:
        # Mark integration tests
        if "integration" in item.nodeid.lower():
            item.add_marker(pytest.mark.integration)

        # Mark API tests
        if "api" in item.name.lower() or "endpoint" in item.name.lower():
            item.add_marker(pytest.mark.api)

        # Mark slow tests
        if "large" in item.name.lower() or "performance" in item.name.lower():
            item.add_marker(pytest.mark.slow)
