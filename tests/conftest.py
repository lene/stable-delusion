"""
Pytest configuration and shared fixtures for the test suite.
"""

import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Add the nano_api package to the Python path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "nano_api"))


# Environment variable fixtures


@pytest.fixture
def base_env():
    return {"GEMINI_API_KEY": "test-key", "STORAGE_TYPE": "local"}


@pytest.fixture
def full_env():
    return {
        "GEMINI_API_KEY": "test-key",
        "GCP_PROJECT_ID": "test-project",
        "GCP_LOCATION": "us-central1",
        "UPLOAD_FOLDER": "test_uploads",
        "DEFAULT_OUTPUT_DIR": "test_output",
        "FLASK_DEBUG": "false",
        "STORAGE_TYPE": "local",
    }


@pytest.fixture
def s3_env():
    return {
        "GEMINI_API_KEY": "test-key",
        "STORAGE_TYPE": "s3",
        "AWS_S3_BUCKET": "test-bucket",
        "AWS_S3_REGION": "us-east-1",
        "AWS_ACCESS_KEY_ID": "test-access-key",
        "AWS_SECRET_ACCESS_KEY": "test-secret-key",
    }


@pytest.fixture
def mock_env(request):
    # Get env vars from test parameter or use base_env as default
    env_vars = getattr(request, "param", {"GEMINI_API_KEY": "test-key", "STORAGE_TYPE": "local"})
    with patch.dict(os.environ, env_vars, clear=True):
        yield env_vars


@pytest.fixture(autouse=True)
def reset_config_manager():
    from nano_api.config import ConfigManager

    # Patch load_dotenv to prevent .env file loading during tests
    with patch("nano_api.config.load_dotenv"):
        ConfigManager.reset_config()
        yield
        ConfigManager.reset_config()


@pytest.fixture(scope="session")
def test_env_vars():
    with patch.dict(
        os.environ,
        {
            "GEMINI_API_KEY": "test-api-key-12345",
        },
        clear=False,
    ):
        yield


@pytest.fixture
def mock_gemini_response():
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
def mock_gemini_setup():
    with patch("nano_api.generate.genai.Client") as mock_client_class:
        with patch("nano_api.generate.aiplatform.init") as mock_init:
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client

            # Standard file upload mock
            mock_uploaded_file = MagicMock()
            mock_uploaded_file.name = "test_file"
            mock_uploaded_file.mime_type = "image/png"
            mock_uploaded_file.size_bytes = 1024
            mock_uploaded_file.uri = "test_uri"
            from datetime import datetime

            mock_uploaded_file.create_time = datetime.now()
            mock_uploaded_file.expiration_time = datetime.now()
            mock_client.files.upload.return_value = mock_uploaded_file

            # Configure generate_content with default response
            mock_client.models.generate_content.return_value = create_mock_gemini_response()

            yield {
                "client_class": mock_client_class,
                "client": mock_client,
                "init": mock_init,
                "uploaded_file": mock_uploaded_file,
            }


@pytest.fixture
def mock_gemini_client():
    with patch("nano_api.generate.genai.Client") as mock_client_class:
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        # Configure the mock client
        mock_client.models.generate_content.return_value = MagicMock()
        mock_client.files.upload.return_value = MagicMock()

        yield mock_client


@pytest.fixture
def mock_aiplatform_init():
    with patch("nano_api.generate.aiplatform.init") as mock_init:
        yield mock_init


@pytest.fixture
def mock_upscale_function():
    with patch("nano_api.generate.upscale_image") as mock_upscale:
        mock_upscaled_image = MagicMock()
        mock_upscaled_image.save.return_value = None
        mock_upscale.return_value = mock_upscaled_image
        yield mock_upscale


@pytest.fixture
def mock_main_gemini_service():
    with patch(
        "nano_api.main.ServiceFactory.create_image_generation_service"
    ) as mock_service_create:
        mock_service = MagicMock()
        mock_service_create.return_value = mock_service

        def create_mock_response(request_dto):
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
                "message": "Image generated successfully",
            }
            return mock_response

        mock_service.generate_image.side_effect = create_mock_response
        yield mock_service


@pytest.fixture
def mock_generate_gemini_client():
    with patch("nano_api.generate.genai.Client") as mock_client_class:
        with patch("nano_api.generate.aiplatform.init"):
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client
            mock_client.files.upload.return_value = MagicMock()
            yield mock_client


@pytest.fixture
def temp_image_file():
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
        # Create a minimal valid PNG file (1x1 pixel, white)
        png_data = (
            b"\x89PNG\r\n\x1a\n"  # PNG signature
            b"\x00\x00\x00\rIHDR"  # IHDR chunk
            b"\x00\x00\x00\x01"  # Width: 1
            b"\x00\x00\x00\x01"  # Height: 1
            b"\x08\x02\x00\x00\x00"  # Bit depth: 8, Color type: 2 (RGB), etc.
            b"\x90wS\xde"  # IHDR CRC
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
    with patch("nano_api.generate.Image.open") as mock_open:
        mock_image = MagicMock()
        mock_open.return_value = mock_image
        yield mock_image


@pytest.fixture
def mock_timestamp():
    with patch("nano_api.utils.get_current_timestamp") as mock_ts:
        mock_ts.return_value = "2024-01-01-12:00:00"
        yield mock_ts


@pytest.fixture
def custom_mock_timestamp():
    def _mock_timestamp(timestamp="2024-01-01-12:00:00"):
        return patch("nano_api.utils.get_current_timestamp", return_value=timestamp)

    return _mock_timestamp


@pytest.fixture
def mock_datetime():
    with patch("nano_api.generate.datetime") as mock_dt:
        mock_dt.now.return_value.strftime.return_value = "2024-01-01-12:00:00"
        yield mock_dt


@pytest.fixture
def temp_upload_dir():
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


@pytest.fixture
def flask_test_client():
    from nano_api.main import app

    with tempfile.TemporaryDirectory() as temp_dir:
        app.config["TESTING"] = True
        app.config["UPLOAD_FOLDER"] = temp_dir

        with app.test_client() as client:
            yield client


@pytest.fixture
def sample_test_args():
    return {
        "prompt": "A beautiful sunset over mountains",
        "output": "test_output.png",
        "image": ["image1.png", "image2.png"],
        "project_id": "test-project-123",
        "location": "us-west1",
        "scale": 4,
    }


@pytest.fixture
def mock_file_operations():
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
    with patch("nano_api.generate.logging") as mock_log:
        yield mock_log


@pytest.fixture
def mock_base64_operations():
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
    with patch("nano_api.upscale.default") as mock_default:
        mock_credentials = MagicMock()
        # NOTE: This is a test-only mock token, not a real credential
        mock_credentials.token = "mock-access-token"  # nosec B105
        mock_default.return_value = (mock_credentials, None)

        yield mock_credentials


# Helper functions for reducing code duplication
def create_mock_gemini_response(image_data=b"fake_generated_image_data", finish_reason="STOP"):
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


def assert_flask_response(
    response,
    expected_status=200,
    expected_success=True,
    required_fields=None,
    expected_message=None,
):
    assert response.status_code == expected_status
    response_data = json.loads(response.data)

    if expected_success is not None:
        assert response_data.get("success") is expected_success

    if expected_message:
        assert response_data.get("message") == expected_message

    if required_fields:
        for field in required_fields:
            assert field in response_data, f"Field '{field}' missing from response"

    return response_data


def assert_successful_flask_response(response, expected_message="Image generated successfully"):
    return assert_flask_response(response, expected_message=expected_message)


# Factory functions for test data creation
def create_mock_file_storage(
    content=b"fake image data", filename="test_image.png", content_type="image/png"
):
    from werkzeug.datastructures import FileStorage
    from io import BytesIO

    return FileStorage(stream=BytesIO(content), filename=filename, content_type=content_type)


@pytest.fixture
def mock_image_file():
    return create_mock_file_storage()


@pytest.fixture
def mock_image_files():
    return [
        create_mock_file_storage(b"fake image data 1", "test1.png"),
        create_mock_file_storage(b"fake image data 2", "test2.png"),
    ]


@pytest.fixture
def malicious_mock_file():
    return create_mock_file_storage(filename="../../../malicious.png", content_type="image/png")


def create_test_png_data():
    return (
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
        b"\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00"
        b"\x00\x0cIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01\r\n-\xdb"
        b"\x00\x00\x00\x00IEND\xaeB`\x82"
    )


def mock_image_operations():
    return patch("nano_api.generate.Image.open"), patch("nano_api.generate.datetime")


# Test configuration
def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "integration: marks tests as integration tests (deselect with '-m \"not integration\"')",
    )
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line("markers", "api: marks tests that make actual API calls")


# Test collection hooks
def pytest_collection_modifyitems(config, items):
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


# =============================================================================
# SEEDREAM S3 INTEGRATION TEST FIXTURES
# =============================================================================


@pytest.fixture
def mock_s3_client():
    mock_client = MagicMock()

    # Mock successful S3 operations
    mock_client.put_object.return_value = {}
    mock_client.head_object.return_value = {"ContentLength": 1024}
    mock_client.get_object.return_value = {"Body": MagicMock(), "ContentType": "image/jpeg"}
    mock_client.exceptions = MagicMock()
    mock_client.exceptions.NoSuchKey = Exception
    mock_client.exceptions.ClientError = Exception

    return mock_client


@pytest.fixture
def mock_seedream_client():
    from nano_api.seedream import SeedreamClient

    mock_client = MagicMock(spec=SeedreamClient)

    # Mock successful API response
    mock_client.generate_image.return_value = [
        "https://generated-image.com/result1.jpg",
        "https://generated-image.com/result2.jpg",
    ]

    # Mock successful generation and save
    mock_client.generate_and_save.return_value = Path("/tmp/generated_image.png")

    # Mock download functionality
    mock_client.download_image.return_value = Path("/tmp/downloaded_image.png")

    return mock_client


@pytest.fixture
def mock_ark_client():
    mock_client = MagicMock()

    # Mock successful image generation response
    mock_response = MagicMock()
    mock_image_data = MagicMock()
    mock_image_data.url = "https://generated-image.com/result.jpg"
    mock_response.data = [mock_image_data]

    mock_client.images.generate.return_value = mock_response

    return mock_client


@pytest.fixture
def sample_s3_urls():
    return [
        "https://test-bucket.s3.us-east-1.amazonaws.com/images/image1.jpg",
        "https://test-bucket.s3.us-east-1.amazonaws.com/images/image2.jpg",
        "https://test-bucket.s3.eu-central-1.amazonaws.com/seedream/inputs/input.png",
    ]


@pytest.fixture
def sample_local_paths():
    return [
        Path("/tmp/local_image1.jpg"),
        Path("/tmp/local_image2.png"),
        Path("/home/user/pictures/vacation.jpg"),
    ]


@pytest.fixture
def mock_aws_credentials():
    return {
        "AWS_ACCESS_KEY_ID": "AKIATEST123456789",
        "AWS_SECRET_ACCESS_KEY": "test-secret-key-mock-12345",
        "AWS_S3_BUCKET": "test-nano-api-bucket",
        "AWS_S3_REGION": "us-east-1",
    }


@pytest.fixture
def seedream_env():
    return {
        "ARK_API_KEY": "test-seedream-api-key-12345",
        "GEMINI_API_KEY": "test-gemini-key",
        "STORAGE_TYPE": "s3",
        "AWS_S3_BUCKET": "test-seedream-bucket",
        "AWS_S3_REGION": "us-east-1",
        "AWS_ACCESS_KEY_ID": "test-access-key",
        "AWS_SECRET_ACCESS_KEY": "test-secret-key",
    }


@pytest.fixture
def mock_s3_repository():
    from nano_api.repositories.s3_image_repository import S3ImageRepository

    mock_repo = MagicMock(spec=S3ImageRepository)

    # Mock save_image to return HTTPS URL
    mock_repo.save_image.return_value = Path(
        "https://test-bucket.s3.us-east-1.amazonaws.com/images/uploaded.jpg"
    )

    # Mock other repository methods
    mock_repo.load_image.return_value = MagicMock()  # Mock PIL Image
    mock_repo.validate_image_file.return_value = True
    mock_repo.generate_image_path.return_value = Path(
        "https://test-bucket.s3.us-east-1.amazonaws.com/images/generated.jpg"
    )

    return mock_repo


@pytest.fixture
def mock_seedream_service():
    from nano_api.services.seedream_service import SeedreamImageGenerationService

    mock_service = MagicMock(spec=SeedreamImageGenerationService)

    # Mock upload functionality
    mock_service.upload_images_to_s3.return_value = [
        "https://test-bucket.s3.us-east-1.amazonaws.com/images/uploaded1.jpg",
        "https://test-bucket.s3.us-east-1.amazonaws.com/images/uploaded2.jpg",
    ]

    mock_service.upload_files.return_value = [
        "https://test-bucket.s3.us-east-1.amazonaws.com/images/uploaded1.jpg"
    ]

    # Mock generate_image to return successful response
    from nano_api.models.responses import GenerateImageResponse
    from nano_api.models.client_config import ImageGenerationConfig, GCPConfig

    def mock_generate_image(request):
        return GenerateImageResponse(
            image_config=ImageGenerationConfig(
                generated_file=Path("/tmp/generated.png"),
                prompt=request.prompt,
                scale=request.scale,
                saved_files=request.images,
                output_dir=request.output_dir or Path("/tmp"),
            ),
            gcp_config=GCPConfig(project_id=request.project_id, location=request.location),
        )

    mock_service.generate_image.side_effect = mock_generate_image

    return mock_service


@pytest.fixture
def mock_pil_image_for_s3():
    mock_image = MagicMock()

    # Mock image save operation
    mock_image.save = MagicMock()

    # Mock image properties
    mock_image.format = "JPEG"
    mock_image.size = (1024, 768)
    mock_image.mode = "RGB"

    return mock_image


@pytest.fixture
def mock_requests_for_seedream():
    with patch("requests.get") as mock_get:
        mock_response = MagicMock()
        mock_response.content = b"fake_downloaded_image_data"
        mock_response.status_code = 200
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        yield mock_get


@pytest.fixture
def mock_timestamped_filename():
    with patch("nano_api.utils.generate_timestamped_filename") as mock_timestamp:
        mock_timestamp.return_value = "seedream_generated_2025-09-27-12:34:56.png"
        yield mock_timestamp


@pytest.fixture
def seedream_test_config():
    from nano_api.config import Config

    config = MagicMock(spec=Config)
    config.s3_bucket = "test-seedream-bucket"
    config.s3_region = "us-east-1"
    config.aws_access_key_id = "test-access-key"
    config.aws_secret_access_key = "test-secret-key"
    config.storage_type = "s3"
    config.default_output_dir = Path("/tmp")

    return config


@pytest.fixture
def mock_boto3_for_seedream():
    with patch("boto3.client") as mock_boto3_client:
        mock_client = MagicMock()

        # Mock S3 client creation
        mock_boto3_client.return_value = mock_client

        # Mock S3 operations
        mock_client.put_object.return_value = {}
        mock_client.head_object.return_value = {"ContentLength": 1024}
        mock_client.get_object.return_value = {"Body": MagicMock()}

        # Mock exceptions
        mock_client.exceptions = MagicMock()
        mock_client.exceptions.NoSuchKey = Exception
        mock_client.exceptions.ClientError = Exception

        yield mock_client


@pytest.fixture
def mock_s3_client_manager():
    with patch("nano_api.repositories.s3_image_repository.S3ClientManager") as mock_manager:
        mock_manager.create_s3_client.return_value = MagicMock()
        mock_manager._validate_s3_access.return_value = None
        yield mock_manager


@pytest.fixture
def valid_seedream_request():
    from nano_api.models.requests import GenerateImageRequest

    return GenerateImageRequest(
        prompt="Edit this image to make it more colorful",
        images=[Path("/tmp/test_image.jpg")],
        model="seedream",
        storage_type="s3",
        project_id="test-project",
        location="us-central1",
        output_dir=Path("/tmp/output"),
        scale=None,
    )


@pytest.fixture
def invalid_seedream_request():
    from nano_api.models.requests import GenerateImageRequest

    # This should trigger validation error (Seedream + images + local storage)
    return lambda: GenerateImageRequest(
        prompt="Edit this image",
        images=[Path("/tmp/test_image.jpg")],
        model="seedream",
        storage_type="local",  # Invalid combination
    )


@pytest.fixture
def mock_dotenv_for_cli():
    with patch("nano_api.generate.load_dotenv") as mock_load_dotenv:
        yield mock_load_dotenv


# Helper functions for Seedream testing
def create_mock_seedream_response(urls=None, error=None):
    if urls is None:
        urls = ["https://generated-image.com/result.jpg"]

    mock_response = MagicMock()

    if error:
        mock_response.data = []
        mock_response.error = error
    else:
        mock_response.data = []
        for url in urls:
            mock_image = MagicMock()
            mock_image.url = url
            mock_response.data.append(mock_image)

    return mock_response


def create_mock_s3_error(error_code="NoSuchKey"):
    from botocore.exceptions import ClientError

    return ClientError(error_response={"Error": {"Code": error_code}}, operation_name="HeadObject")


@pytest.fixture
def mock_service_factory():
    with patch("nano_api.factories.service_factory.ServiceFactory") as mock_factory:
        # Configure different service creation methods
        mock_factory.create_image_generation_service.return_value = MagicMock()
        mock_factory.create_file_service.return_value = MagicMock()
        mock_factory.create_upscaling_service.return_value = MagicMock()

        yield mock_factory
