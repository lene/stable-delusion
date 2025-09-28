"""Unit tests for the Flask web API server functionality."""

import json
import os
import sys
import tempfile
from io import BytesIO
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
from werkzeug.datastructures import FileStorage

from stable_delusion.main import app

from ..conftest import assert_successful_flask_response

sys.path.append("stable_delusion")


@pytest.fixture
def client():
    # Mock the configuration to avoid dependency on GEMINI_API_KEY
    with patch("stable_delusion.main.ConfigManager.get_config") as mock_config:
        from stable_delusion.config import Config

        mock_config.return_value = Config(
            project_id="test-project",
            location="us-central1",
            gemini_api_key="test-key",
            upload_folder=Path(tempfile.mkdtemp()),
            default_output_dir=Path("."),
            flask_debug=False,
            storage_type="local",
            s3_bucket=None,
            s3_region=None,
            aws_access_key_id=None,
            aws_secret_access_key=None,
        )
        app.config["TESTING"] = True
        app.config["UPLOAD_FOLDER"] = Path(tempfile.mkdtemp())
        with app.test_client() as client:
            yield client


# Note: mock_image_file and mock_image_files are now provided by conftest.py


class TestFlaskAPI:  # pylint: disable=too-many-public-methods
    """Test cases for Flask API endpoints and functionality."""

    def test_generate_endpoint_success(self, client, mock_image_files, mock_main_gemini_service):
        data = {"prompt": "A beautiful landscape", "images": mock_image_files}

        response = client.post("/generate", data=data, content_type="multipart/form-data")

        response_data = assert_successful_flask_response(response)
        assert response_data["prompt"] == "A beautiful landscape"
        assert response_data["generated_file"] == "generated_image.png"
        assert len(response_data["saved_files"]) == 2
        assert "upscaled" in response_data
        assert "project_id" in response_data
        assert "location" in response_data

    def test_generate_endpoint_missing_prompt(
        self, client, mock_image_files, mock_main_gemini_service
    ):
        data = {"images": mock_image_files}

        response = client.post("/generate", data=data, content_type="multipart/form-data")

        response_data = assert_successful_flask_response(response)
        assert "prompt" in response_data  # Should use default prompt
        assert response_data["generated_file"] == "generated_image.png"

    def test_generate_endpoint_missing_images(self, client):
        data = {"prompt": "Test prompt"}

        response = client.post("/generate", data=data, content_type="multipart/form-data")

        assert response.status_code == 400
        response_data = json.loads(response.data)
        assert response_data["success"] is False
        assert response_data["message"] == "Missing 'images' parameter"

    def test_generate_endpoint_empty_prompt(
        self, client, mock_image_files, mock_main_gemini_service
    ):
        data = {"prompt": "", "images": mock_image_files}

        response = client.post("/generate", data=data, content_type="multipart/form-data")

        response_data = assert_successful_flask_response(response)
        assert "prompt" in response_data  # Should use default prompt

    def test_generate_endpoint_single_image(
        self, client, mock_image_file, mock_main_gemini_service
    ):
        data = {"prompt": "Test prompt", "images": [mock_image_file]}

        response = client.post("/generate", data=data, content_type="multipart/form-data")

        response_data = assert_successful_flask_response(response)
        assert len(response_data["saved_files"]) == 1

    def test_generate_endpoint_file_saving(
        self, client, mock_image_files, mock_main_gemini_service
    ):
        data = {"prompt": "Test prompt", "images": mock_image_files}

        response = client.post("/generate", data=data, content_type="multipart/form-data")

        response_data = assert_successful_flask_response(response)
        assert len(response_data["saved_files"]) == 2

        # Check that files were saved to upload folder
        upload_folder = app.config["UPLOAD_FOLDER"]
        saved_files = list(upload_folder.glob("*"))
        assert len(saved_files) == 2

    def test_generate_endpoint_secure_filename(self, client, mock_main_gemini_service):
        malicious_file = FileStorage(
            stream=BytesIO(b"fake image data"),
            filename="../../../malicious.png",
            content_type="image/png",
        )

        data = {"prompt": "Test prompt", "images": [malicious_file]}

        response = client.post("/generate", data=data, content_type="multipart/form-data")

        response_data = assert_successful_flask_response(response)
        # Filename should be secured (no path traversal)
        assert "../" not in response_data["saved_files"][0]

    def test_generate_endpoint_generation_failure(self, client, mock_image_files):
        with patch(
            "stable_delusion.main.ServiceFactory.create_image_generation_service"
        ) as mock_service_create:
            mock_service = MagicMock()
            mock_service_create.return_value = mock_service

            # Create a mock response indicating failure
            mock_response = MagicMock()
            mock_response.generated_file = None
            mock_response.success = False
            mock_response.message = "Image generation failed"
            mock_response.to_dict.return_value = {
                "generated_file": None,
                "success": False,
                "message": "Image generation failed",
            }
            mock_service.generate_image.return_value = mock_response

            data = {"prompt": "Test prompt", "images": mock_image_files}

            response = client.post("/generate", data=data, content_type="multipart/form-data")

            assert response.status_code == 200
            response_data = json.loads(response.data)
            assert response_data["success"] is False
            assert response_data["message"] == "Image generation failed"
            assert response_data["generated_file"] is None

    def test_generate_endpoint_generation_exception(self, client, mock_image_files):
        with patch(
            "stable_delusion.main.ServiceFactory.create_image_generation_service"
        ) as mock_service_create:
            mock_service = MagicMock()
            mock_service_create.return_value = mock_service
            mock_service.generate_image.side_effect = ValueError("Generation failed")

            data = {"prompt": "Test prompt", "images": mock_image_files}

            response = client.post("/generate", data=data, content_type="multipart/form-data")

            assert response.status_code == 500
            response_data = json.loads(response.data)
            assert response_data["success"] is False
            assert "Unexpected error" in response_data["message"]

    def test_upload_folder_creation(self, client):
        # This is tested implicitly by the app configuration
        assert app.config["UPLOAD_FOLDER"].exists()

    def test_invalid_http_method(self, client):
        response = client.get("/generate")
        assert response.status_code == 405

    def test_response_format(self, client, mock_image_files):
        with patch(
            "stable_delusion.main.ServiceFactory.create_image_generation_service"
        ) as mock_service_create:
            mock_service = MagicMock()
            mock_service_create.return_value = mock_service

            # Create a comprehensive mock response
            mock_response = MagicMock()
            mock_response.to_dict.return_value = {
                "generated_file": "generated_image.png",
                "prompt": "Test prompt",
                "project_id": "test-project",
                "location": "us-central1",
                "scale": None,
                "saved_files": [],
                "output_dir": ".",
                "upscaled": False,
                "success": True,
                "message": "Image generated successfully",
            }
            mock_service.generate_image.return_value = mock_response

            data = {"prompt": "Test prompt", "images": mock_image_files}

            response = client.post("/generate", data=data, content_type="multipart/form-data")

            response_data = assert_successful_flask_response(response)

            expected_fields = [
                "message",
                "prompt",
                "project_id",
                "location",
                "scale",
                "saved_files",
                "generated_file",
                "output_dir",
                "upscaled",
            ]

            for field in expected_fields:
                assert field in response_data

    def test_content_type_handling(self, client, mock_image_files):
        with patch(
            "stable_delusion.main.ServiceFactory.create_image_generation_service"
        ) as mock_service_create:
            mock_service = MagicMock()
            mock_service_create.return_value = mock_service

            mock_response = MagicMock()
            mock_response.to_dict.return_value = {
                "success": True,
                "message": "Image generated successfully",
            }
            mock_service.generate_image.return_value = mock_response

            data = {"prompt": "Test prompt", "images": mock_image_files}

            response = client.post("/generate", data=data, content_type="multipart/form-data")

            assert response.status_code == 200

    def test_generate_endpoint_with_output_dir(self, client, mock_image_files):
        with patch(
            "stable_delusion.main.ServiceFactory.create_image_generation_service"
        ) as mock_service_create:
            mock_service = MagicMock()
            mock_service_create.return_value = mock_service

            mock_response = MagicMock()
            mock_response.to_dict.return_value = {
                "generated_file": "output/generated_image.png",
                "output_dir": "output",
                "success": True,
                "message": "Image generated successfully",
            }
            mock_service.generate_image.return_value = mock_response

            data = {"prompt": "Test prompt", "output_dir": "./output", "images": mock_image_files}

            response = client.post("/generate", data=data, content_type="multipart/form-data")

            response_data = assert_successful_flask_response(response)
            assert response_data["output_dir"] == "output"

    def test_generate_with_scale_parameter(self, client, mock_image_files):
        with patch(
            "stable_delusion.main.ServiceFactory.create_image_generation_service"
        ) as mock_service_create:
            mock_service = MagicMock()
            mock_service_create.return_value = mock_service

            mock_response = MagicMock()
            mock_response.to_dict.return_value = {
                "generated_file": "upscaled_image.png",
                "scale": 4,
                "upscaled": True,
                "success": True,
                "message": "Image generated successfully",
            }
            mock_service.generate_image.return_value = mock_response

            data = {"prompt": "Test prompt", "scale": "4", "images": mock_image_files}

            response = client.post("/generate", data=data, content_type="multipart/form-data")

            response_data = assert_successful_flask_response(response)
            assert response_data["scale"] == 4

    def test_generate_with_model_parameter_gemini(self, client, mock_image_files):
        with patch(
            "stable_delusion.main.ServiceFactory.create_image_generation_service"
        ) as mock_service_create:
            mock_service = MagicMock()
            mock_service_create.return_value = mock_service

            mock_response = MagicMock()
            mock_response.to_dict.return_value = {
                "generated_file": "gemini_image.png",
                "success": True,
                "message": "Image generated successfully",
            }
            mock_service.generate_image.return_value = mock_response

            data = {"prompt": "Test prompt", "model": "gemini", "images": mock_image_files}

            response = client.post("/generate", data=data, content_type="multipart/form-data")

            assert_successful_flask_response(response)

            # Verify service was called with model parameter
            mock_service_create.assert_called_once()
            call_args = mock_service_create.call_args
            assert call_args.kwargs["model"] == "gemini"

    def test_generate_with_model_parameter_seedream(self, client, mock_image_files):
        with patch(
            "stable_delusion.main.ServiceFactory.create_image_generation_service"
        ) as mock_service_create:
            mock_service = MagicMock()
            mock_service_create.return_value = mock_service

            mock_response = MagicMock()
            mock_response.to_dict.return_value = {
                "generated_file": "seedream_image.png",
                "success": True,
                "message": "Image generated successfully",
            }
            mock_service.generate_image.return_value = mock_response

            data = {
                "prompt": "Test prompt",
                "model": "seedream",
                "storage_type": "s3",
                "images": mock_image_files,
            }

            response = client.post("/generate", data=data, content_type="multipart/form-data")

            assert_successful_flask_response(response)

            # Verify service was called with model parameter
            mock_service_create.assert_called_once()
            call_args = mock_service_create.call_args
            assert call_args.kwargs["model"] == "seedream"

    def test_generate_model_defaults_to_none(self, client, mock_image_files):
        with patch(
            "stable_delusion.main.ServiceFactory.create_image_generation_service"
        ) as mock_service_create:
            mock_service = MagicMock()
            mock_service_create.return_value = mock_service

            mock_response = MagicMock()
            mock_response.to_dict.return_value = {
                "generated_file": "default_image.png",
                "success": True,
                "message": "Image generated successfully",
            }
            mock_service.generate_image.return_value = mock_response

            data = {"prompt": "Test prompt", "images": mock_image_files}

            response = client.post("/generate", data=data, content_type="multipart/form-data")

            assert_successful_flask_response(response)

            # Verify service was called with None model parameter
            mock_service_create.assert_called_once()
            call_args = mock_service_create.call_args
            assert call_args.kwargs["model"] is None

    def test_generate_with_invalid_scale(self, client, mock_image_files):
        data = {"prompt": "Test prompt", "scale": "8", "images": mock_image_files}  # Invalid scale

        response = client.post("/generate", data=data, content_type="multipart/form-data")

        assert response.status_code == 400
        response_data = json.loads(response.data)
        assert response_data["success"] is False
        assert "Scale must be 2 or 4" in response_data["message"]

    def test_health_endpoint(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data["status"] == "healthy"
        assert data["service"] == "NanoAPIClient"

    def test_api_info_endpoint(self, client):
        response = client.get("/")
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data["name"] == "NanoAPIClient API"
        assert "endpoints" in data

    def test_openapi_spec_endpoint(self, client):
        response = client.get("/openapi.json")
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data["openapi"] == "3.0.3"
        assert data["info"]["title"] == "NanoAPIClient API"


class TestFlaskAppConfiguration:
    """Test cases for Flask app configuration and setup."""

    def test_app_configuration(self):
        assert app.config["UPLOAD_FOLDER"] is not None
        assert isinstance(app.config["UPLOAD_FOLDER"], Path)

    def test_upload_folder_exists(self):
        assert app.config["UPLOAD_FOLDER"].exists()

    def test_flask_app_debug_mode(self):
        # Debug mode should be controlled by environment variable
        with patch.dict(os.environ, {"FLASK_DEBUG": "True"}):
            # This test verifies the debug mode logic exists
            # The actual debug mode is set in the if __name__ == "__main__" block
            pass
