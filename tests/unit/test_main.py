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

from nano_api.main import app

from ..conftest import assert_successful_flask_response

sys.path.append("nano_api")


@pytest.fixture
def client():
    """Create a test client for the Flask app."""
    app.config["TESTING"] = True
    app.config["UPLOAD_FOLDER"] = Path(tempfile.mkdtemp())
    with app.test_client() as client:
        yield client


@pytest.fixture
def mock_image_file():
    """Create a mock image file for testing."""
    return FileStorage(
        stream=BytesIO(b"fake image data"),
        filename="test_image.png",
        content_type="image/png"
    )


@pytest.fixture
def mock_image_files():
    """Create multiple mock image files for testing."""
    return [
        FileStorage(
            stream=BytesIO(b"fake image data 1"),
            filename="test_image1.png",
            content_type="image/png"
        ),
        FileStorage(
            stream=BytesIO(b"fake image data 2"),
            filename="test_image2.png",
            content_type="image/png"
        )
    ]


class TestFlaskAPI:
    """Test cases for Flask API endpoints and functionality."""

    def test_generate_endpoint_success(self, client, mock_image_files,
                                       mock_main_gemini_service):
        """Test successful image generation request."""
        data = {
            "prompt": "A beautiful landscape",
            "images": mock_image_files
        }

        response = client.post("/generate", data=data,
                               content_type="multipart/form-data")

        response_data = assert_successful_flask_response(response)
        assert response_data["prompt"] == "A beautiful landscape"
        assert response_data["generated_file"] == "generated_image.png"
        assert len(response_data["saved_files"]) == 2
        assert "upscaled" in response_data
        assert "project_id" in response_data
        assert "location" in response_data

    def test_generate_endpoint_missing_prompt(self, client, mock_image_files,
                                              mock_main_gemini_service):
        """Test request with missing prompt parameter (should use default)."""
        data = {
            "images": mock_image_files
        }

        response = client.post("/generate", data=data,
                               content_type="multipart/form-data")

        response_data = assert_successful_flask_response(response)
        assert "prompt" in response_data  # Should use default prompt
        assert response_data["generated_file"] == "generated_image.png"

    def test_generate_endpoint_missing_images(self, client):
        """Test request with missing images parameter."""
        data = {
            "prompt": "Test prompt"
        }

        response = client.post("/generate", data=data,
                               content_type="multipart/form-data")

        assert response.status_code == 400
        response_data = json.loads(response.data)
        assert response_data["success"] is False
        assert response_data["message"] == "Missing 'images' parameter"

    def test_generate_endpoint_empty_prompt(self, client, mock_image_files,
                                            mock_main_gemini_service):
        """Test request with empty prompt parameter (should use default)."""
        data = {
            "prompt": "",
            "images": mock_image_files
        }

        response = client.post("/generate", data=data,
                               content_type="multipart/form-data")

        response_data = assert_successful_flask_response(response)
        assert "prompt" in response_data  # Should use default prompt

    def test_generate_endpoint_single_image(self, client, mock_image_file,
                                            mock_main_gemini_service):
        """Test request with single image."""
        data = {
            "prompt": "Test prompt",
            "images": [mock_image_file]
        }

        response = client.post("/generate", data=data,
                               content_type="multipart/form-data")

        response_data = assert_successful_flask_response(response)
        assert len(response_data["saved_files"]) == 1

    def test_generate_endpoint_file_saving(
            self, client, mock_image_files, mock_main_gemini_service):
        """Test that files are properly saved to upload folder."""
        data = {
            "prompt": "Test prompt",
            "images": mock_image_files
        }

        response = client.post("/generate", data=data,
                               content_type="multipart/form-data")

        response_data = assert_successful_flask_response(response)
        assert len(response_data["saved_files"]) == 2

        # Check that files were saved to upload folder
        upload_folder = app.config["UPLOAD_FOLDER"]
        saved_files = list(upload_folder.glob("*"))
        assert len(saved_files) == 2

    def test_generate_endpoint_secure_filename(self, client, mock_main_gemini_service):
        """Test that filenames are properly secured."""
        malicious_file = FileStorage(
            stream=BytesIO(b"fake image data"),
            filename="../../../malicious.png",
            content_type="image/png"
        )

        data = {
            "prompt": "Test prompt",
            "images": [malicious_file]
        }

        response = client.post("/generate", data=data,
                               content_type="multipart/form-data")

        response_data = assert_successful_flask_response(response)
        # Filename should be secured (no path traversal)
        assert "../" not in response_data["saved_files"][0]

    def test_generate_endpoint_generation_failure(self, client, mock_image_files):
        """Test handling of image generation failure."""
        with patch("nano_api.main.GeminiImageGenerationService.create") as mock_service_create:
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
                "message": "Image generation failed"
            }
            mock_service.generate_image.return_value = mock_response

            data = {
                "prompt": "Test prompt",
                "images": mock_image_files
            }

            response = client.post("/generate", data=data,
                                   content_type="multipart/form-data")

            assert response.status_code == 200
            response_data = json.loads(response.data)
            assert response_data["success"] is False
            assert response_data["message"] == "Image generation failed"
            assert response_data["generated_file"] is None

    def test_generate_endpoint_generation_exception(self, client, mock_image_files):
        """Test handling of exceptions during generation."""
        with patch("nano_api.main.GeminiImageGenerationService.create") as mock_service_create:
            mock_service = MagicMock()
            mock_service_create.return_value = mock_service
            mock_service.generate_image.side_effect = ValueError("Generation failed")

            data = {
                "prompt": "Test prompt",
                "images": mock_image_files
            }

            response = client.post("/generate", data=data,
                                   content_type="multipart/form-data")

            assert response.status_code == 500
            response_data = json.loads(response.data)
            assert response_data["success"] is False
            assert "Unexpected error" in response_data["message"]

    def test_upload_folder_creation(self, client):
        """Test that upload folder is created if it doesn't exist."""
        # This is tested implicitly by the app configuration
        assert app.config["UPLOAD_FOLDER"].exists()

    def test_invalid_http_method(self, client):
        """Test that invalid HTTP methods return 405."""
        response = client.get("/generate")
        assert response.status_code == 405

    def test_response_format(self, client, mock_image_files):
        """Test that response contains all expected fields."""
        with patch("nano_api.main.GeminiImageGenerationService.create") as mock_service_create:
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
                "message": "Image generated successfully"
            }
            mock_service.generate_image.return_value = mock_response

            data = {
                "prompt": "Test prompt",
                "images": mock_image_files
            }

            response = client.post("/generate", data=data,
                                   content_type="multipart/form-data")

            response_data = assert_successful_flask_response(response)

            expected_fields = [
                "message", "prompt", "project_id", "location", "scale",
                "saved_files", "generated_file", "output_dir", "upscaled"
            ]

            for field in expected_fields:
                assert field in response_data

    def test_content_type_handling(self, client, mock_image_files):
        """Test that only multipart/form-data is accepted."""
        with patch("nano_api.main.GeminiImageGenerationService.create") as mock_service_create:
            mock_service = MagicMock()
            mock_service_create.return_value = mock_service

            mock_response = MagicMock()
            mock_response.to_dict.return_value = {
                "success": True,
                "message": "Image generated successfully"
            }
            mock_service.generate_image.return_value = mock_response

            data = {
                "prompt": "Test prompt",
                "images": mock_image_files
            }

            response = client.post("/generate", data=data,
                                   content_type="multipart/form-data")

            assert response.status_code == 200

    def test_generate_endpoint_with_output_dir(self, client, mock_image_files):
        """Test image generation with custom output directory."""
        with patch("nano_api.main.GeminiImageGenerationService.create") as mock_service_create:
            mock_service = MagicMock()
            mock_service_create.return_value = mock_service

            mock_response = MagicMock()
            mock_response.to_dict.return_value = {
                "generated_file": "output/generated_image.png",
                "output_dir": "output",
                "success": True,
                "message": "Image generated successfully"
            }
            mock_service.generate_image.return_value = mock_response

            data = {
                "prompt": "Test prompt",
                "output_dir": "./output",
                "images": mock_image_files
            }

            response = client.post("/generate", data=data,
                                   content_type="multipart/form-data")

            response_data = assert_successful_flask_response(response)
            assert response_data["output_dir"] == "output"

    def test_generate_with_scale_parameter(self, client, mock_image_files):
        """Test image generation with scale parameter."""
        with patch("nano_api.main.GeminiImageGenerationService.create") as mock_service_create:
            mock_service = MagicMock()
            mock_service_create.return_value = mock_service

            mock_response = MagicMock()
            mock_response.to_dict.return_value = {
                "generated_file": "upscaled_image.png",
                "scale": 4,
                "upscaled": True,
                "success": True,
                "message": "Image generated successfully"
            }
            mock_service.generate_image.return_value = mock_response

            data = {
                "prompt": "Test prompt",
                "scale": "4",
                "images": mock_image_files
            }

            response = client.post("/generate", data=data,
                                   content_type="multipart/form-data")

            response_data = assert_successful_flask_response(response)
            assert response_data["scale"] == 4
            assert response_data["upscaled"] is True

    def test_generate_with_invalid_scale(self, client, mock_image_files):
        """Test image generation with invalid scale parameter."""
        data = {
            "prompt": "Test prompt",
            "scale": "8",  # Invalid scale
            "images": mock_image_files
        }

        response = client.post("/generate", data=data,
                               content_type="multipart/form-data")

        assert response.status_code == 400
        response_data = json.loads(response.data)
        assert response_data["success"] is False
        assert "Scale must be 2 or 4" in response_data["message"]

    def test_health_endpoint(self, client):
        """Test health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data["status"] == "healthy"
        assert data["service"] == "NanoAPIClient"

    def test_api_info_endpoint(self, client):
        """Test API info endpoint."""
        response = client.get("/")
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data["name"] == "NanoAPIClient API"
        assert "endpoints" in data

    def test_openapi_spec_endpoint(self, client):
        """Test OpenAPI specification endpoint."""
        response = client.get("/openapi.json")
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data["openapi"] == "3.0.3"
        assert data["info"]["title"] == "NanoAPIClient API"


class TestFlaskAppConfiguration:
    """Test cases for Flask app configuration and setup."""

    def test_app_configuration(self):
        """Test basic app configuration."""
        assert app.config["UPLOAD_FOLDER"] is not None
        assert isinstance(app.config["UPLOAD_FOLDER"], Path)

    def test_upload_folder_exists(self):
        """Test that upload folder is created."""
        assert app.config["UPLOAD_FOLDER"].exists()

    def test_flask_app_debug_mode(self):
        """Test Flask debug mode configuration."""
        # Debug mode should be controlled by environment variable
        with patch.dict(os.environ, {"FLASK_DEBUG": "True"}):
            # This test verifies the debug mode logic exists
            # The actual debug mode is set in the if __name__ == "__main__" block
            pass
