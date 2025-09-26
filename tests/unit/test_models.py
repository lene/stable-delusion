"""
Unit tests for DTOs and model classes.
Tests request and response models with validation.
"""

import tempfile
from pathlib import Path

import pytest

from nano_api.exceptions import ValidationError
from nano_api.models.requests import (GenerateImageRequest, UpscaleImageRequest)
from nano_api.models.responses import (GenerateImageResponse, UpscaleImageResponse,
                                       HealthResponse, APIInfoResponse,
                                       ErrorResponse)
from nano_api.models.client_config import GCPConfig, ImageGenerationConfig


class TestGenerateImageRequest:
    """Test GenerateImageRequest DTO."""

    def test_valid_request(self):
        """Test creating a valid request."""
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "test.jpg"
            image_path.write_bytes(b"test data")

            request = GenerateImageRequest(
                prompt="Test prompt",
                images=[image_path],
                scale=2
            )

            assert request.prompt == "Test prompt"
            assert request.images == [image_path]
            assert request.scale == 2

    def test_empty_prompt(self):
        """Test validation fails for empty prompt."""
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "test.jpg"
            image_path.write_bytes(b"test data")

            with pytest.raises(ValidationError, match="Prompt cannot be empty"):
                GenerateImageRequest(
                    prompt="",
                    images=[image_path]
                )

    def test_no_images(self):
        """Test validation fails when no images provided."""
        with pytest.raises(ValidationError, match="At least one image is required"):
            GenerateImageRequest(
                prompt="Test prompt",
                images=[]
            )

    def test_invalid_scale(self):
        """Test validation fails for invalid scale."""
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "test.jpg"
            image_path.write_bytes(b"test data")

            with pytest.raises(ValidationError, match="Scale must be 2 or 4"):
                GenerateImageRequest(
                    prompt="Test prompt",
                    images=[image_path],
                    scale=3
                )

    def test_string_to_path_conversion(self):
        """Test automatic conversion of string to Path."""
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "test.jpg"
            image_path.write_bytes(b"test data")

            request = GenerateImageRequest(
                prompt="Test prompt",
                images=[image_path],
                output_dir=temp_dir  # Pass string instead of Path
            )

            assert isinstance(request.output_dir, Path)
            assert request.output_dir == Path(temp_dir)


class TestUpscaleImageRequest:
    """Test UpscaleImageRequest DTO."""

    def test_valid_request(self):
        """Test creating a valid upscale request."""
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "test.jpg"
            image_path.write_bytes(b"test data")

            request = UpscaleImageRequest(
                image_path=image_path,
                scale_factor="x4"
            )

            assert request.image_path == image_path
            assert request.scale_factor == "x4"

    def test_invalid_scale_factor(self):
        """Test validation fails for invalid scale factor."""
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "test.jpg"
            image_path.write_bytes(b"test data")

            with pytest.raises(ValidationError, match="Scale factor must be one of"):
                UpscaleImageRequest(
                    image_path=image_path,
                    scale_factor="x8"
                )

    def test_string_to_path_conversion(self):
        """Test automatic conversion of string to Path."""
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path_str = str(Path(temp_dir) / "test.jpg")

            request = UpscaleImageRequest(
                image_path=image_path_str,  # Pass string instead of Path
                scale_factor="x2"
            )

            assert isinstance(request.image_path, Path)
            assert request.image_path == Path(image_path_str)


class TestGenerateImageResponse:
    """Test GenerateImageResponse DTO."""

    def test_successful_response(self):
        """Test creating a successful response."""
        with tempfile.TemporaryDirectory() as temp_dir:
            generated_file = Path(temp_dir) / "generated.jpg"
            saved_files = [Path(temp_dir) / "input.jpg"]
            output_dir = Path(temp_dir)

            response = GenerateImageResponse(
                image_config=ImageGenerationConfig(
                    generated_file=generated_file,
                    prompt="Test prompt",
                    scale=2,
                    saved_files=saved_files,
                    output_dir=output_dir
                ),
                gcp_config=GCPConfig(
                    project_id="test-project",
                    location="us-central1"
                )
            )

            assert response.success is True
            assert response.message == "Image generated successfully"
            assert response.upscaled is True
            assert response.scale == 2

    def test_failed_response(self):
        """Test creating a failed response."""
        with tempfile.TemporaryDirectory() as temp_dir:
            saved_files = [Path(temp_dir) / "input.jpg"]
            output_dir = Path(temp_dir)

            response = GenerateImageResponse(
                image_config=ImageGenerationConfig(
                    generated_file=None,  # Failed generation
                    prompt="Test prompt",
                    scale=None,
                    saved_files=saved_files,
                    output_dir=output_dir
                ),
                gcp_config=GCPConfig(
                    project_id="test-project",
                    location="us-central1"
                )
            )

            assert response.success is False
            assert response.message == "Image generation failed"
            assert response.upscaled is False

    def test_to_dict_conversion(self):
        """Test dictionary conversion for JSON serialization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            generated_file = Path(temp_dir) / "generated.jpg"
            saved_files = [Path(temp_dir) / "input.jpg"]
            output_dir = Path(temp_dir)

            response = GenerateImageResponse(
                image_config=ImageGenerationConfig(
                    generated_file=generated_file,
                    prompt="Test prompt",
                    scale=2,
                    saved_files=saved_files,
                    output_dir=output_dir
                ),
                gcp_config=GCPConfig(
                    project_id="test-project",
                    location="us-central1"
                )
            )

            data = response.to_dict()

            # Check that Path objects are converted to strings
            assert isinstance(data["generated_file"], str)
            assert isinstance(data["saved_files"][0], str)
            assert isinstance(data["output_dir"], str)
            assert data["success"] is True


class TestUpscaleImageResponse:
    """Test UpscaleImageResponse DTO."""

    def test_successful_response(self):
        """Test creating a successful upscale response."""
        with tempfile.TemporaryDirectory() as temp_dir:
            original_file = Path(temp_dir) / "original.jpg"
            upscaled_file = Path(temp_dir) / "upscaled.jpg"

            response = UpscaleImageResponse(
                upscaled_file=upscaled_file,
                original_file=original_file,
                scale_factor="x4",
                gcp_config=GCPConfig(
                    project_id="test-project",
                    location="us-central1"
                )
            )

            assert response.success is True
            assert response.message == "Image upscaled successfully"
            assert response.scale_factor == "x4"

    def test_to_dict_conversion(self):
        """Test dictionary conversion for JSON serialization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            original_file = Path(temp_dir) / "original.jpg"
            upscaled_file = Path(temp_dir) / "upscaled.jpg"

            response = UpscaleImageResponse(
                upscaled_file=upscaled_file,
                original_file=original_file,
                scale_factor="x4",
                gcp_config=GCPConfig(
                    project_id="test-project",
                    location="us-central1"
                )
            )

            data = response.to_dict()

            # Check that Path objects are converted to strings
            assert isinstance(data["upscaled_file"], str)
            assert isinstance(data["original_file"], str)


class TestHealthResponse:
    """Test HealthResponse DTO."""

    def test_default_response(self):
        """Test creating a default health response."""
        response = HealthResponse()

        assert response.success is True
        assert response.service == "NanoAPIClient"
        assert response.version == "1.0.0"
        assert response.status == "healthy"

    def test_custom_response(self):
        """Test creating a custom health response."""
        response = HealthResponse(
            service="CustomService",
            version="2.0.0",
            status="degraded"
        )

        assert response.service == "CustomService"
        assert response.version == "2.0.0"
        assert response.status == "degraded"
        assert response.message == "Service degraded"


class TestAPIInfoResponse:
    """Test APIInfoResponse DTO."""

    def test_response_creation(self):
        """Test creating an API info response."""
        response = APIInfoResponse()

        assert response.success is True
        assert response.name == "NanoAPIClient API"
        assert response.description.startswith("Flask web API")
        assert response.version == "1.0.0"
        assert "/" in response.endpoints
        assert "/health" in response.endpoints


class TestErrorResponse:
    """Test ErrorResponse DTO."""

    def test_basic_error(self):
        """Test creating a basic error response."""
        response = ErrorResponse("Something went wrong")

        assert response.success is False
        assert response.message == "Something went wrong"
        assert response.error_code is None
        assert response.details is None

    def test_detailed_error(self):
        """Test creating a detailed error response."""
        response = ErrorResponse(
            "Validation failed",
            error_code="VALIDATION_ERROR",
            details="Field 'name' is required"
        )

        assert response.success is False
        assert response.message == "Validation failed"
        assert response.error_code == "VALIDATION_ERROR"
        assert response.details == "Field 'name' is required"
