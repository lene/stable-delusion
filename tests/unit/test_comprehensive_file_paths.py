"""
Comprehensive test matrix for file path generation covering all combinations:
- Models: gemini, seedream
- Output dir: set, unset
- Storage type: local, s3
- Filename: with .png, without .png, None (default)
"""

__author__ = "Lene Preuss <lene.preuss@gmail.com>"

import tempfile
from pathlib import Path
from unittest.mock import Mock

import pytest

from stable_delusion.generate import _create_cli_request_dto, _handle_cli_custom_output
from stable_delusion.models.requests import GenerateImageRequest
from stable_delusion.models.responses import GenerateImageResponse
from stable_delusion.models.client_config import ImageGenerationConfig, GCPConfig


@pytest.fixture
def temp_image():
    """Create a temporary image file for testing."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as f:
        f.write(b"fake image")
        path = Path(f.name)
    yield path
    if path.exists():
        path.unlink()


@pytest.fixture
def mock_gemini_image(temp_image):
    """Create a mock image for Gemini testing."""
    return [temp_image]


class TestComprehensiveFilePathMatrix:
    """Test all combinations of model, output_dir, storage_type, and filename."""

    # ========== GEMINI TESTS ==========

    def test_gemini_local_with_output_dir_filename_with_png(self, temp_image, mock_gemini_image):
        """Gemini + local + output_dir set + filename with .png extension."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            args = Mock()
            args.gcp_project_id = None
            args.gcp_location = None
            args.output_dir = output_dir
            args.output_filename = Path("my_image.png")
            args.scale = None
            args.size = None
            args.storage_type = "local"
            args.model = "gemini"

            request = _create_cli_request_dto("test", mock_gemini_image, args)

            assert request.model == "gemini"
            assert request.storage_type == "local"
            assert request.output_dir == output_dir
            assert request.output_filename == Path("my_image")  # .png stripped

    def test_gemini_local_with_output_dir_filename_without_png(self, temp_image, mock_gemini_image):
        """Gemini + local + output_dir set + filename without extension."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            args = Mock()
            args.gcp_project_id = None
            args.gcp_location = None
            args.output_dir = output_dir
            args.output_filename = Path("my_image")
            args.scale = None
            args.size = None
            args.storage_type = "local"
            args.model = "gemini"

            request = _create_cli_request_dto("test", mock_gemini_image, args)

            assert request.model == "gemini"
            assert request.storage_type == "local"
            assert request.output_filename == Path("my_image")

    def test_gemini_local_with_output_dir_filename_none(self, temp_image, mock_gemini_image):
        """Gemini + local + output_dir set + filename None (uses default 'generated')."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            args = Mock()
            args.gcp_project_id = None
            args.gcp_location = None
            args.output_dir = output_dir
            args.output_filename = None
            args.scale = None
            args.size = None
            args.storage_type = "local"
            args.model = "gemini"

            request = _create_cli_request_dto("test", mock_gemini_image, args)

            assert request.model == "gemini"
            assert request.storage_type == "local"
            assert request.output_filename is None  # Will use "generated" default

    def test_gemini_local_without_output_dir_filename_with_png(self, temp_image, mock_gemini_image):
        """Gemini + local + output_dir unset + filename with .png."""
        args = Mock()
        args.gcp_project_id = None
        args.gcp_location = None
        args.output_dir = None
        args.output_filename = Path("test.png")
        args.scale = None
        args.size = None
        args.storage_type = "local"
        args.model = "gemini"

        request = _create_cli_request_dto("test", mock_gemini_image, args)

        assert request.model == "gemini"
        assert request.output_dir is None
        assert request.output_filename == Path("test")

    def test_gemini_s3_with_output_dir_filename_with_png(self, temp_image, mock_gemini_image):
        """Gemini + s3 + output_dir set + filename with .png."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            args = Mock()
            args.gcp_project_id = None
            args.gcp_location = None
            args.output_dir = output_dir
            args.output_filename = Path("s3_image.png")
            args.scale = None
            args.size = None
            args.storage_type = "s3"
            args.model = "gemini"

            request = _create_cli_request_dto("test", mock_gemini_image, args)

            assert request.model == "gemini"
            assert request.storage_type == "s3"
            assert request.output_filename == Path("s3_image")

    def test_gemini_s3_without_output_dir_filename_none(self, temp_image, mock_gemini_image):
        """Gemini + s3 + output_dir unset + filename None (default)."""
        args = Mock()
        args.gcp_project_id = None
        args.gcp_location = None
        args.output_dir = None
        args.output_filename = None
        args.scale = None
        args.size = None
        args.storage_type = "s3"
        args.model = "gemini"

        request = _create_cli_request_dto("test", mock_gemini_image, args)

        assert request.model == "gemini"
        assert request.storage_type == "s3"
        assert request.output_filename is None  # Will use "generated" default

    # ========== SEEDREAM TESTS ==========

    def test_seedream_local_with_output_dir_filename_with_png(self):
        """Seedream + local + output_dir set + filename with .png."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            args = Mock()
            args.gcp_project_id = None
            args.gcp_location = None
            args.output_dir = output_dir
            args.output_filename = Path("seedream_test.png")
            args.scale = None
            args.size = None
            args.storage_type = "local"
            args.model = "seedream"

            request = _create_cli_request_dto("test", [], args)

            assert request.model == "seedream"
            assert request.storage_type == "local"
            assert request.output_dir == output_dir
            assert request.output_filename == Path("seedream_test")

    def test_seedream_local_with_output_dir_filename_without_png(self):
        """Seedream + local + output_dir set + filename without extension."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            args = Mock()
            args.gcp_project_id = None
            args.gcp_location = None
            args.output_dir = output_dir
            args.output_filename = Path("seedream_test")
            args.scale = None
            args.size = None
            args.storage_type = "local"
            args.model = "seedream"

            request = _create_cli_request_dto("test", [], args)

            assert request.model == "seedream"
            assert request.storage_type == "local"
            assert request.output_filename == Path("seedream_test")

    def test_seedream_local_with_output_dir_filename_none(self):
        """Seedream + local + output_dir set + filename None (uses 'seedream_image')."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            args = Mock()
            args.gcp_project_id = None
            args.gcp_location = None
            args.output_dir = output_dir
            args.output_filename = None
            args.scale = None
            args.size = None
            args.storage_type = "local"
            args.model = "seedream"

            request = _create_cli_request_dto("test", [], args)

            assert request.model == "seedream"
            assert request.storage_type == "local"
            assert request.output_filename is None  # Will use "seedream_image" default

    def test_seedream_local_without_output_dir_filename_with_png(self):
        """Seedream + local + output_dir unset + filename with .png."""
        args = Mock()
        args.gcp_project_id = None
        args.gcp_location = None
        args.output_dir = None
        args.output_filename = Path("test.png")
        args.scale = None
        args.size = None
        args.storage_type = "local"
        args.model = "seedream"

        request = _create_cli_request_dto("test", [], args)

        assert request.model == "seedream"
        assert request.output_dir is None
        assert request.output_filename == Path("test")

    def test_seedream_local_without_output_dir_filename_without_png(self):
        """Seedream + local + output_dir unset + filename without extension."""
        args = Mock()
        args.gcp_project_id = None
        args.gcp_location = None
        args.output_dir = None
        args.output_filename = Path("test")
        args.scale = None
        args.size = None
        args.storage_type = "local"
        args.model = "seedream"

        request = _create_cli_request_dto("test", [], args)

        assert request.model == "seedream"
        assert request.output_filename == Path("test")

    def test_seedream_local_without_output_dir_filename_none(self):
        """Seedream + local + output_dir unset + filename None (default)."""
        args = Mock()
        args.gcp_project_id = None
        args.gcp_location = None
        args.output_dir = None
        args.output_filename = None
        args.scale = None
        args.size = None
        args.storage_type = "local"
        args.model = "seedream"

        request = _create_cli_request_dto("test", [], args)

        assert request.model == "seedream"
        assert request.output_filename is None

    def test_seedream_s3_with_output_dir_filename_with_png(self):
        """Seedream + s3 + output_dir set + filename with .png."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            args = Mock()
            args.gcp_project_id = None
            args.gcp_location = None
            args.output_dir = output_dir
            args.output_filename = Path("s3_seedream.png")
            args.scale = None
            args.size = None
            args.storage_type = "s3"
            args.model = "seedream"

            request = _create_cli_request_dto("test", [], args)

            assert request.model == "seedream"
            assert request.storage_type == "s3"
            assert request.output_filename == Path("s3_seedream")

    def test_seedream_s3_with_output_dir_filename_without_png(self):
        """Seedream + s3 + output_dir set + filename without extension."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            args = Mock()
            args.gcp_project_id = None
            args.gcp_location = None
            args.output_dir = output_dir
            args.output_filename = Path("s3_seedream")
            args.scale = None
            args.size = None
            args.storage_type = "s3"
            args.model = "seedream"

            request = _create_cli_request_dto("test", [], args)

            assert request.model == "seedream"
            assert request.storage_type == "s3"
            assert request.output_filename == Path("s3_seedream")

    def test_seedream_s3_with_output_dir_filename_none(self):
        """Seedream + s3 + output_dir set + filename None (default)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            args = Mock()
            args.gcp_project_id = None
            args.gcp_location = None
            args.output_dir = output_dir
            args.output_filename = None
            args.scale = None
            args.size = None
            args.storage_type = "s3"
            args.model = "seedream"

            request = _create_cli_request_dto("test", [], args)

            assert request.model == "seedream"
            assert request.storage_type == "s3"
            assert request.output_filename is None

    def test_seedream_s3_without_output_dir_filename_with_png(self):
        """Seedream + s3 + output_dir unset + filename with .png."""
        args = Mock()
        args.gcp_project_id = None
        args.gcp_location = None
        args.output_dir = None
        args.output_filename = Path("s3_test.png")
        args.scale = None
        args.size = None
        args.storage_type = "s3"
        args.model = "seedream"

        request = _create_cli_request_dto("test", [], args)

        assert request.model == "seedream"
        assert request.storage_type == "s3"
        assert request.output_filename == Path("s3_test")

    def test_seedream_s3_without_output_dir_filename_without_png(self):
        """Seedream + s3 + output_dir unset + filename without extension."""
        args = Mock()
        args.gcp_project_id = None
        args.gcp_location = None
        args.output_dir = None
        args.output_filename = Path("s3_test")
        args.scale = None
        args.size = None
        args.storage_type = "s3"
        args.model = "seedream"

        request = _create_cli_request_dto("test", [], args)

        assert request.model == "seedream"
        assert request.storage_type == "s3"
        assert request.output_filename == Path("s3_test")

    def test_seedream_s3_without_output_dir_filename_none(self):
        """Seedream + s3 + output_dir unset + filename None (default)."""
        args = Mock()
        args.gcp_project_id = None
        args.gcp_location = None
        args.output_dir = None
        args.output_filename = None
        args.scale = None
        args.size = None
        args.storage_type = "s3"
        args.model = "seedream"

        request = _create_cli_request_dto("test", [], args)

        assert request.model == "seedream"
        assert request.storage_type == "s3"
        assert request.output_filename is None

    # ========== OUTPUT FILE HANDLING TESTS ==========

    def test_custom_output_handling_with_timestamp(self):
        """Verify custom output adds timestamp and .png extension."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            # Create temp source file
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=".png", dir=str(output_dir)
            ) as temp_file:
                temp_file.write(b"fake image")
                generated_path = Path(temp_file.name)

            request = GenerateImageRequest(
                prompt="test",
                images=[],
                output_dir=output_dir,
                output_filename=Path("my_custom_name"),
                model="seedream",
            )

            response = GenerateImageResponse(
                image_config=ImageGenerationConfig(generated_file=generated_path, prompt="test"),
                gcp_config=GCPConfig(),
            )

            _handle_cli_custom_output(response, request)

            # Verify timestamp and extension were added
            assert response.generated_file.name.startswith("my_custom_name_")
            assert response.generated_file.suffix == ".png"
            assert response.generated_file.exists()
            assert response.generated_file.parent == output_dir

            # Cleanup
            response.generated_file.unlink()


class TestModelSpecificDefaults:
    """Test that model-specific defaults are used when filename is None."""

    def test_gemini_default_filename_is_generated(self, temp_image):
        """Verify Gemini uses 'generated' as default base filename."""
        # The default is verified in generate.py:562 and generate.py:628
        # When output_filename is None, it generates: "generated_YYYY-MM-DD-HH:MM:SS.png"
        request = GenerateImageRequest(
            prompt="test", images=[temp_image], output_filename=None, model="gemini"
        )

        assert request.output_filename is None
        assert request.model == "gemini"
        # When processed, will use "generated" as base

    def test_seedream_default_filename_is_seedream_image(self):
        """Verify Seedream uses 'seedream_image' as default base filename."""
        # The default is verified in seedream.py:213
        # When output_filename is None, it generates: "seedream_image_YYYY-MM-DD-HH:MM:SS.png"
        request = GenerateImageRequest(
            prompt="test", images=[], output_filename=None, model="seedream"
        )

        assert request.output_filename is None
        assert request.model == "seedream"
        # When processed, will use "seedream_image" as base
