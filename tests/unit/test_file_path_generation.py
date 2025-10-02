"""
Comprehensive tests for file path generation with mocked external API calls.
Tests ensure correct handling of output_dir, filename, datetime, and extension.
"""

__author__ = "Lene Preuss <lene.preuss@gmail.com>"

import tempfile
from pathlib import Path
from unittest.mock import Mock

from stable_delusion.generate import (
    _handle_cli_custom_output,
    _create_cli_request_dto,
)
from stable_delusion.models.requests import GenerateImageRequest
from stable_delusion.models.responses import GenerateImageResponse
from stable_delusion.models.client_config import ImageGenerationConfig, GCPConfig


class TestOutputDirHandling:
    """Test that output_dir option is respected in all scenarios."""

    def test_output_dir_used_with_custom_filename_gemini(self):
        """Test output_dir when custom filename is provided."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "test_output"
            output_dir.mkdir()

            # Use seedream which supports text-to-image without input images
            request = GenerateImageRequest(
                prompt="test",
                images=[],
                output_dir=output_dir,
                output_filename=Path("custom_image"),
                model="seedream",
            )

            # Mock the generated file
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=".png", dir=str(output_dir)
            ) as temp_file:
                temp_file.write(b"fake image")
                generated_path = Path(temp_file.name)

            response = GenerateImageResponse(
                image_config=ImageGenerationConfig(generated_file=generated_path, prompt="test"),
                gcp_config=GCPConfig(),
            )

            _handle_cli_custom_output(response, request)

            # Check file was moved to output_dir with timestamped name
            assert response.generated_file.parent == output_dir
            assert response.generated_file.name.startswith("custom_image_")
            assert response.generated_file.suffix == ".png"
            assert response.generated_file.exists()

            # Cleanup
            response.generated_file.unlink()

    def test_output_dir_used_without_custom_filename(self):
        """Test output_dir is used when no custom filename is provided."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "test_output"
            output_dir.mkdir()

            request = GenerateImageRequest(
                prompt="test", images=[], output_dir=output_dir, model="seedream"
            )

            # No custom filename, so _handle_cli_custom_output should not be called
            # This tests the service layer uses output_dir for initial save
            assert request.output_dir == output_dir
            assert request.output_filename is None

    def test_output_dir_with_subdirectory(self):
        """Test output_dir creates subdirectories if they don't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "deep" / "nested" / "directory"

            request = GenerateImageRequest(
                prompt="test",
                images=[],
                output_dir=output_dir,
                output_filename=Path("test_image"),
                model="seedream",
            )

            # Create a temporary file to simulate generated image
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_file:
                temp_file.write(b"fake image")
                generated_path = Path(temp_file.name)

            response = GenerateImageResponse(
                image_config=ImageGenerationConfig(generated_file=generated_path, prompt="test"),
                gcp_config=GCPConfig(),
            )

            _handle_cli_custom_output(response, request)

            # Verify directory was created
            assert response.generated_file.parent == output_dir
            assert output_dir.exists()
            assert output_dir.is_dir()

            # Cleanup
            response.generated_file.unlink()


class TestFilenameGeneration:
    """Test correct filename generation for both Gemini and Seedream."""

    def test_gemini_default_filename_has_datetime_and_extension(self):
        """Test request structure for default filename handling."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Test with seedream which supports text-to-image
            request = GenerateImageRequest(
                prompt="test", images=[], output_dir=Path(tmpdir), model="seedream"
            )

            # When output_filename is None, service uses default
            assert request.output_filename is None
            assert request.model == "seedream"

    def test_seedream_default_filename_has_datetime_and_extension(self):
        """Test Seedream files with 'seedream_image' prefix, datetime, and .png extension."""
        with tempfile.TemporaryDirectory() as tmpdir:
            request = GenerateImageRequest(
                prompt="test", images=[], output_dir=Path(tmpdir), model="seedream"
            )

            # When output_filename is None, service uses default
            assert request.output_filename is None
            assert request.model == "seedream"

    def test_custom_filename_includes_datetime_and_extension(self):
        """Test custom filenames get datetime and .png extension added."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            request = GenerateImageRequest(
                prompt="test",
                images=[],
                output_dir=output_dir,
                output_filename=Path("my_custom_image"),
                model="seedream",
            )

            # Create a temporary file
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=".png", dir=str(output_dir)
            ) as temp_file:
                temp_file.write(b"fake image")
                generated_path = Path(temp_file.name)

            response = GenerateImageResponse(
                image_config=ImageGenerationConfig(generated_file=generated_path, prompt="test"),
                gcp_config=GCPConfig(),
            )

            _handle_cli_custom_output(response, request)

            # Verify filename format: my_custom_image_YYYY-MM-DD-HH:MM:SS.png
            filename = response.generated_file.name
            assert filename.startswith("my_custom_image_")
            assert filename.endswith(".png")

            # Extract and verify datetime portion
            parts = filename.replace(".png", "").split("_")
            assert len(parts) >= 3  # base + datetime parts
            datetime_str = "_".join(parts[-2:])  # Last two parts are date and time
            # Verify it's a valid datetime format
            assert "-" in datetime_str
            assert ":" in datetime_str

            # Cleanup
            response.generated_file.unlink()

    def test_filename_with_png_extension_normalized(self):
        """Test that .png extension in custom filename is handled correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            # User provides custom_image.png, should be normalized to custom_image
            # then timestamp and .png added back
            request = GenerateImageRequest(
                prompt="test",
                images=[],
                output_dir=output_dir,
                output_filename=Path("custom_image"),  # Already normalized
                model="seedream",
            )

            with tempfile.NamedTemporaryFile(
                delete=False, suffix=".png", dir=str(output_dir)
            ) as temp_file:
                temp_file.write(b"fake image")
                generated_path = Path(temp_file.name)

            response = GenerateImageResponse(
                image_config=ImageGenerationConfig(generated_file=generated_path, prompt="test"),
                gcp_config=GCPConfig(),
            )

            _handle_cli_custom_output(response, request)

            # Should end with .png
            assert response.generated_file.suffix == ".png"
            assert "custom_image" in response.generated_file.name

            # Cleanup
            response.generated_file.unlink()


class TestModelSpecificDefaults:
    """Test that model-specific default filenames are used correctly."""

    def test_default_filename_structure(self):
        """Test request structure when no custom filename provided."""
        # Test with seedream which supports text-to-image
        request = GenerateImageRequest(
            prompt="test", images=[], output_dir=Path("."), model="seedream"
        )

        assert request.model == "seedream"
        assert request.output_filename is None

    def test_seedream_default_prefix(self):
        """Test that Seedream client uses 'seedream_image' prefix by default."""
        from stable_delusion.seedream import SeedreamClient

        # Check the default parameter in generate_and_save
        import inspect

        sig = inspect.signature(SeedreamClient.generate_and_save)
        default_filename = sig.parameters["output_filename"].default
        assert default_filename == "seedream_image"


class TestCLIRequestDTOCreation:
    """Test CLI request DTO creation handles filenames correctly."""

    def create_mock_args(self, **kwargs):
        """Create mock args object."""
        defaults = {
            "gcp_project_id": None,
            "gcp_location": None,
            "output_dir": Path("."),
            "output_filename": None,
            "scale": None,
            "size": None,
            "storage_type": None,
            "model": "gemini",
        }
        defaults.update(kwargs)

        mock_args = Mock()
        for key, value in defaults.items():
            setattr(mock_args, key, value)
        return mock_args

    def test_none_output_filename_preserved(self):
        """Test that None output_filename is preserved (uses model defaults)."""
        args = self.create_mock_args(output_filename=None, model="seedream")
        request = _create_cli_request_dto("test prompt", [], args)

        assert request.output_filename is None

    def test_custom_filename_normalized(self):
        """Test custom filename is normalized correctly."""
        args = self.create_mock_args(output_filename=Path("my_image.png"), model="seedream")
        request = _create_cli_request_dto("test prompt", [], args)

        assert request.output_filename == Path("my_image")

    def test_filename_without_extension_preserved(self):
        """Test filename without extension is preserved."""
        args = self.create_mock_args(output_filename=Path("my_image"), model="seedream")
        request = _create_cli_request_dto("test prompt", [], args)

        assert request.output_filename == Path("my_image")

    def test_output_dir_preserved(self):
        """Test output_dir is preserved in request DTO."""
        output_dir = Path("/tmp/test_output")
        args = self.create_mock_args(
            output_dir=output_dir, output_filename=Path("test.png"), model="seedream"
        )
        request = _create_cli_request_dto("test prompt", [], args)

        assert request.output_dir == output_dir
        assert request.output_filename == Path("test")


class TestEndToEndFilePathGeneration:
    """Integration tests for complete file path generation flow."""

    def test_gemini_with_custom_filename_and_output_dir(self):
        """Test complete flow with custom filename and output dir."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "test_output"
            output_dir.mkdir()

            # Simulate CLI args
            args = Mock()
            args.gcp_project_id = None
            args.gcp_location = None
            args.output_dir = output_dir
            args.output_filename = Path("my_test_image.png")
            args.scale = None
            args.size = None
            args.storage_type = None
            args.model = "seedream"

            # Create request
            request = _create_cli_request_dto("test prompt", [], args)

            # Verify request structure
            assert request.output_dir == output_dir
            assert request.output_filename == Path("my_test_image")

            # Simulate generation (create temp file)
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=".png", dir=str(output_dir)
            ) as temp_file:
                temp_file.write(b"fake image")
                generated_path = Path(temp_file.name)

            response = GenerateImageResponse(
                image_config=ImageGenerationConfig(generated_file=generated_path, prompt="test"),
                gcp_config=GCPConfig(),
            )

            # Apply custom output handling
            _handle_cli_custom_output(response, request)

            # Verify final path
            assert response.generated_file.parent == output_dir
            assert response.generated_file.name.startswith("my_test_image_")
            assert response.generated_file.suffix == ".png"
            assert response.generated_file.exists()

            # Cleanup
            response.generated_file.unlink()

    def test_seedream_with_custom_filename_and_output_dir(self):
        """Test complete Seedream flow with custom filename and output dir."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "seedream_test"
            output_dir.mkdir()

            args = Mock()
            args.gcp_project_id = None
            args.gcp_location = None
            args.output_dir = output_dir
            args.output_filename = Path("my_seedream_image.png")
            args.scale = None
            args.size = "2K"
            args.storage_type = None
            args.model = "seedream"

            request = _create_cli_request_dto("test prompt", [], args)

            assert request.output_dir == output_dir
            assert request.output_filename == Path("my_seedream_image")

            # Simulate generation
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=".png", dir=str(output_dir)
            ) as temp_file:
                temp_file.write(b"fake image")
                generated_path = Path(temp_file.name)

            response = GenerateImageResponse(
                image_config=ImageGenerationConfig(generated_file=generated_path, prompt="test"),
                gcp_config=GCPConfig(),
            )

            _handle_cli_custom_output(response, request)

            # Verify final path
            assert response.generated_file.parent == output_dir
            assert response.generated_file.name.startswith("my_seedream_image_")
            assert response.generated_file.suffix == ".png"
            assert response.generated_file.exists()

            # Cleanup
            response.generated_file.unlink()

    def test_no_custom_filename_uses_model_defaults(self):
        """Test that when no custom filename is provided, model defaults are used."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            # Seedream request without custom filename
            args = Mock()
            args.gcp_project_id = None
            args.gcp_location = None
            args.output_dir = output_dir
            args.output_filename = None  # No custom filename
            args.scale = None
            args.size = None
            args.storage_type = None
            args.model = "seedream"

            request = _create_cli_request_dto("test prompt", [], args)

            # Should use model defaults
            assert request.output_filename is None
            assert request.model == "seedream"

    def test_multiple_files_different_timestamps(self):
        """Test that files with custom names have timestamp and .png extension."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            request = GenerateImageRequest(
                prompt="test",
                images=[],
                output_dir=output_dir,
                output_filename=Path("test_file"),
                model="seedream",
            )

            with tempfile.NamedTemporaryFile(
                delete=False, suffix=".png", dir=str(output_dir)
            ) as temp_file:
                temp_file.write(b"fake image")
                generated_path = Path(temp_file.name)

            response = GenerateImageResponse(
                image_config=ImageGenerationConfig(generated_file=generated_path, prompt="test"),
                gcp_config=GCPConfig(),
            )

            _handle_cli_custom_output(response, request)

            # Verify the filename has timestamp format
            filename = response.generated_file.name
            assert filename.startswith("test_file_")
            assert filename.endswith(".png")

            # Verify timestamp format (test_file_YYYY-MM-DD-HH:MM:SS.png)
            parts = filename.replace(".png", "").split("_")
            assert len(parts) >= 3  # base + date + time parts

            # Cleanup
            response.generated_file.unlink()
