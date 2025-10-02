"""
Unit tests for custom output filename handling in generate.py.
Tests the _handle_cli_custom_output function and related logic.
"""

__author__ = "Lene Preuss <lene.preuss@gmail.com>"

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from stable_delusion.generate import _handle_cli_custom_output, _create_cli_request_dto
from stable_delusion.models.requests import GenerateImageRequest
from stable_delusion.models.responses import GenerateImageResponse
from stable_delusion.models.client_config import ImageGenerationConfig, GCPConfig


class TestCustomOutputHandling:
    """Test custom output filename handling functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        # Create a temporary file to simulate generated image
        # pylint: disable=consider-using-with
        self.temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        self.temp_file.write(b"fake image content")
        self.temp_file.close()
        self.generated_file_path = Path(self.temp_file.name)

    def teardown_method(self):
        """Clean up test fixtures."""
        # Clean up any files that might exist
        if self.generated_file_path.exists():
            self.generated_file_path.unlink()

        # Clean up potential renamed files
        for pattern in ["test_output.png", "custom.jpg", "renamed_file.png"]:
            test_file = Path(pattern)
            if test_file.exists():
                test_file.unlink()

    def create_test_response(self, generated_file_path: Path) -> GenerateImageResponse:
        """Create a test response object."""
        return GenerateImageResponse(
            image_config=ImageGenerationConfig(
                generated_file=generated_file_path, prompt="test prompt"
            ),
            gcp_config=GCPConfig(),
        )

    def test_handle_custom_output_with_output_dir(self):
        """Test custom output handling with specified output directory."""
        request = GenerateImageRequest(
            prompt="test",
            images=[],
            model="seedream",
            output_dir=Path("."),
            output_filename="test_output",  # No extension - will be added automatically
        )

        response = self.create_test_response(self.generated_file_path)

        _handle_cli_custom_output(response, request)

        # Check that file was renamed with timestamp and extension
        assert response.generated_file.name.startswith("test_output_")
        assert response.generated_file.suffix == ".png"
        assert response.generated_file.exists()
        assert response.generated_file.parent == Path(".")

    def test_handle_custom_output_without_output_dir(self):
        """Test custom output handling without specified output directory."""
        request = GenerateImageRequest(
            prompt="test", images=[], model="seedream", output_filename="custom"
        )

        response = self.create_test_response(self.generated_file_path)

        _handle_cli_custom_output(response, request)

        # Should use same directory as source file, with timestamp and .png extension
        assert response.generated_file.parent == self.generated_file_path.parent
        assert response.generated_file.name.startswith("custom_")
        assert response.generated_file.suffix == ".png"
        assert response.generated_file.exists()

    def test_handle_custom_output_no_filename(self):
        """Test that function does nothing when no custom filename is specified."""
        request = GenerateImageRequest(
            prompt="test", images=[], model="seedream", output_dir=Path(".")
        )

        response = self.create_test_response(self.generated_file_path)
        original_path = response.generated_file

        _handle_cli_custom_output(response, request)

        # Should not change anything
        assert response.generated_file == original_path
        assert response.generated_file.exists()

    def test_handle_custom_output_no_generated_file(self):
        """Test that function handles missing generated file gracefully."""
        request = GenerateImageRequest(
            prompt="test",
            images=[],
            model="seedream",
            output_dir=Path("."),
            output_filename="test_output.png",
        )

        response = GenerateImageResponse(
            image_config=ImageGenerationConfig(generated_file=None, prompt="test prompt"),
            gcp_config=GCPConfig(),
        )

        # Should not raise an exception
        _handle_cli_custom_output(response, request)

    @patch("stable_delusion.generate.logging")
    @patch("stable_delusion.generate.shutil.move")
    def test_handle_custom_output_rename_failure(self, mock_move, mock_logging):
        """Test error handling when file rename fails."""
        # Mock shutil.move to raise an exception
        mock_move.side_effect = OSError("Permission denied")

        request = GenerateImageRequest(
            prompt="test",
            images=[],
            model="seedream",
            output_dir=Path("."),
            output_filename="test_output.png",
        )

        response = self.create_test_response(self.generated_file_path)

        # This should handle the error gracefully
        _handle_cli_custom_output(response, request)

        # Check that error was logged
        mock_logging.error.assert_called()
        # Verify shutil.move was called
        mock_move.assert_called_once()

    def test_handle_custom_output_creates_target_directory(self):
        """Test that target directory is created if it doesn't exist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            target_dir = Path(temp_dir) / "new_subdir"

            request = GenerateImageRequest(
                prompt="test",
                images=[],
                model="seedream",
                output_dir=target_dir,
                output_filename="test_output",  # No extension - will be added automatically
            )

            response = self.create_test_response(self.generated_file_path)

            _handle_cli_custom_output(response, request)

            # Directory should be created and file with timestamp should exist
            assert target_dir.exists()
            assert response.generated_file.parent == target_dir
            assert response.generated_file.name.startswith("test_output_")
            assert response.generated_file.suffix == ".png"
            assert response.generated_file.exists()

    @patch("stable_delusion.generate.logging")
    def test_handle_custom_output_debug_logging(self, mock_logging):
        """Test that debug logging occurs during custom output handling."""
        request = GenerateImageRequest(
            prompt="test",
            images=[],
            model="seedream",
            output_dir=Path("."),
            output_filename="renamed_file.png",
        )

        response = self.create_test_response(self.generated_file_path)

        _handle_cli_custom_output(response, request)

        # Check that debug logging was called
        assert mock_logging.debug.call_count >= 3  # Should have multiple debug calls


class TestRequestDTOCreation:
    """Test CLI request DTO creation with output filename."""

    def create_mock_args(self, **kwargs):
        """Create a mock args object with default values."""
        defaults = {
            "gcp_project_id": None,
            "gcp_location": None,
            "output_dir": Path("."),
            "output_filename": Path("test.png"),
            "scale": None,
            "size": None,
            "storage_type": None,
            "model": "seedream",
        }
        defaults.update(kwargs)

        mock_args = Mock()
        for key, value in defaults.items():
            setattr(mock_args, key, value)
        return mock_args

    def test_create_cli_request_dto_with_output(self):
        """Test that CLI request DTO includes output_filename with PNG extension stripped."""
        args = self.create_mock_args(output_filename=Path("custom_output.png"))

        request = _create_cli_request_dto("test prompt", [], args)

        assert request.output_filename == Path("custom_output")
        assert request.prompt == "test prompt"
        assert request.model == "seedream"

    def test_create_cli_request_dto_without_output(self):
        """Test CLI request DTO creation when output is None."""
        args = self.create_mock_args(output_filename=None)

        request = _create_cli_request_dto("test prompt", [], args)

        assert request.output_filename is None

    def test_create_cli_request_dto_output_dir_handling(self):
        """Test that output_dir is properly included in request DTO."""
        args = self.create_mock_args(
            output_filename=Path("test.png"), output_dir=Path("/tmp/custom")
        )

        request = _create_cli_request_dto("test prompt", [], args)

        assert request.output_filename == Path("test")
        assert request.output_dir == Path("/tmp/custom")

    def test_create_cli_request_dto_path_handling(self):
        """Test that Path objects are properly handled for PNG files."""
        args = self.create_mock_args(output_filename=Path("complex/path/file.png"))

        request = _create_cli_request_dto("test prompt", [], args)

        assert request.output_filename == Path("complex/path/file")
        assert isinstance(request.output_filename, Path)


class TestPNGValidation:
    """Test PNG extension validation for output parameter."""

    def create_mock_args(self, **kwargs):
        """Create a mock args object with default values."""
        defaults = {
            "gcp_project_id": None,
            "gcp_location": None,
            "output_dir": Path("."),
            "output_filename": None,
            "scale": None,
            "size": None,
            "storage_type": None,
            "model": "seedream",
        }
        defaults.update(kwargs)

        mock_args = Mock()
        for key, value in defaults.items():
            setattr(mock_args, key, value)
        return mock_args

    def test_png_extension_stripped(self):
        """Test that .png extension is stripped from output filename."""
        args = self.create_mock_args(output_filename=Path("my_image.png"))
        request = _create_cli_request_dto("test prompt", [], args)
        assert request.output_filename == Path("my_image")

    def test_no_extension_preserved(self):
        """Test that filenames without extensions are preserved."""
        args = self.create_mock_args(output_filename=Path("my_image"))
        request = _create_cli_request_dto("test prompt", [], args)
        assert request.output_filename == Path("my_image")

    @patch("builtins.print")
    @patch("sys.exit")
    def test_unsupported_extension_fails(self, mock_exit, mock_print):
        """Test that unsupported extensions cause system exit."""
        # Make sys.exit actually raise SystemExit to stop execution
        mock_exit.side_effect = SystemExit(1)

        args = self.create_mock_args(output_filename=Path("my_image.jpg"))

        with pytest.raises(SystemExit):
            _create_cli_request_dto("test prompt", [], args)

        mock_print.assert_called_with(
            "Error: file type not supported for --output-filename: '.jpg'. "
            "Only PNG files are supported."
        )
        mock_exit.assert_called_with(1)

    @patch("builtins.print")
    @patch("sys.exit")
    def test_multiple_extensions_fail(self, mock_exit, mock_print):
        """Test that files with non-PNG extensions fail validation."""
        # Make sys.exit actually raise SystemExit to stop execution
        mock_exit.side_effect = SystemExit(1)

        test_cases = [".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".tiff"]

        for ext in test_cases:
            filename = f"test{ext}"
            args = self.create_mock_args(output_filename=Path(filename))

            with pytest.raises(SystemExit):
                _create_cli_request_dto("test prompt", [], args)

            mock_print.assert_called_with(
                f"Error: file type not supported for --output-filename: '{ext}'. "
                "Only PNG files are supported."
            )
            mock_exit.assert_called_with(1)

    def test_case_insensitive_png_validation(self):
        """Test that PNG extension validation is case insensitive."""
        test_cases = ["image.PNG", "image.png", "image.Png", "image.pNg"]

        for filename in test_cases:
            args = self.create_mock_args(output_filename=Path(filename))
            request = _create_cli_request_dto("test prompt", [], args)
            assert request.output_filename == Path("image")

    def test_complex_path_with_png(self):
        """Test PNG validation with complex paths."""
        args = self.create_mock_args(output_filename=Path("folder/subfolder/image.png"))
        request = _create_cli_request_dto("test prompt", [], args)
        assert request.output_filename == Path("folder/subfolder/image")


class TestOutputParameterIntegration:
    """Integration tests for output parameter end-to-end functionality."""

    def test_output_parameter_flow_with_mocked_service(self):
        """Test the complete flow from CLI args to custom output handling."""
        with patch(
            "stable_delusion.generate.builders.create_image_generation_service"
        ) as mock_builder:
            # Mock the service and its response
            mock_service = Mock()
            mock_builder.return_value = mock_service

            # Create a temporary file for the mock response
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_file:
                temp_file.write(b"mock image")
                temp_path = Path(temp_file.name)

            try:
                mock_response = GenerateImageResponse(
                    image_config=ImageGenerationConfig(generated_file=temp_path, prompt="test"),
                    gcp_config=GCPConfig(),
                )
                mock_service.generate_image.return_value = mock_response

                # Test the request DTO creation and custom output handling

                args = Mock()
                args.gcp_project_id = None
                args.gcp_location = None
                args.output_dir = Path(".")
                args.output_filename = Path("integration_test.png")
                args.scale = None
                args.size = None
                args.storage_type = None
                args.model = "seedream"

                request = _create_cli_request_dto("integration test", [], args)
                _handle_cli_custom_output(mock_response, request)

                # Verify the output filename was applied with timestamp
                assert mock_response.generated_file.name.startswith("integration_test_")
                assert mock_response.generated_file.suffix == ".png"
                assert mock_response.generated_file.exists()

            finally:
                # Cleanup
                if temp_path.exists():
                    temp_path.unlink()
                # Clean up the timestamped file
                if mock_response.generated_file.exists():
                    mock_response.generated_file.unlink()
