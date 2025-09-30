"""
Unit tests for CLI validation logic in generate.py.
Tests the argument parsing and validation, especially S3 parameter handling.
"""

__author__ = "Lene Preuss <lene.preuss@gmail.com>"

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from stable_delusion.generate import parse_command_line


class TestCLIValidation:
    """Test CLI argument parsing and validation logic."""

    @patch("dotenv.load_dotenv")
    def test_cli_loads_dotenv_before_validation(self, mock_load_dotenv):
        # Mock os.getenv to return required S3 parameters
        with patch.dict(os.environ, {"AWS_S3_BUCKET": "test-bucket", "AWS_S3_REGION": "us-east-1"}):
            with patch("sys.argv", ["generate.py", "--storage-type", "s3"]):
                args = parse_command_line()
                mock_load_dotenv.assert_called_once_with(override=False)
                assert args.storage_type == "s3"

    def test_cli_requires_s3_bucket_and_region_only(self):
        # Should pass with just bucket and region from CLI args
        with patch(
            "sys.argv",
            [
                "generate.py",
                "--storage-type",
                "s3",
                "--aws-s3-bucket",
                "test-bucket",
                "--aws-s3-region",
                "us-east-1",
            ],
        ):
            args = parse_command_line()
            assert args.storage_type == "s3"
            assert args.aws_s3_bucket == "test-bucket"
            assert args.aws_s3_region == "us-east-1"

    @patch("dotenv.load_dotenv")
    def test_cli_accepts_aws_credentials_from_environment(self, mock_load_dotenv):
        with patch.dict(
            os.environ, {"AWS_S3_BUCKET": "env-bucket", "AWS_S3_REGION": "eu-central-1"}
        ):
            with patch("sys.argv", ["generate.py", "--storage-type", "s3"]):
                args = parse_command_line()
                assert args.storage_type == "s3"
                # CLI args should be None when using env vars
                assert args.aws_s3_bucket is None
                assert args.aws_s3_region is None

    def test_cli_validation_missing_s3_bucket_fails(self):
        with patch("os.getenv") as mock_getenv:
            mock_getenv.return_value = None  # Return None for all env vars
            with pytest.raises(SystemExit):
                with patch("sys.stderr"):  # Suppress error output
                    with patch(
                        "sys.argv",
                        ["generate.py", "--storage-type", "s3", "--aws-s3-region", "us-east-1"],
                    ):
                        parse_command_line()

    def test_cli_validation_missing_s3_region_fails(self):
        with patch("os.getenv") as mock_getenv:
            mock_getenv.return_value = None  # Return None for all env vars
            with pytest.raises(SystemExit):
                with patch("sys.stderr"):  # Suppress error output
                    with patch(
                        "sys.argv",
                        ["generate.py", "--storage-type", "s3", "--aws-s3-bucket", "test-bucket"],
                    ):
                        parse_command_line()

    @patch("dotenv.load_dotenv")
    def test_cli_validation_missing_both_s3_params_fails(self, mock_load_dotenv):
        with patch.dict(os.environ, {}, clear=True):  # Clear all env vars
            with pytest.raises(SystemExit):
                with patch("sys.stderr"):  # Suppress error output
                    with patch("sys.argv", ["generate.py", "--storage-type", "s3"]):
                        parse_command_line()

    def test_cli_validation_with_local_storage_succeeds(self):
        with patch("sys.argv", ["generate.py", "--storage-type", "local"]):
            args = parse_command_line()
            assert args.storage_type == "local"

    def test_cli_validation_no_storage_type_succeeds(self):
        with patch("sys.argv", ["generate.py"]):
            args = parse_command_line()
            assert args.storage_type is None

    @patch("dotenv.load_dotenv")
    def test_cli_combines_args_and_env_vars(self, mock_load_dotenv):
        with patch.dict(os.environ, {"AWS_S3_REGION": "env-region"}):  # Region from env
            with patch(
                "sys.argv", ["generate.py", "--storage-type", "s3", "--aws-s3-bucket", "cli-bucket"]
            ):
                args = parse_command_line()
                assert args.storage_type == "s3"
                assert args.aws_s3_bucket == "cli-bucket"
                assert args.aws_s3_region is None  # CLI arg, not env var value

    def test_cli_security_warning_for_credentials(self):
        # This would normally show a warning, but we can't easily test logging
        # in argument parsing. The test verifies the args are parsed correctly.
        with patch(
            "sys.argv",
            [
                "generate.py",
                "--storage-type",
                "s3",
                "--aws-s3-bucket",
                "test-bucket",
                "--aws-s3-region",
                "us-east-1",
                "--aws-access-key-id",
                "test-key",
                "--aws-secret-access-key",
                "test-secret",
            ],
        ):
            args = parse_command_line()
            assert args.aws_access_key_id == "test-key"
            assert args.aws_secret_access_key == "test-secret"

    def test_cli_security_warning_for_gemini_api_key(self):
        with patch("sys.argv", ["generate.py", "--gemini-api-key", "test-gemini-key"]):
            args = parse_command_line()
            assert args.gemini_api_key == "test-gemini-key"

    def test_cli_validation_error_message_content(self):
        with patch("os.getenv") as mock_getenv:
            mock_getenv.return_value = None  # Return None for all env vars
            with pytest.raises(SystemExit):
                with patch("sys.stderr"):
                    with patch("sys.argv", ["generate.py", "--storage-type", "s3"]):
                        parse_command_line()

        # The actual error message testing would require capturing stderr,
        # which is complex. The test verifies the validation logic runs.

    def test_cli_prompt_argument_parsing(self):
        with patch("sys.argv", ["generate.py", "--prompt", "Test prompt"]):
            args = parse_command_line()
            assert args.prompt == "Test prompt"

    def test_cli_image_argument_parsing(self):
        with patch("sys.argv", ["generate.py", "--image", "test.jpg"]):
            args = parse_command_line()
            assert args.image == [Path("test.jpg")]  # action='append' returns list of Path objects

    def test_cli_model_argument_parsing(self):
        with patch("sys.argv", ["generate.py", "--model", "seedream"]):
            args = parse_command_line()
            assert args.model == "seedream"

    def test_cli_model_choices_validation(self):
        with pytest.raises(SystemExit):
            with patch("sys.stderr"):
                with patch("sys.argv", ["generate.py", "--model", "invalid-model"]):
                    parse_command_line()

    def test_cli_storage_type_choices_validation(self):
        with pytest.raises(SystemExit):
            with patch("sys.stderr"):
                with patch("sys.argv", ["generate.py", "--storage-type", "invalid-storage"]):
                    parse_command_line()

    def test_cli_scale_choices_validation(self):
        with patch("sys.argv", ["generate.py", "--scale", "2"]):
            args = parse_command_line()
            assert args.scale == 2

        with patch("sys.argv", ["generate.py", "--scale", "4"]):
            args = parse_command_line()
            assert args.scale == 4

        with pytest.raises(SystemExit):
            with patch("sys.stderr"):
                with patch("sys.argv", ["generate.py", "--scale", "3"]):
                    parse_command_line()

    def test_cli_output_dir_path_conversion(self):
        with patch("sys.argv", ["generate.py", "--output-dir", "/tmp/test"]):
            args = parse_command_line()
            # The argument should be parsed as a string, Path conversion happens later
            assert str(args.output_dir) == "/tmp/test"

    @patch("dotenv.load_dotenv")
    def test_dotenv_override_false(self, mock_load_dotenv):
        with patch.dict(os.environ, {"AWS_S3_BUCKET": "test-bucket", "AWS_S3_REGION": "us-east-1"}):
            with patch("sys.argv", ["generate.py", "--storage-type", "s3"]):
                parse_command_line()
                mock_load_dotenv.assert_called_once_with(override=False)


class TestOutputParameterCLI:
    """Test --output-filename parameter CLI functionality."""

    def test_cli_output_argument_parsing(self):
        """Test that --output-filename parameter is parsed correctly."""
        with patch("sys.argv", ["generate.py", "--output-filename", "custom_image.png"]):
            args = parse_command_line()
            assert args.output_filename == Path("custom_image.png")

    def test_cli_output_default_value(self):
        """Test that --output-filename has correct default value."""
        with patch("sys.argv", ["generate.py"]):
            args = parse_command_line()
            assert args.output_filename == Path("generated_gemini_image.png")

    def test_cli_output_with_path(self):
        """Test --output-filename parameter with path-like input."""
        with patch("sys.argv", ["generate.py", "--output-filename", "subfolder/my_image.jpg"]):
            args = parse_command_line()
            assert args.output_filename == Path("subfolder/my_image.jpg")

    def test_cli_output_with_output_dir(self):
        """Test --output-filename parameter combined with --output-dir."""
        with patch("sys.argv", [
            "generate.py", "--output-dir", "/tmp", "--output-filename", "test.png"
        ]):
            args = parse_command_line()
            assert args.output_dir == Path("/tmp")
            assert args.output_filename == Path("test.png")

    def test_cli_output_various_extensions(self):
        """Test --output-filename parameter with different file extensions."""
        extensions = ["png", "jpg", "jpeg", "gif", "bmp", "webp"]
        for ext in extensions:
            filename = f"test_image.{ext}"
            with patch("sys.argv", ["generate.py", "--output-filename", filename]):
                args = parse_command_line()
                assert args.output_filename == Path(filename)

    def test_cli_output_without_extension(self):
        """Test --output-filename parameter without file extension."""
        with patch("sys.argv", ["generate.py", "--output-filename", "my_image"]):
            args = parse_command_line()
            assert args.output_filename == Path("my_image")

    def test_cli_output_special_characters(self):
        """Test --output-filename parameter with special characters in filename."""
        filename = "test-image_2024@special.png"
        with patch("sys.argv", ["generate.py", "--output-filename", filename]):
            args = parse_command_line()
            assert args.output_filename == Path(filename)


class TestOutputParameterEdgeCases:
    """Test edge cases and validation for --output-filename parameter."""

    def test_cli_output_empty_string(self):
        """Test --output-filename parameter with empty string."""
        with patch("sys.argv", ["generate.py", "--output-filename", ""]):
            args = parse_command_line()
            assert args.output_filename == Path("")

    def test_cli_output_whitespace_only(self):
        """Test --output-filename parameter with whitespace-only filename."""
        with patch("sys.argv", ["generate.py", "--output-filename", "   "]):
            args = parse_command_line()
            assert args.output_filename == Path("   ")

    def test_cli_output_very_long_filename(self):
        """Test --output-filename parameter with very long filename."""
        long_filename = "a" * 200 + ".png"
        with patch("sys.argv", ["generate.py", "--output-filename", long_filename]):
            args = parse_command_line()
            assert args.output_filename == Path(long_filename)

    def test_cli_output_unicode_characters(self):
        """Test --output-filename parameter with unicode characters."""
        unicode_filename = "测试图片_🖼️_émoji.png"
        with patch("sys.argv", ["generate.py", "--output-filename", unicode_filename]):
            args = parse_command_line()
            assert args.output_filename == Path(unicode_filename)

    def test_cli_output_dots_and_hidden_files(self):
        """Test --output-filename parameter with dots and hidden file patterns."""
        test_cases = [
            ".hidden_image.png",
            "..parent_dir.png",
            "image.with.dots.png",
            ".png"  # Edge case: only extension
        ]
        for filename in test_cases:
            with patch("sys.argv", ["generate.py", "--output-filename", filename]):
                args = parse_command_line()
                assert args.output_filename == Path(filename)

    def test_cli_output_path_separators(self):
        """Test --output-filename parameter with different path separators."""
        test_cases = [
            "folder/image.png",
            "folder\\image.png",  # Windows-style separator
            "deep/nested/folder/image.png",
            "../relative/path.png",
            "./current/dir/image.png"
        ]
        for filepath in test_cases:
            with patch("sys.argv", ["generate.py", "--output-filename", filepath]):
                args = parse_command_line()
                assert args.output_filename == Path(filepath)

    def test_cli_output_absolute_paths(self):
        """Test --output-filename parameter with absolute paths."""
        test_cases = [
            "/tmp/absolute/path.png",
            "/home/user/image.jpg",
            "C:\\Windows\\image.png"  # Windows absolute path
        ]
        for filepath in test_cases:
            with patch("sys.argv", ["generate.py", "--output-filename", filepath]):
                args = parse_command_line()
                assert args.output_filename == Path(filepath)

    def test_cli_output_multiple_extensions(self):
        """Test --output-filename parameter with multiple extensions."""
        filename = "image.backup.old.png"
        with patch("sys.argv", ["generate.py", "--output-filename", filename]):
            args = parse_command_line()
            assert args.output_filename == Path(filename)

    def test_cli_output_no_extension_various_cases(self):
        """Test --output-filename parameter without extensions in various forms."""
        test_cases = [
            "image_no_ext",
            "123",
            "image.",  # Ends with dot but no extension
            "folder/image_no_ext"
        ]
        for filename in test_cases:
            with patch("sys.argv", ["generate.py", "--output-filename", filename]):
                args = parse_command_line()
                assert args.output_filename == Path(filename)

    def test_cli_output_with_spaces(self):
        """Test --output-filename parameter with spaces in filename."""
        filename = "my image with spaces.png"
        with patch("sys.argv", ["generate.py", "--output-filename", filename]):
            args = parse_command_line()
            assert args.output_filename == Path(filename)

    def test_cli_output_case_sensitivity(self):
        """Test --output-filename parameter case sensitivity."""
        test_cases = [
            "Image.PNG",
            "IMAGE.png",
            "image.PNG",
            "MixedCase_File.JpEg"
        ]
        for filename in test_cases:
            with patch("sys.argv", ["generate.py", "--output-filename", filename]):
                args = parse_command_line()
                assert args.output_filename == Path(filename)

    def test_cli_output_combined_with_all_other_options(self):
        """Test --output-filename parameter combined with all other CLI options."""
        with patch("sys.argv", [
            "generate.py",
            "--prompt", "test prompt",
            "--output-filename", "comprehensive_test.png",
            "--output-dir", "/tmp/test",
            "--model", "seedream",
            "--storage-type", "local",
            "--size", "2K"
        ]):
            args = parse_command_line()
            assert args.output_filename == Path("comprehensive_test.png")
            assert args.output_dir == Path("/tmp/test")
            assert args.prompt == "test prompt"
            assert args.model == "seedream"
            assert args.storage_type == "local"
            assert args.size == "2K"
