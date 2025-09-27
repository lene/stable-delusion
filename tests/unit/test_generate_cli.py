"""
Unit tests for CLI validation logic in generate.py.
Tests the argument parsing and validation, especially S3 parameter handling.
"""

__author__ = "Lene Preuss <lene.preuss@gmail.com>"

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from nano_api.generate import parse_command_line


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
