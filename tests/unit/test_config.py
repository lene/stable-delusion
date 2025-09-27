"""
Unit tests for configuration management system.
Tests environment variable loading, validation, and default values.
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from nano_api.config import Config, ConfigManager
from nano_api.exceptions import ConfigurationError


# Note: .env file loading prevention is now handled globally in conftest.py


class TestConfig:
    """Test Config dataclass functionality."""

    def test_config_with_valid_data(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config = Config(
                project_id="test-project",
                location="us-central1",
                gemini_api_key="test-key",
                upload_folder=Path(temp_dir) / "uploads",
                default_output_dir=Path(temp_dir) / "output",
                flask_debug=False,
                storage_type="local",
                s3_bucket=None,
                s3_region=None,
                aws_access_key_id=None,
                aws_secret_access_key=None,
            )

            assert config.project_id == "test-project"
            assert config.location == "us-central1"
            assert config.gemini_api_key == "test-key"
            assert config.flask_debug is False
            # Check directories were created for local storage
            assert config.upload_folder.exists()
            assert config.default_output_dir.exists()
            # Check S3 settings default to None
            assert config.storage_type == "local"
            assert config.s3_bucket is None

    def test_config_missing_api_key(self):
        # GEMINI_API_KEY validation is now done only when GeminiClient is created
        # Config creation should succeed even with empty API key
        with tempfile.TemporaryDirectory() as temp_dir:
            config = Config(
                project_id="test-project",
                location="us-central1",
                gemini_api_key="",
                upload_folder=Path(temp_dir) / "uploads",
                default_output_dir=Path(temp_dir) / "output",
                flask_debug=False,
                storage_type="local",
                s3_bucket=None,
                s3_region=None,
                aws_access_key_id=None,
                aws_secret_access_key=None,
            )
            assert config.gemini_api_key == ""


class TestConfigManager:
    """Test ConfigManager functionality."""

    def setup_method(self):
        ConfigManager.reset_config()

    def test_config_manager_singleton(self):
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}):
            config1 = ConfigManager.get_config()
            config2 = ConfigManager.get_config()
            assert config1 is config2

    def test_config_manager_reset(self):
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}):
            config1 = ConfigManager.get_config()
            ConfigManager.reset_config()
            config2 = ConfigManager.get_config()
            assert config1 is not config2

    @patch.dict(
        os.environ,
        {
            "GEMINI_API_KEY": "test-key",
            "GCP_PROJECT_ID": "custom-project",
            "GCP_LOCATION": "us-west1",
            "UPLOAD_FOLDER": "custom_uploads",
            "DEFAULT_OUTPUT_DIR": "custom_output",
            "FLASK_DEBUG": "true",
        },
    )
    def test_config_from_environment_variables(self):
        config = ConfigManager.get_config()

        assert config.project_id == "custom-project"
        assert config.location == "us-west1"
        assert config.gemini_api_key == "test-key"
        assert config.upload_folder == Path("custom_uploads")
        assert config.default_output_dir == Path("custom_output")
        assert config.flask_debug is True

    @patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"})
    def test_config_with_defaults(self):
        config = ConfigManager.get_config()

        # Should use defaults from conf.py
        assert config.gemini_api_key == "test-key"
        assert config.upload_folder == Path("uploads")
        assert config.default_output_dir == Path(".")
        assert config.flask_debug is False

    @pytest.mark.parametrize(
        "debug_value,expected",
        [
            ("false", False),
            ("False", False),
            ("0", False),
            ("", False),
            ("no", False),
            ("true", True),
            ("True", True),
            ("1", True),
            ("yes", True),
            ("YES", True),
        ],
    )
    def test_flask_debug_parsing(self, debug_value, expected):
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key", "FLASK_DEBUG": debug_value}):
            ConfigManager.reset_config()
            config = ConfigManager.get_config()
            assert config.flask_debug is expected

    def test_config_missing_gemini_api_key(self):
        # GEMINI_API_KEY validation is now done only when GeminiClient is created
        # ConfigManager.get_config() should succeed even without GEMINI_API_KEY
        with patch.dict(os.environ, {}, clear=True):
            ConfigManager.reset_config()
            config = ConfigManager.get_config()
            assert config.gemini_api_key == ""

    @patch.dict(
        os.environ,
        {
            "GEMINI_API_KEY": "test-key",
            "STORAGE_TYPE": "s3",
            "AWS_S3_BUCKET": "test-bucket",
            "AWS_S3_REGION": "us-west2",
        },
    )
    def test_config_s3_storage_valid(self):
        ConfigManager.reset_config()
        config = ConfigManager.get_config()

        assert config.storage_type == "s3"
        assert config.s3_bucket == "test-bucket"
        assert config.s3_region == "us-west2"
        # Local directories should not be created for S3 storage

    @patch.dict(os.environ, {"GEMINI_API_KEY": "test-key", "STORAGE_TYPE": "s3"})
    def test_config_s3_missing_bucket(self):
        ConfigManager.reset_config()
        with pytest.raises(ConfigurationError, match="AWS_S3_BUCKET.*required"):
            ConfigManager.get_config()

    @patch.dict(
        os.environ,
        {"GEMINI_API_KEY": "test-key", "STORAGE_TYPE": "s3", "AWS_S3_BUCKET": "test-bucket"},
    )
    def test_config_s3_missing_region(self):
        ConfigManager.reset_config()
        with pytest.raises(ConfigurationError, match="AWS_S3_REGION.*required"):
            ConfigManager.get_config()

    @patch.dict(os.environ, {"GEMINI_API_KEY": "test-key", "STORAGE_TYPE": "local"})
    def test_config_local_storage_default(self):
        ConfigManager.reset_config()
        config = ConfigManager.get_config()

        assert config.storage_type == "local"
        assert config.s3_bucket is None
        assert config.s3_region is None
