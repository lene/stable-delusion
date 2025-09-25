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


class TestConfig:
    """Test Config dataclass functionality."""

    def test_config_with_valid_data(self):
        """Test Config creation with valid data."""
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
                aws_secret_access_key=None
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
        """Test Config validation fails with missing API key."""
        with tempfile.TemporaryDirectory() as temp_dir:
            with pytest.raises(ConfigurationError, match="GEMINI_API_KEY.*required"):
                Config(
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
                    aws_secret_access_key=None
                )


class TestConfigManager:
    """Test ConfigManager functionality."""

    def setup_method(self):
        """Reset ConfigManager before each test."""
        ConfigManager.reset_config()

    def test_config_manager_singleton(self):
        """Test ConfigManager implements singleton pattern."""
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}):
            config1 = ConfigManager.get_config()
            config2 = ConfigManager.get_config()
            assert config1 is config2

    def test_config_manager_reset(self):
        """Test ConfigManager reset functionality."""
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}):
            config1 = ConfigManager.get_config()
            ConfigManager.reset_config()
            config2 = ConfigManager.get_config()
            assert config1 is not config2

    @patch.dict(os.environ, {
        "GEMINI_API_KEY": "test-key",
        "GCP_PROJECT_ID": "custom-project",
        "GCP_LOCATION": "us-west1",
        "UPLOAD_FOLDER": "custom_uploads",
        "DEFAULT_OUTPUT_DIR": "custom_output",
        "FLASK_DEBUG": "true"
    })
    def test_config_from_environment_variables(self):
        """Test configuration loading from environment variables."""
        config = ConfigManager.get_config()

        assert config.project_id == "custom-project"
        assert config.location == "us-west1"
        assert config.gemini_api_key == "test-key"
        assert config.upload_folder == Path("custom_uploads")
        assert config.default_output_dir == Path("custom_output")
        assert config.flask_debug is True

    @patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"})
    def test_config_with_defaults(self):
        """Test configuration uses defaults when env vars not set."""
        config = ConfigManager.get_config()

        # Should use defaults from conf.py
        assert config.gemini_api_key == "test-key"
        assert config.upload_folder == Path("uploads")
        assert config.default_output_dir == Path(".")
        assert config.flask_debug is False

    @patch.dict(os.environ, {
        "GEMINI_API_KEY": "test-key",
        "FLASK_DEBUG": "false"
    })
    def test_flask_debug_false_values(self):
        """Test Flask debug recognizes false values."""
        config = ConfigManager.get_config()
        assert config.flask_debug is False

    @patch.dict(os.environ, {
        "GEMINI_API_KEY": "test-key",
        "FLASK_DEBUG": "1"
    })
    def test_flask_debug_true_values(self):
        """Test Flask debug recognizes various true values."""
        config = ConfigManager.get_config()
        assert config.flask_debug is True

    def test_config_missing_gemini_api_key(self):
        """Test ConfigManager raises error when GEMINI_API_KEY missing."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ConfigurationError, match="GEMINI_API_KEY.*required"):
                ConfigManager.get_config()

    @patch.dict(os.environ, {
        "GEMINI_API_KEY": "test-key",
        "STORAGE_TYPE": "s3",
        "AWS_S3_BUCKET": "test-bucket",
        "AWS_S3_REGION": "us-west2"
    })
    def test_config_s3_storage_valid(self):
        """Test S3 storage configuration with valid settings."""
        ConfigManager.reset_config()
        config = ConfigManager.get_config()

        assert config.storage_type == "s3"
        assert config.s3_bucket == "test-bucket"
        assert config.s3_region == "us-west2"
        # Local directories should not be created for S3 storage

    @patch.dict(os.environ, {
        "GEMINI_API_KEY": "test-key",
        "STORAGE_TYPE": "s3"
    })
    def test_config_s3_missing_bucket(self):
        """Test S3 storage fails without bucket configuration."""
        ConfigManager.reset_config()
        with pytest.raises(ConfigurationError, match="AWS_S3_BUCKET.*required"):
            ConfigManager.get_config()

    @patch.dict(os.environ, {
        "GEMINI_API_KEY": "test-key",
        "STORAGE_TYPE": "s3",
        "AWS_S3_BUCKET": "test-bucket"
    })
    def test_config_s3_missing_region(self):
        """Test S3 storage fails without region configuration."""
        ConfigManager.reset_config()
        with pytest.raises(ConfigurationError, match="AWS_S3_REGION.*required"):
            ConfigManager.get_config()

    @patch.dict(os.environ, {
        "GEMINI_API_KEY": "test-key",
        "STORAGE_TYPE": "local"
    })
    def test_config_local_storage_default(self):
        """Test local storage configuration (default behavior)."""
        ConfigManager.reset_config()
        config = ConfigManager.get_config()

        assert config.storage_type == "local"
        assert config.s3_bucket is None
        assert config.s3_region is None
