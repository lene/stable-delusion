"""Unit tests for metadata repository implementations."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from nano_api.config import Config
from nano_api.exceptions import FileOperationError
from nano_api.models.metadata import GenerationMetadata
from nano_api.repositories.local_metadata_repository import LocalMetadataRepository
from nano_api.repositories.s3_metadata_repository import S3MetadataRepository


class TestLocalMetadataRepository:
    """Test cases for LocalMetadataRepository."""

    @pytest.fixture
    def temp_dir(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    @pytest.fixture
    def config(self, temp_dir):
        return Config(
            project_id="test-project",
            location="us-central1",
            gemini_api_key="test-key",
            upload_folder=temp_dir / "uploads",
            default_output_dir=temp_dir,
            flask_debug=False,
            storage_type="local",
            s3_bucket=None,
            s3_region=None,
            aws_access_key_id=None,
            aws_secret_access_key=None,
        )

    @pytest.fixture
    def local_repo(self, config):
        return LocalMetadataRepository(config)

    @pytest.fixture
    def sample_metadata(self):
        return GenerationMetadata(
            prompt="Test prompt",
            images=["image1.jpg", "image2.jpg"],
            generated_image="output.png",
            gcp_project_id="test-project",
            scale=2,
        )

    def test_save_and_load_metadata(self, local_repo, sample_metadata):
        # Save metadata
        saved_path = local_repo.save_metadata(sample_metadata)
        assert Path(saved_path).exists()

        # Load metadata
        loaded_metadata = local_repo.load_metadata(saved_path)
        assert loaded_metadata.prompt == sample_metadata.prompt
        assert loaded_metadata.images == sample_metadata.images
        assert loaded_metadata.content_hash == sample_metadata.content_hash

    def test_metadata_exists(self, local_repo, sample_metadata):
        # Initially should not exist
        assert local_repo.metadata_exists(sample_metadata.content_hash) is None

        # Save metadata
        local_repo.save_metadata(sample_metadata)

        # Should now exist
        existing_path = local_repo.metadata_exists(sample_metadata.content_hash)
        assert existing_path is not None
        assert Path(existing_path).exists()

    def test_list_metadata_by_hash_prefix(self, local_repo, sample_metadata):
        # Save metadata
        local_repo.save_metadata(sample_metadata)

        # List by hash prefix
        hash_prefix = sample_metadata.content_hash[:4]
        matching_files = local_repo.list_metadata_by_hash_prefix(hash_prefix)

        assert len(matching_files) >= 1
        assert any(hash_prefix in str(f) for f in matching_files)

    def test_load_nonexistent_metadata(self, local_repo):
        with pytest.raises(FileOperationError):
            local_repo.load_metadata("nonexistent.json")


class TestS3MetadataRepository:
    """Test cases for S3MetadataRepository."""

    @pytest.fixture
    def config(self):
        return Config(
            project_id="test-project",
            location="us-central1",
            gemini_api_key="test-key",
            upload_folder=Path("uploads"),
            default_output_dir=Path("."),
            flask_debug=False,
            storage_type="s3",
            s3_bucket="test-bucket",
            s3_region="us-east-1",
            aws_access_key_id="test-access-key",
            aws_secret_access_key="test-secret-key",
        )

    @pytest.fixture
    def mock_s3_client(self):
        return MagicMock()

    @pytest.fixture
    def s3_repo(self, config, mock_s3_client):
        with patch(
            "nano_api.repositories.s3_metadata_repository.S3ClientManager.create_s3_client",
            return_value=mock_s3_client,
        ):
            repo = S3MetadataRepository(config)
            repo.s3_client = mock_s3_client
            return repo

    @pytest.fixture
    def sample_metadata(self):
        return GenerationMetadata(
            prompt="S3 test prompt",
            images=["s3://bucket/image1.jpg"],
            generated_image="s3://bucket/output.png",
            gcp_project_id="test-project",
        )

    def test_save_metadata(self, s3_repo, mock_s3_client, sample_metadata):
        # Mock successful put_object
        mock_s3_client.put_object.return_value = {}

        # Save metadata
        s3_key = s3_repo.save_metadata(sample_metadata)

        # Verify S3 call
        mock_s3_client.put_object.assert_called_once()
        call_args = mock_s3_client.put_object.call_args
        assert call_args[1]["Bucket"] == "test-bucket"
        assert call_args[1]["Key"].startswith("metadata/")
        assert s3_key == call_args[1]["Key"]
        assert call_args[1]["ACL"] == "public-read"
        assert call_args[1]["ContentType"] == "application/json"

        # Verify metadata content
        json_content = call_args[1]["Body"].decode("utf-8")
        restored_metadata = GenerationMetadata.from_json(json_content)
        assert restored_metadata.prompt == sample_metadata.prompt

    def test_load_metadata(self, s3_repo, mock_s3_client, sample_metadata):
        # Mock S3 response
        mock_response = {"Body": MagicMock()}
        mock_response["Body"].read.return_value = sample_metadata.to_json().encode("utf-8")
        mock_s3_client.get_object.return_value = mock_response

        # Load metadata
        loaded_metadata = s3_repo.load_metadata("metadata/test_key.json")

        # Verify result
        assert loaded_metadata.prompt == sample_metadata.prompt
        assert loaded_metadata.content_hash == sample_metadata.content_hash

        # Verify S3 call
        mock_s3_client.get_object.assert_called_once_with(
            Bucket="test-bucket", Key="metadata/test_key.json"
        )

    def test_metadata_exists(self, s3_repo, mock_s3_client, sample_metadata):
        # Mock list_objects_v2 response
        mock_s3_client.list_objects_v2.return_value = {
            "Contents": [{"Key": f"metadata/metadata_{sample_metadata.content_hash[:8]}_test.json"}]
        }

        # Mock get_object for verification
        mock_response = {"Body": MagicMock()}
        mock_response["Body"].read.return_value = sample_metadata.to_json().encode("utf-8")
        mock_s3_client.get_object.return_value = mock_response

        # Check existence
        existing_key = s3_repo.metadata_exists(sample_metadata.content_hash)

        assert existing_key is not None
        assert sample_metadata.content_hash[:8] in existing_key

    def test_metadata_does_not_exist(self, s3_repo, mock_s3_client):
        # Mock empty list_objects_v2 response
        mock_s3_client.list_objects_v2.return_value = {}

        # Check existence
        existing_key = s3_repo.metadata_exists("nonexistent_hash")

        assert existing_key is None

    def test_list_metadata_by_hash_prefix(self, s3_repo, mock_s3_client):
        # Mock paginator
        mock_paginator = MagicMock()
        mock_s3_client.get_paginator.return_value = mock_paginator

        # Mock pages
        mock_pages = [
            {
                "Contents": [
                    {"Key": "metadata/metadata_abc12345_test1.json"},
                    {"Key": "metadata/metadata_abc67890_test2.json"},
                    {"Key": "metadata/metadata_def12345_test3.json"},
                ]
            }
        ]
        mock_paginator.paginate.return_value = mock_pages

        # List metadata
        matching_keys = s3_repo.list_metadata_by_hash_prefix("abc")

        # Should find 2 matching keys
        assert len(matching_keys) == 2
        assert all("metadata_abc" in key for key in matching_keys)

    def test_save_metadata_s3_error(self, s3_repo, mock_s3_client, sample_metadata):
        from botocore.exceptions import ClientError

        # Mock S3 error
        mock_s3_client.put_object.side_effect = ClientError(
            {"Error": {"Code": "AccessDenied"}}, "PutObject"
        )

        # Should raise FileOperationError
        with pytest.raises(FileOperationError):
            s3_repo.save_metadata(sample_metadata)
