"""Unit tests for metadata model and functionality."""

from nano_api.models.metadata import GenerationMetadata


class TestGenerationMetadata:
    """Test cases for GenerationMetadata model."""

    def test_metadata_creation_with_defaults(self):
        """Test creating metadata with default values."""
        metadata = GenerationMetadata(
            prompt="Test prompt",
            images=["image1.jpg", "image2.jpg"],
            generated_image="output.png"
        )

        assert metadata.prompt == "Test prompt"
        assert metadata.images == ["image1.jpg", "image2.jpg"]
        assert metadata.generated_image == "output.png"
        assert metadata.gcp_project_id is None
        assert metadata.gcp_location is None
        assert metadata.scale is None
        assert metadata.timestamp is not None
        assert metadata.content_hash is not None

    def test_metadata_creation_with_full_params(self):
        """Test creating metadata with all parameters."""
        timestamp = "2024-01-01T12:00:00Z"
        metadata = GenerationMetadata(
            prompt="Full test prompt",
            images=["s3://bucket/image1.jpg"],
            generated_image="s3://bucket/output.png",
            gcp_project_id="test-project",
            gcp_location="us-central1",
            scale=4,
            timestamp=timestamp
        )

        assert metadata.prompt == "Full test prompt"
        assert metadata.gcp_project_id == "test-project"
        assert metadata.gcp_location == "us-central1"
        assert metadata.scale == 4
        assert metadata.timestamp == timestamp
        assert metadata.content_hash is not None

    def test_content_hash_consistency(self):
        """Test that content hash is consistent for same inputs."""
        metadata1 = GenerationMetadata(
            prompt="Same prompt",
            images=["image1.jpg", "image2.jpg"],
            generated_image="output.png",
            gcp_project_id="project",
            scale=2
        )

        metadata2 = GenerationMetadata(
            prompt="Same prompt",
            images=["image1.jpg", "image2.jpg"],
            generated_image="different_output.png",  # This shouldn't affect hash
            gcp_project_id="project",
            scale=2
        )

        assert metadata1.content_hash == metadata2.content_hash

    def test_content_hash_different_for_different_inputs(self):
        """Test that different inputs produce different hashes."""
        metadata1 = GenerationMetadata(
            prompt="Prompt 1",
            images=["image1.jpg"],
            generated_image="output.png"
        )

        metadata2 = GenerationMetadata(
            prompt="Prompt 2",  # Different prompt
            images=["image1.jpg"],
            generated_image="output.png"
        )

        assert metadata1.content_hash != metadata2.content_hash

    def test_content_hash_image_order_independence(self):
        """Test that image order doesn't affect content hash."""
        metadata1 = GenerationMetadata(
            prompt="Test",
            images=["image1.jpg", "image2.jpg"],
            generated_image="output.png"
        )

        metadata2 = GenerationMetadata(
            prompt="Test",
            images=["image2.jpg", "image1.jpg"],  # Different order
            generated_image="output.png"
        )

        assert metadata1.content_hash == metadata2.content_hash

    def test_to_json_and_from_json(self):
        """Test JSON serialization and deserialization."""
        original = GenerationMetadata(
            prompt="JSON test",
            images=["image1.jpg", "image2.jpg"],
            generated_image="output.png",
            gcp_project_id="test-project",
            scale=2
        )

        # Serialize to JSON
        json_str = original.to_json()
        assert isinstance(json_str, str)

        # Deserialize from JSON
        restored = GenerationMetadata.from_json(json_str)

        assert restored.prompt == original.prompt
        assert restored.images == original.images
        assert restored.generated_image == original.generated_image
        assert restored.gcp_project_id == original.gcp_project_id
        assert restored.scale == original.scale
        assert restored.content_hash == original.content_hash

    def test_to_dict_and_from_dict(self):
        """Test dictionary conversion."""
        original = GenerationMetadata(
            prompt="Dict test",
            images=["image1.jpg"],
            generated_image="output.png",
            timestamp="2024-01-01T12:00:00Z"
        )

        # Convert to dict
        data_dict = original.to_dict()
        assert isinstance(data_dict, dict)
        assert data_dict['prompt'] == "Dict test"
        assert data_dict['timestamp'] == "2024-01-01T12:00:00Z"

        # Create from dict
        restored = GenerationMetadata.from_dict(data_dict)
        assert restored.prompt == original.prompt
        assert restored.timestamp == original.timestamp
        assert restored.content_hash == original.content_hash

    def test_metadata_filename_generation(self):
        """Test metadata filename generation."""
        metadata = GenerationMetadata(
            prompt="Filename test",
            images=["image.jpg"],
            generated_image="output.png",
            timestamp="2024-01-01T12:30:45Z"
        )

        filename = metadata.get_metadata_filename()

        # Should follow format: metadata_{hash_prefix}_{date}.json
        assert filename.startswith("metadata_")
        assert filename.endswith(".json")
        assert "20240101_123045" in filename

        # Hash prefix should be 8 characters
        parts = filename.replace(".json", "").split("_")
        assert len(parts) >= 3  # metadata, hash, date parts (may be split by underscores)
        assert len(parts[1]) == 8  # hash prefix length

    def test_metadata_filename_with_invalid_timestamp(self):
        """Test filename generation with invalid timestamp."""
        metadata = GenerationMetadata(
            prompt="Invalid timestamp test",
            images=["image.jpg"],
            generated_image="output.png",
            timestamp="invalid-timestamp"
        )

        filename = metadata.get_metadata_filename()

        # Should handle invalid timestamp gracefully
        assert "unknown" in filename
        assert filename.endswith(".json")
