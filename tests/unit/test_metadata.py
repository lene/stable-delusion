"""Unit tests for metadata model and functionality."""

from stable_delusion.models.metadata import GenerationMetadata


class TestGenerationMetadata:
    """Test cases for GenerationMetadata model."""

    def test_metadata_creation_with_defaults(self):
        metadata = GenerationMetadata(
            prompt="Test prompt", images=["image1.jpg", "image2.jpg"], generated_image="output.png"
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
        timestamp = "2024-01-01T12:00:00Z"
        metadata = GenerationMetadata(
            prompt="Full test prompt",
            images=["s3://bucket/image1.jpg"],
            generated_image="s3://bucket/output.png",
            gcp_project_id="test-project",
            gcp_location="us-central1",
            scale=4,
            model="gemini-2.5-flash-image-preview",
            timestamp=timestamp,
        )

        assert metadata.prompt == "Full test prompt"
        assert metadata.gcp_project_id == "test-project"
        assert metadata.gcp_location == "us-central1"
        assert metadata.scale == 4
        assert metadata.model == "gemini-2.5-flash-image-preview"
        assert metadata.timestamp == timestamp
        assert metadata.content_hash is not None

    def test_content_hash_consistency(self):
        metadata1 = GenerationMetadata(
            prompt="Same prompt",
            images=["image1.jpg", "image2.jpg"],
            generated_image="output.png",
            gcp_project_id="project",
            scale=2,
        )

        metadata2 = GenerationMetadata(
            prompt="Same prompt",
            images=["image1.jpg", "image2.jpg"],
            generated_image="different_output.png",  # This shouldn't affect hash
            gcp_project_id="project",
            scale=2,
        )

        assert metadata1.content_hash == metadata2.content_hash

    def test_content_hash_different_for_different_inputs(self):
        metadata1 = GenerationMetadata(
            prompt="Prompt 1", images=["image1.jpg"], generated_image="output.png"
        )

        metadata2 = GenerationMetadata(
            prompt="Prompt 2",  # Different prompt
            images=["image1.jpg"],
            generated_image="output.png",
        )

        assert metadata1.content_hash != metadata2.content_hash

    def test_content_hash_image_order_independence(self):
        metadata1 = GenerationMetadata(
            prompt="Test", images=["image1.jpg", "image2.jpg"], generated_image="output.png"
        )

        metadata2 = GenerationMetadata(
            prompt="Test",
            images=["image2.jpg", "image1.jpg"],  # Different order
            generated_image="output.png",
        )

        assert metadata1.content_hash == metadata2.content_hash

    def test_to_json_and_from_json(self):
        original = GenerationMetadata(
            prompt="JSON test",
            images=["image1.jpg", "image2.jpg"],
            generated_image="output.png",
            gcp_project_id="test-project",
            scale=2,
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
        original = GenerationMetadata(
            prompt="Dict test",
            images=["image1.jpg"],
            generated_image="output.png",
            timestamp="2024-01-01T12:00:00Z",
        )

        # Convert to dict
        data_dict = original.to_dict()
        assert isinstance(data_dict, dict)
        assert data_dict["prompt"] == "Dict test"
        assert data_dict["timestamp"] == "2024-01-01T12:00:00Z"

        # Create from dict
        restored = GenerationMetadata.from_dict(data_dict)
        assert restored.prompt == original.prompt
        assert restored.timestamp == original.timestamp
        assert restored.content_hash == original.content_hash

    def test_metadata_filename_generation(self):
        metadata = GenerationMetadata(
            prompt="Filename test",
            images=["image.jpg"],
            generated_image="output.png",
            timestamp="2024-01-01T12:30:45Z",
        )

        filename = metadata.get_metadata_filename()

        # Should follow format: metadata_{date}.json (no hash prefix)
        assert filename.startswith("metadata_")
        assert filename.endswith(".json")
        assert "20240101_123045" in filename
        # Verify format is metadata_YYYYMMDD_HHMMSS.json
        assert filename == "metadata_20240101_123045.json"

    def test_metadata_filename_with_invalid_timestamp(self):
        metadata = GenerationMetadata(
            prompt="Invalid timestamp test",
            images=["image.jpg"],
            generated_image="output.png",
            timestamp="invalid-timestamp",
        )

        filename = metadata.get_metadata_filename()

        # Should handle invalid timestamp gracefully
        assert "unknown" in filename
        assert filename.endswith(".json")

    def test_metadata_with_api_details(self):
        """Test that API request details are properly stored and included in hash."""
        api_params = {
            "prompt": "Test prompt",
            "images": ["image1.jpg"],
            "model": "test-model",
            "temperature": 0.7,
        }

        metadata = GenerationMetadata(
            prompt="Test prompt",
            images=["image1.jpg"],
            generated_image="output.png",
            api_endpoint="https://api.example.com/v1/generate",
            api_model="test-model-v1.0",
            api_params=api_params,
        )

        assert metadata.api_endpoint == "https://api.example.com/v1/generate"
        assert metadata.api_model == "test-model-v1.0"
        assert metadata.api_params == api_params
        assert metadata.content_hash is not None

    def test_content_hash_includes_api_details(self):
        """Test that content hash changes when API details change."""
        metadata1 = GenerationMetadata(
            prompt="Same prompt",
            images=["image1.jpg"],
            generated_image="output.png",
            api_endpoint="https://api.example.com/v1/generate",
            api_model="model-v1",
            api_params={"temperature": 0.5},
        )

        metadata2 = GenerationMetadata(
            prompt="Same prompt",
            images=["image1.jpg"],
            generated_image="output.png",
            api_endpoint="https://api.example.com/v1/generate",
            api_model="model-v1",
            api_params={"temperature": 0.7},  # Different parameter
        )

        # Hash should be different because API params differ
        assert metadata1.content_hash != metadata2.content_hash

    def test_api_details_serialization(self):
        """Test that API details are preserved through JSON serialization."""
        api_params = {"model": "test-model", "temperature": 0.8, "max_tokens": 100}

        original = GenerationMetadata(
            prompt="Test",
            images=["image.jpg"],
            generated_image="output.png",
            api_endpoint="https://api.test.com/generate",
            api_model="test-model-v2",
            api_params=api_params,
        )

        # Serialize and deserialize
        json_str = original.to_json()
        restored = GenerationMetadata.from_json(json_str)

        assert restored.api_endpoint == original.api_endpoint
        assert restored.api_model == original.api_model
        assert restored.api_params == original.api_params
