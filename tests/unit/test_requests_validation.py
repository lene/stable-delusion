"""
Unit tests for request validation, specifically Seedream + S3 storage requirements.
Tests the validation logic in GenerateImageRequest.__post_init__().
"""

__author__ = "Lene Preuss <lene.preuss@gmail.com>"

from pathlib import Path

import pytest

from nano_api.models.requests import GenerateImageRequest
from nano_api.exceptions import ValidationError


class TestGenerateImageRequestValidation:  # pylint: disable=too-many-public-methods
    """Test validation rules for GenerateImageRequest."""

    def test_seedream_with_images_requires_s3_storage(self):
        with pytest.raises(ValidationError) as exc_info:
            GenerateImageRequest(
                prompt="Edit this image",
                images=[Path("test.jpg")],
                model="seedream",
                storage_type="local",
            )

        assert "Seedream model with input images requires S3 storage type" in str(exc_info.value)
        assert exc_info.value.field == "storage_type"
        assert exc_info.value.value == "local"

    def test_seedream_with_images_requires_s3_storage_none_type(self):
        with pytest.raises(ValidationError) as exc_info:
            GenerateImageRequest(
                prompt="Edit this image",
                images=[Path("test.jpg")],
                model="seedream",
                storage_type=None,
            )

        assert "Seedream model with input images requires S3 storage type" in str(exc_info.value)
        assert exc_info.value.field == "storage_type"
        assert exc_info.value.value == "None"

    def test_seedream_with_images_allows_s3_storage(self):
        # Should not raise any exception
        request = GenerateImageRequest(
            prompt="Edit this image", images=[Path("test.jpg")], model="seedream", storage_type="s3"
        )
        assert request.model == "seedream"
        assert request.storage_type == "s3"
        assert len(request.images) == 1

    def test_seedream_without_images_allows_local_storage(self):
        # Should not raise any exception
        request = GenerateImageRequest(
            prompt="Generate a beautiful sunset",
            images=[],  # No input images
            model="seedream",
            storage_type="local",
        )
        assert request.model == "seedream"
        assert request.storage_type == "local"
        assert len(request.images) == 0

    def test_seedream_without_images_allows_none_storage(self):
        # Should not raise any exception
        request = GenerateImageRequest(
            prompt="Generate a beautiful sunset",
            images=[],  # No input images
            model="seedream",
            storage_type=None,
        )
        assert request.model == "seedream"
        assert request.storage_type is None
        assert len(request.images) == 0

    def test_gemini_with_images_allows_local_storage(self):
        # Should not raise any exception
        request = GenerateImageRequest(
            prompt="Edit this image",
            images=[Path("test.jpg")],
            model="gemini",
            storage_type="local",
        )
        assert request.model == "gemini"
        assert request.storage_type == "local"
        assert len(request.images) == 1

    def test_gemini_with_images_allows_s3_storage(self):
        # Should not raise any exception
        request = GenerateImageRequest(
            prompt="Edit this image", images=[Path("test.jpg")], model="gemini", storage_type="s3"
        )
        assert request.model == "gemini"
        assert request.storage_type == "s3"
        assert len(request.images) == 1

    def test_none_model_with_images_allows_local_storage(self):
        # Should not raise any exception
        request = GenerateImageRequest(
            prompt="Edit this image",
            images=[Path("test.jpg")],
            model=None,  # Default model
            storage_type="local",
        )
        assert request.model is None
        assert request.storage_type == "local"
        assert len(request.images) == 1

    def test_empty_prompt_validation_still_works(self):
        with pytest.raises(ValidationError) as exc_info:
            GenerateImageRequest(
                prompt="",  # Empty prompt
                images=[Path("test.jpg")],
                model="seedream",
                storage_type="s3",
            )

        assert "Prompt cannot be empty" in str(exc_info.value)
        assert exc_info.value.field == "prompt"

    def test_invalid_model_validation_still_works(self):
        with pytest.raises(ValidationError) as exc_info:
            GenerateImageRequest(
                prompt="Edit this image",
                images=[Path("test.jpg")],
                model="invalid_model",
                storage_type="s3",
            )

        assert "Model must be one of" in str(exc_info.value)
        assert exc_info.value.field == "model"
        assert exc_info.value.value == "invalid_model"

    def test_invalid_storage_type_validation_still_works(self):
        with pytest.raises(ValidationError) as exc_info:
            GenerateImageRequest(
                prompt="Edit this image",
                images=[Path("test.jpg")],
                model="gemini",
                storage_type="invalid_storage",
            )

        assert "Storage type must be 'local' or 's3'" in str(exc_info.value)
        assert exc_info.value.field == "storage_type"
        assert exc_info.value.value == "invalid_storage"

    def test_seedream_with_multiple_images_requires_s3(self):
        with pytest.raises(ValidationError) as exc_info:
            GenerateImageRequest(
                prompt="Edit these images",
                images=[Path("test1.jpg"), Path("test2.jpg"), Path("test3.jpg")],
                model="seedream",
                storage_type="local",
            )

        assert "Seedream model with input images requires S3 storage type" in str(exc_info.value)
        assert exc_info.value.field == "storage_type"
        assert exc_info.value.value == "local"

    def test_scale_and_image_size_mutual_exclusivity(self):
        with pytest.raises(ValidationError) as exc_info:
            GenerateImageRequest(
                prompt="Test prompt",
                images=[Path("test.jpg")],
                model="gemini",
                scale=2,
                image_size="2K",
                storage_type="local",
            )

        assert "Scale and image_size are mutually exclusive" in str(exc_info.value)
        assert exc_info.value.field == "scale"

    def test_scale_only_for_gemini_model(self):
        with pytest.raises(ValidationError) as exc_info:
            GenerateImageRequest(
                prompt="Test prompt",
                images=[Path("test.jpg")],
                model="seedream",
                scale=2,
                storage_type="s3",
            )

        assert "Scale parameter is only available for Gemini model" in str(exc_info.value)
        assert exc_info.value.field == "scale"
        assert exc_info.value.value == "2"

    def test_image_size_only_for_seedream_model(self):
        with pytest.raises(ValidationError) as exc_info:
            GenerateImageRequest(
                prompt="Test prompt",
                images=[Path("test.jpg")],
                model="gemini",
                image_size="2K",
                storage_type="local",
            )

        assert "Image size parameter is only available for Seedream model" in str(exc_info.value)
        assert exc_info.value.field == "image_size"
        assert exc_info.value.value == "2K"

    def test_scale_with_gemini_model_valid(self):
        # Should not raise any exception
        request = GenerateImageRequest(
            prompt="Test prompt",
            images=[Path("test.jpg")],
            model="gemini",
            scale=4,
            storage_type="local",
        )
        assert request.model == "gemini"
        assert request.scale == 4
        assert request.image_size is None

    def test_image_size_with_seedream_model_valid(self):
        # Should not raise any exception
        request = GenerateImageRequest(
            prompt="Test prompt",
            images=[Path("test.jpg")],
            model="seedream",
            image_size="4K",
            storage_type="s3",
        )
        assert request.model == "seedream"
        assert request.image_size == "4K"
        assert request.scale is None

    def test_image_size_validation_predefined_sizes_valid(self):
        # Test all valid predefined sizes
        for size in ["1K", "2K", "4K"]:
            request = GenerateImageRequest(
                prompt="Test prompt",
                images=[Path("test.jpg")],
                model="seedream",
                image_size=size,
                storage_type="s3",
            )
            assert request.image_size == size

    def test_image_size_validation_custom_dimensions_valid(self):
        # Test valid custom dimensions
        valid_sizes = [
            "1280x720",  # Minimum dimensions
            "1920x1080",  # Common HD
            "2560x1440",  # Common QHD
            "3840x2160",  # 4K
            "4096x4096",  # Maximum dimensions
        ]
        for size in valid_sizes:
            request = GenerateImageRequest(
                prompt="Test prompt",
                images=[Path("test.jpg")],
                model="seedream",
                image_size=size,
                storage_type="s3",
            )
            assert request.image_size == size

    def test_image_size_validation_invalid_predefined_sizes(self):
        invalid_sizes = ["0K", "5K", "8K", "1k", "2k", "4k"]  # Case sensitive and invalid values
        for size in invalid_sizes:
            with pytest.raises(ValidationError) as exc_info:
                GenerateImageRequest(
                    prompt="Test prompt",
                    images=[Path("test.jpg")],
                    model="seedream",
                    image_size=size,
                    storage_type="s3",
                )
            assert "Image size must be" in str(exc_info.value)
            assert exc_info.value.field == "image_size"

    def test_image_size_validation_invalid_custom_dimensions(self):
        invalid_sizes = [
            "1279x720",  # Width too small
            "1280x719",  # Height too small
            "4097x1080",  # Width too large
            "1920x4097",  # Height too large
            "1280",  # Missing height
            "x720",  # Missing width
            "1920x",  # Missing height
            "1920by1080",  # Wrong separator
            "1920*1080",  # Wrong separator
            "abc x 720",  # Non-numeric
            "1920 x 1080",  # Spaces
            "1920.5x1080",  # Decimals not allowed
        ]
        for size in invalid_sizes:
            with pytest.raises(ValidationError) as exc_info:
                GenerateImageRequest(
                    prompt="Test prompt",
                    images=[Path("test.jpg")],
                    model="seedream",
                    image_size=size,
                    storage_type="s3",
                )
            assert "Image size must be" in str(exc_info.value)
            assert exc_info.value.field == "image_size"

    def test_image_size_validation_empty_and_none(self):
        # Empty string should fail
        with pytest.raises(ValidationError) as exc_info:
            GenerateImageRequest(
                prompt="Test prompt",
                images=[Path("test.jpg")],
                model="seedream",
                image_size="",
                storage_type="s3",
            )
        assert "Image size must be" in str(exc_info.value)

        # None should be allowed (uses default)
        request = GenerateImageRequest(
            prompt="Test prompt",
            images=[Path("test.jpg")],
            model="seedream",
            image_size=None,
            storage_type="s3",
        )
        assert request.image_size is None
