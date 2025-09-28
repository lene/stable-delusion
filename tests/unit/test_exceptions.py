"""
Unit tests for custom exception hierarchy.
Tests exception creation, string representation, and inheritance.
"""

from stable_delusion.exceptions import (
    NanoAPIError,
    ConfigurationError,
    ImageGenerationError,
    UpscalingError,
    ValidationError,
    FileOperationError,
    APIError,
    AuthenticationError,
)


class TestNanoAPIError:
    """Test base NanoAPIError functionality."""

    def test_basic_creation(self):
        error = NanoAPIError("Test message")
        assert str(error) == "Test message"
        assert error.message == "Test message"
        assert error.details == ""

    def test_creation_with_details(self):
        error = NanoAPIError("Test message", "Additional details")
        assert str(error) == "Test message: Additional details"
        assert error.message == "Test message"
        assert error.details == "Additional details"

    def test_inheritance(self):
        error = NanoAPIError("Test")
        assert isinstance(error, Exception)


class TestConfigurationError:
    """Test ConfigurationError functionality."""

    def test_creation_without_config_key(self):
        error = ConfigurationError("Config error")
        assert str(error) == "Config error"
        assert error.config_key == ""

    def test_creation_with_config_key(self):
        error = ConfigurationError("Missing key", "API_KEY")
        assert str(error) == "Missing key: Configuration key: API_KEY"
        assert error.config_key == "API_KEY"

    def test_inheritance(self):
        error = ConfigurationError("Test")
        assert isinstance(error, NanoAPIError)


class TestImageGenerationError:
    """Test ImageGenerationError functionality."""

    def test_creation_basic(self):
        error = ImageGenerationError("Generation failed")
        assert str(error) == "Generation failed"
        assert error.prompt == ""
        assert error.api_response == ""

    def test_creation_with_prompt(self):
        error = ImageGenerationError("Failed", "test prompt")
        assert str(error) == "Failed: Prompt: test prompt"
        assert error.prompt == "test prompt"

    def test_creation_with_all_details(self):
        error = ImageGenerationError("Failed", "test prompt", "api response")
        expected = "Failed: Prompt: test prompt; API response: api response"
        assert str(error) == expected
        assert error.prompt == "test prompt"
        assert error.api_response == "api response"

    def test_inheritance(self):
        error = ImageGenerationError("Test")
        assert isinstance(error, NanoAPIError)


class TestUpscalingError:
    """Test UpscalingError functionality."""

    def test_creation_basic(self):
        error = UpscalingError("Upscaling failed")
        assert str(error) == "Upscaling failed"
        assert error.scale_factor == ""
        assert error.image_path == ""

    def test_creation_with_details(self):
        error = UpscalingError("Failed", "x4", "/path/to/image.jpg")
        expected = "Failed: Scale factor: x4; Image: /path/to/image.jpg"
        assert str(error) == expected
        assert error.scale_factor == "x4"
        assert error.image_path == "/path/to/image.jpg"

    def test_inheritance(self):
        error = UpscalingError("Test")
        assert isinstance(error, NanoAPIError)


class TestValidationError:
    """Test ValidationError functionality."""

    def test_creation_basic(self):
        error = ValidationError("Invalid input")
        assert str(error) == "Invalid input"
        assert error.field == ""
        assert error.value == ""

    def test_creation_with_field_and_value(self):
        error = ValidationError("Invalid scale", "scale", "10")
        expected = "Invalid scale: Field: scale; Value: 10"
        assert str(error) == expected
        assert error.field == "scale"
        assert error.value == "10"

    def test_inheritance(self):
        error = ValidationError("Test")
        assert isinstance(error, NanoAPIError)


class TestFileOperationError:
    """Test FileOperationError functionality."""

    def test_creation_basic(self):
        error = FileOperationError("File error")
        assert str(error) == "File error"
        assert error.file_path == ""
        assert error.operation == ""

    def test_creation_with_details(self):
        error = FileOperationError("Cannot read file", "/path/file.txt", "read")
        expected = "Cannot read file: Operation: read; File: /path/file.txt"
        assert str(error) == expected
        assert error.file_path == "/path/file.txt"
        assert error.operation == "read"

    def test_inheritance(self):
        error = FileOperationError("Test")
        assert isinstance(error, NanoAPIError)


class TestAPIError:
    """Test APIError functionality."""

    def test_creation_basic(self):
        error = APIError("API failed")
        assert str(error) == "API failed"
        assert error.status_code == 0
        assert error.response_body == ""

    def test_creation_with_details(self):
        error = APIError("Request failed", 404, "Not found")
        expected = "Request failed: Status code: 404; Response: Not found"
        assert str(error) == expected
        assert error.status_code == 404
        assert error.response_body == "Not found"

    def test_inheritance(self):
        error = APIError("Test")
        assert isinstance(error, NanoAPIError)


class TestAuthenticationError:
    """Test AuthenticationError functionality."""

    def test_creation_default(self):
        error = AuthenticationError()
        assert str(error) == "Authentication failed: Status code: 401"
        assert error.status_code == 401

    def test_creation_custom_message(self):
        error = AuthenticationError("Custom auth error")
        assert str(error) == "Custom auth error: Status code: 401"
        assert error.status_code == 401

    def test_inheritance(self):
        error = AuthenticationError()
        assert isinstance(error, APIError)
        assert isinstance(error, NanoAPIError)


class TestExceptionChaining:
    """Test exception chaining and context preservation."""

    def test_exception_chaining(self):
        try:
            try:
                raise ValueError("Original error")
            except ValueError as e:
                raise ValidationError("Validation failed", "field", "value") from e
        except ValidationError as validation_error:
            assert validation_error.__cause__ is not None
            assert isinstance(validation_error.__cause__, ValueError)
            assert str(validation_error.__cause__) == "Original error"
