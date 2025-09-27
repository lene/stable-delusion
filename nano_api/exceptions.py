"""
Custom exception hierarchy for NanoAPIClient.
Provides specific exception types for different error scenarios.
"""

__author__ = "Lene Preuss <lene.preuss@gmail.com>"


class NanoAPIError(Exception):
    """Base exception for all NanoAPI-related errors."""

    def __init__(self, message: str, details: str = "") -> None:
        """
        Initialize NanoAPI exception.

        Args:
            message: The main error message
            details: Additional error details
        """
        self.message = message
        self.details = details
        super().__init__(self.message)

    def __str__(self) -> str:
        if self.details:
            return f"{self.message}: {self.details}"
        return self.message


class ConfigurationError(NanoAPIError):
    """Exception raised for configuration-related errors."""

    def __init__(self, message: str, config_key: str = "") -> None:
        """
        Initialize configuration error.

        Args:
            message: The main error message
            config_key: The configuration key that caused the error
        """
        self.config_key = config_key
        details = f"Configuration key: {config_key}" if config_key else ""
        super().__init__(message, details)


class ImageGenerationError(NanoAPIError):
    """Exception raised when image generation fails."""

    def __init__(self, message: str, prompt: str = "", api_response: str = "") -> None:
        """
        Initialize image generation error.

        Args:
            message: The main error message
            prompt: The prompt that failed
            api_response: The API response details
        """
        self.prompt = prompt
        self.api_response = api_response
        details_parts = []
        if prompt:
            details_parts.append(f"Prompt: {prompt}")
        if api_response:
            details_parts.append(f"API response: {api_response}")
        details = "; ".join(details_parts)
        super().__init__(message, details)


class UpscalingError(NanoAPIError):
    """Exception raised when image upscaling fails."""

    def __init__(self, message: str, scale_factor: str = "", image_path: str = "") -> None:
        """
        Initialize upscaling error.

        Args:
            message: The main error message
            scale_factor: The scale factor that failed
            image_path: Path to the image that failed to upscale
        """
        self.scale_factor = scale_factor
        self.image_path = image_path
        details_parts = []
        if scale_factor:
            details_parts.append(f"Scale factor: {scale_factor}")
        if image_path:
            details_parts.append(f"Image: {image_path}")
        details = "; ".join(details_parts)
        super().__init__(message, details)


class ValidationError(NanoAPIError):
    """Exception raised for input validation errors."""

    def __init__(self, message: str, field: str = "", value: str = "") -> None:
        """
        Initialize validation error.

        Args:
            message: The main error message
            field: The field that failed validation
            value: The invalid value
        """
        self.field = field
        self.value = value
        details_parts = []
        if field:
            details_parts.append(f"Field: {field}")
        if value:
            details_parts.append(f"Value: {value}")
        details = "; ".join(details_parts)
        super().__init__(message, details)


class FileOperationError(NanoAPIError):
    """Exception raised for file operation errors."""

    def __init__(self, message: str, file_path: str = "", operation: str = "") -> None:
        """
        Initialize file operation error.

        Args:
            message: The main error message
            file_path: The file path that caused the error
            operation: The operation that failed (read, write, delete, etc.)
        """
        self.file_path = file_path
        self.operation = operation
        details_parts = []
        if operation:
            details_parts.append(f"Operation: {operation}")
        if file_path:
            details_parts.append(f"File: {file_path}")
        details = "; ".join(details_parts)
        super().__init__(message, details)


class APIError(NanoAPIError):
    """Exception raised for external API errors."""

    def __init__(self, message: str, status_code: int = 0, response_body: str = "") -> None:
        """
        Initialize API error.

        Args:
            message: The main error message
            status_code: HTTP status code from the API
            response_body: Response body from the API
        """
        self.status_code = status_code
        self.response_body = response_body
        details_parts = []
        if status_code:
            details_parts.append(f"Status code: {status_code}")
        if response_body:
            details_parts.append(f"Response: {response_body}")
        details = "; ".join(details_parts)
        super().__init__(message, details)


class AuthenticationError(APIError):
    """Exception raised for authentication errors with external APIs."""

    def __init__(self, message: str = "Authentication failed") -> None:
        """
        Initialize authentication error.

        Args:
            message: The main error message
        """
        super().__init__(message, status_code=401)
