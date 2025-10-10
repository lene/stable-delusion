"""Image generation exception class."""

__author__ = "Lene Preuss <lene.preuss@gmail.com>"

from stable_delusion.exceptions.nano_api_error import NanoAPIError


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
