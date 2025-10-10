"""External API exception class."""

__author__ = "Lene Preuss <lene.preuss@gmail.com>"

from stable_delusion.exceptions.nano_api_error import NanoAPIError


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
