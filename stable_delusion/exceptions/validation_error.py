"""Input validation exception class."""

__author__ = "Lene Preuss <lene.preuss@gmail.com>"

from stable_delusion.exceptions.nano_api_error import NanoAPIError


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
