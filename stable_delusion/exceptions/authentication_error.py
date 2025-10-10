"""Authentication exception class."""

__author__ = "Lene Preuss <lene.preuss@gmail.com>"

from stable_delusion.exceptions.api_error import APIError


class AuthenticationError(APIError):
    """Exception raised for authentication errors with external APIs."""

    def __init__(self, message: str = "Authentication failed") -> None:
        """
        Initialize authentication error.

        Args:
            message: The main error message
        """
        super().__init__(message, status_code=401)
