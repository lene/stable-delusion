"""Base exception class for NanoAPI-related errors."""

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
