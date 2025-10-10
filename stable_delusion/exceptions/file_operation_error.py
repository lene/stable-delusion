"""File operation exception class."""

__author__ = "Lene Preuss <lene.preuss@gmail.com>"

from stable_delusion.exceptions.nano_api_error import NanoAPIError


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
