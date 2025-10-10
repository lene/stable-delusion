"""Configuration-related exception class."""

__author__ = "Lene Preuss <lene.preuss@gmail.com>"

from stable_delusion.exceptions.nano_api_error import NanoAPIError


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
