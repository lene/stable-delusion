"""Image upscaling exception class."""

__author__ = "Lene Preuss <lene.preuss@gmail.com>"

from stable_delusion.exceptions.nano_api_error import NanoAPIError


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
