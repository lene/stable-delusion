"""Image upscaling request model."""

__author__ = "Lene Preuss <lene.preuss@gmail.com>"

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from stable_delusion.exceptions import ValidationError


@dataclass
class UpscaleImageRequest:
    """Request DTO for image upscaling."""

    image_path: Path
    scale_factor: str = "x2"
    project_id: Optional[str] = None
    location: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.image_path:
            raise ValidationError("Image path is required", field="image_path")

        valid_scales = ["x2", "x4"]
        if self.scale_factor not in valid_scales:
            raise ValidationError(
                f"Scale factor must be one of {valid_scales}",
                field="scale_factor",
                value=self.scale_factor,
            )

        # Ensure image_path is Path object if provided as string
        if isinstance(self.image_path, str):
            self.image_path = Path(self.image_path)
