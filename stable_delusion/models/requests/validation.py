"""Validation utilities for request models."""

__author__ = "Lene Preuss <lene.preuss@gmail.com>"

import re


def validate_image_size(size: str) -> bool:
    """
    Validate image size parameter.

    Args:
        size: Image size specification

    Returns:
        True if valid, False otherwise
    """
    if not size:
        return False

    # Check predefined sizes
    if size in ["1K", "2K", "4K"]:
        return True

    # Check custom dimensions format: {width}x{height}
    pattern = r"^(\d+)x(\d+)$"
    match = re.match(pattern, size)
    if match:
        width = int(match.group(1))
        height = int(match.group(2))

        # Validate width and height ranges
        if 1280 <= width <= 4096 and 720 <= height <= 4096:
            return True

    return False
