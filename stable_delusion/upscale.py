"""
Image upscaling functionality using Google Vertex AI Imagen model.
Provides 2x and 4x upscaling capabilities for generated images.
Supports both CLI usage and programmatic integration.
"""

__author__ = "Lene Preuss <lene.preuss@gmail.com>"

import argparse
import base64
import io
from pathlib import Path
from typing import Dict, Any

import requests
from google.auth import default
from google.auth.transport.requests import Request
from PIL import Image

from stable_delusion.config import ConfigManager
from stable_delusion.exceptions import UpscalingError, APIError, AuthenticationError


def _get_authenticated_headers() -> Dict[str, str]:
    credentials, _ = default()
    auth_req = Request()
    credentials.refresh(auth_req)
    return {"Authorization": f"Bearer {credentials.token}", "Content-Type": "application/json"}


def _build_upscale_url(project_id: str, location: str) -> str:
    return (
        f"https://{location}-aiplatform.googleapis.com/v1/projects/"
        f"{project_id}/locations/{location}/publishers/google/models/"
        f"imagegeneration@002:predict"
    )


def _create_upscale_payload(base64_image: str, upscale_factor: str) -> Dict[str, Any]:
    return {
        "instances": [{"prompt": "", "image": {"bytesBase64Encoded": base64_image}}],
        "parameters": {
            "sampleCount": 1,
            "mode": "upscale",
            "upscaleConfig": {"upscaleFactor": upscale_factor},
        },
    }


def _decode_upscaled_image(response_data: Dict[str, Any]) -> Image.Image:
    upscaled_base64 = response_data["predictions"][0]["bytesBase64Encoded"]
    image_data = base64.b64decode(upscaled_base64)
    return Image.open(io.BytesIO(image_data))


def upscale_image(
    image_path: Path,
    project_id: str,
    location: str = "us-central1",
    upscale_factor: str = "x2",
) -> Image.Image:
    try:
        # Get authentication headers first (preserves original error behavior)
        headers = _get_authenticated_headers()
    except Exception as e:
        raise AuthenticationError(f"Failed to get authentication credentials: {e}") from e

    try:
        # Load and encode image to base64
        base64_image = base64.b64encode(image_path.read_bytes()).decode("utf-8")
    except (IOError, OSError) as e:
        raise UpscalingError(f"Failed to read image file: {e}", image_path=str(image_path)) from e

    try:
        # Make the API call
        response = requests.post(
            _build_upscale_url(project_id, location),
            json=_create_upscale_payload(base64_image, upscale_factor),
            headers=headers,
            timeout=60,
        )
        response.raise_for_status()
    except requests.exceptions.HTTPError as e:
        if response.status_code == 401:
            raise AuthenticationError("Authentication failed with Vertex AI") from e
        raise APIError(
            f"Upscaling API request failed: {e}",
            status_code=response.status_code,
            response_body=response.text,
        ) from e
    except requests.exceptions.RequestException as e:
        raise UpscalingError(
            f"Network error during upscaling: {e}",
            scale_factor=upscale_factor,
            image_path=str(image_path),
        ) from e

    try:
        return _decode_upscaled_image(response.json())
    except (KeyError, ValueError) as e:
        raise UpscalingError(
            f"Failed to decode upscaled image: {e}",
            scale_factor=upscale_factor,
            image_path=str(image_path),
        ) from e


# Usage example:
# upscaled_img = upscale_image("my_image.jpg", "my-project-id",
#                              upscale_factor="x4")
# upscaled_img.save("upscaled_image.png")

# --- Run the upscaling process ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upscale an image using Google Vertex AI.")
    parser.add_argument("image_path", type=Path, help="Path to the image to upscale.")
    parser.add_argument(
        "--scale",
        type=int,
        default=4,
        choices=[2, 4],
        help="Upscale factor: 2 or 4 (default: 4).",
    )
    args = parser.parse_args()

    config = ConfigManager.get_config()
    input_path = args.image_path
    upscaled_img = upscale_image(
        input_path,
        config.project_id,
        config.location,
        upscale_factor=f"x{args.scale}",
    )
    output_path = input_path.parent / f"upscaled_{input_path.name}"
    upscaled_img.save(str(output_path))
    print(f"Upscaled image saved to {output_path}")
