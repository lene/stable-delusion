__author__ = 'Lene Preuss <lene.preuss@gmail.com>'
import base64
import logging

from google.cloud import aiplatform

import argparse

import base64
import io
import requests
from PIL import Image
from google.auth import default
from google.auth.transport.requests import Request

from conf import PROJECT_ID, LOCATION


def upscale_image(image_path: str, project_id: str, location: str = "us-central1",
                  upscale_factor: str = "x2") -> Image.Image:
    """
    Upscale an image using Google Vertex AI's built-in Imagen model

    Args:
        image_path: Path to the input image
        project_id: Your Google Cloud Project ID
        location: Region (e.g., "us-central1", "europe-west2")
        upscale_factor: "x2" or "x4" scaling factor

    Returns:
        PIL Image object of the upscaled image
    """
    # Get authentication credentials
    credentials, _ = default()
    auth_req = Request()
    credentials.refresh(auth_req)

    # Load and encode image to base64
    with open(image_path, "rb") as image_file:
        image_data = image_file.read()
        base64_image = base64.b64encode(image_data).decode('utf-8')

    # Prepare the request
    url = f"https://{location}-aiplatform.googleapis.com/v1/projects/{project_id}/locations/{location}/publishers/google/models/imagegeneration@002:predict"

    headers = {
        "Authorization": f"Bearer {credentials.token}",
        "Content-Type": "application/json"
    }

    payload = {
        "instances": [{
            "prompt": "",
            "image": {
                "bytesBase64Encoded": base64_image
            }
        }],
        "parameters": {
            "sampleCount": 1,
            "mode": "upscale",
            "upscaleConfig": {
                "upscaleFactor": upscale_factor
            }
        }
    }

    # Make the API call
    response = requests.post(url, json=payload, headers=headers)
    response.raise_for_status()

    # Extract upscaled image from response
    result = response.json()
    upscaled_base64 = result["predictions"][0]["bytesBase64Encoded"]

    # Convert back to PIL Image
    image_data = base64.b64decode(upscaled_base64)
    upscaled_image = Image.open(io.BytesIO(image_data))

    return upscaled_image


# Usage example:
# upscaled_img = upscale_image("my_image.jpg", "my-project-id", upscale_factor="x4")
# upscaled_img.save("upscaled_image.png")

# --- Run the upscaling process ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upscale an image using Google Vertex AI.")
    parser.add_argument("image_path", type=str, help="Path to the image to upscale.")
    parser.add_argument("--upscale_factor", type=str, default="x4", choices=["x2", "x4"],
                        help="Upscale factor (default: x4).")
    args = parser.parse_args()

    upscaled_img = upscale_image(args.image_path, PROJECT_ID, LOCATION, upscale_factor=args.upscale_factor)
    output_path = f"upscaled_{args.image_path}"
    upscaled_img.save(output_path)
    print(f"Upscaled image saved to {output_path}")