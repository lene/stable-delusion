"""
SeeEdit Seedream 4.0 client implementation.
Provides image generation and editing capabilities using ByteDance's Seedream 4.0 model.
"""

__author__ = "Lene Preuss <lene.preuss@gmail.com>"

import logging
import os
from pathlib import Path
from typing import List, Optional

from byteplussdkarkruntime import Ark  # type: ignore

from stable_delusion.conf import DEFAULT_SEEDREAM_MODEL
from stable_delusion.exceptions import ImageGenerationError, AuthenticationError
from stable_delusion.utils import generate_timestamped_filename


def _is_valid_url(url_string: str) -> bool:
    # Fix Path normalization
    if url_string.startswith("https:/") and not url_string.startswith("https://"):
        url_string = url_string.replace("https:/", "https://", 1)
    elif url_string.startswith("http:/") and not url_string.startswith("http://"):
        url_string = url_string.replace("http:/", "http://", 1)

    # Basic validation - must start with http/https and have more content
    if not url_string.startswith(("http://", "https://")):
        return False

    # Must have more than just the protocol
    if url_string in ("http://", "https://"):
        return False

    # Must have at least a domain part
    try:
        # Remove protocol and check if there's a valid domain
        without_protocol = url_string.split("://", 1)[1]
        if not without_protocol or without_protocol.startswith("/"):
            return False

        # Basic domain validation - must have at least one dot or be localhost
        domain_part = without_protocol.split("/")[0]
        if not domain_part or ("." not in domain_part and domain_part != "localhost"):
            return False

        return True
    except (IndexError, AttributeError):
        return False


class SeedreamClient:
    """Client for interacting with SeeEdit Seedream 4.0 API via BytePlus SDK."""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.model = DEFAULT_SEEDREAM_MODEL

        # Initialize BytePlus Ark client
        logging.info("🔧 Initializing BytePlus Ark client with API key: %s***", api_key[:8])
        self.client = Ark(api_key=api_key)

    def _process_input_images(self, image_urls: Optional[List[str]]) -> List[str]:
        if not image_urls or not isinstance(image_urls, list):
            return []

        logging.info("   Input image URLs: %d URLs", len(image_urls))
        return self._normalize_image_urls(image_urls)

    def _normalize_image_urls(self, image_urls: List[str]) -> List[str]:
        validated_urls = []
        for url in image_urls[:10]:  # API supports max 10 images
            if _is_valid_url(url):
                # Fix any URL normalization issues (shouldn't be needed for string URLs)
                if url.startswith("https:/") and not url.startswith("https://"):
                    url = url.replace("https:/", "https://", 1)
                elif url.startswith("http:/") and not url.startswith("http://"):
                    url = url.replace("http:/", "http://", 1)
                validated_urls.append(url)
            else:
                logging.warning("⚠️  Skipping invalid URL: %s", url)

        if validated_urls:
            for i, url in enumerate(validated_urls):
                logging.info("     %d. Using URL: %s", i + 1, url)
        else:
            logging.warning("⚠️  No valid URLs found in input")

        return validated_urls

    def _prepare_api_parameters(
        self, prompt: str, input_images: List[str], image_size: str, seed: Optional[int]
    ) -> dict:
        api_params = {
            "model": self.model,
            "prompt": prompt,
            "size": image_size,
            "sequential_image_generation": "disabled",
            "response_format": "url",
            "watermark": True,
        }

        if input_images:
            api_params["image"] = input_images

        if seed is not None:
            api_params["seed"] = seed

        return api_params

    def _parse_api_response(self, response) -> List[str]:
        logging.info("📥 Received response from BytePlus")
        logging.info("   Response type: %s", type(response))
        logging.info("   Response: %s", response)

        # Parse response based on BytePlus Ark SDK structure
        image_urls = []
        if hasattr(response, "data") and response.data:
            for item in response.data:
                if hasattr(item, "url") and item.url:
                    image_urls.append(item.url)
                    logging.info("✅ Found generated image URL: %s", item.url)

        logging.info("📊 Response structure: %s", type(response))
        if hasattr(response, "data"):
            data_count = len(response.data) if response.data else 0
            logging.info("📊 Response data: %d items", data_count)

        if not image_urls:
            logging.warning("No image URLs found in response: %s", response)
            logging.warning("Response attributes: %s", dir(response))
            raise ImageGenerationError(
                "No images were generated by Seedream API", api_response=str(response)
            )

        logging.info("✅ Generated %d images using Seedream 4.0", len(image_urls))
        for i, url in enumerate(image_urls):
            logging.info("   %d. %s", i + 1, url)

        return image_urls

    def generate_image(
        self,
        prompt: str,
        image_urls: Optional[List[str]] = None,
        image_size: str = "2K",
        seed: Optional[int] = None,
    ) -> List[str]:
        logging.info("🌱 Starting Seedream image generation via BytePlus SDK")
        logging.info("   Prompt: %s", prompt)
        logging.info("   Model: %s", self.model)
        api_key_display = (
            f"{self.api_key[:8]}***{self.api_key[-4:]}"
            if len(self.api_key) > 12
            else f"{self.api_key[:8]}***"
        )
        logging.info("   API Key: %s", api_key_display)

        if seed is not None:
            logging.info("   Seed: %s", seed)

        try:
            # Process input images
            input_images = self._process_input_images(image_urls)

            # Prepare API parameters
            api_params = self._prepare_api_parameters(prompt, input_images, image_size, seed)

            logging.info("📡 Making API request via BytePlus Ark SDK")
            logging.info("   Final API params: %s", list(api_params.keys()))

            # Make API call
            response = self.client.images.generate(**api_params)

            # Parse and return results
            return self._parse_api_response(response)

        except ImageGenerationError:
            # Re-raise ImageGenerationError as-is
            raise
        except Exception as e:
            logging.error("❌ Seedream generation failed: %s", str(e))
            if "401" in str(e) or "Unauthorized" in str(e):
                raise AuthenticationError("Invalid API key for BytePlus Seedream API") from e
            raise ImageGenerationError(f"Seedream image generation failed: {str(e)}") from e

    def download_image(self, image_url: str, output_path: Path) -> Path:
        try:
            import requests

            logging.info("⬇️  Downloading image from: %s", image_url)
            response = requests.get(image_url, timeout=60)
            response.raise_for_status()

            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Save image
            with open(output_path, "wb") as f:
                f.write(response.content)

            logging.info("💾 Downloaded Seedream image to: %s", output_path)
            return output_path

        except Exception as e:
            raise ImageGenerationError(
                f"Failed to download image from {image_url} to {output_path}: {str(e)}"
            ) from e

    def generate_and_save(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        prompt: str,
        output_dir: Path,
        image_urls: Optional[List[str]] = None,
        image_size: str = "2K",
        base_name: str = "seedream_generated",
    ) -> Path:
        logging.info("💾 SeedreamClient.generate_and_save() called")
        logging.info("   Output dir: %s", output_dir)
        logging.info("   Base name: %s", base_name)

        # Generate image using URLs
        logging.info("🎯 Calling generate_image()")
        generated_image_urls = self.generate_image(prompt, image_urls, image_size)

        if not generated_image_urls:
            logging.error("❌ No images generated by Seedream API")
            raise ImageGenerationError("No images generated by Seedream API")

        logging.info("✅ Generated %d image(s)", len(generated_image_urls))
        for i, url in enumerate(generated_image_urls):
            logging.info("   %d. %s", i + 1, url)

        # Use first generated image
        image_url = generated_image_urls[0]
        logging.info("📥 Using first image: %s", image_url)

        # Generate output filename
        output_filename = generate_timestamped_filename(base_name, "png")
        output_path = output_dir / output_filename
        logging.info("💽 Saving to: %s", output_path)

        # Download and save
        logging.info("⬇️  Starting download...")
        result_path = self.download_image(image_url, output_path)

        # Verify the file was actually created and has content
        if result_path.exists() and result_path.stat().st_size > 0:
            logging.info(
                "✅ Download complete: %s (size: %d bytes)", result_path, result_path.stat().st_size
            )
            return result_path
        raise ImageGenerationError(f"Generated image file is missing or empty: {result_path}")

    @classmethod
    def create_with_env_key(cls, api_key_env: str = "ARK_API_KEY") -> "SeedreamClient":
        api_key = os.getenv(api_key_env)
        if not api_key:
            raise AuthenticationError(
                f"BytePlus ARK API key not found in environment variable: {api_key_env}"
            )

        return cls(api_key)
