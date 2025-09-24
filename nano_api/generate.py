"""
Image generation using Google Gemini 2.5 Flash Image Preview API.
Supports multi-image input, custom prompts, and automatic upscaling integration.
Provides both CLI interface and programmatic API for image generation workflows.
"""

__author__ = "Lene Preuss <lene.preuss@gmail.com>"

import argparse
import logging
from io import BytesIO
from pathlib import Path
from typing import List, Optional, Any

from google import genai
from google.cloud import aiplatform
from google.genai.types import GenerateContentResponse
from PIL import Image

from nano_api.config import ConfigManager
from nano_api.conf import DEFAULT_PROJECT_ID, DEFAULT_LOCATION
from nano_api.upscale import upscale_image
from nano_api.utils import (log_upload_info, validate_image_file,
                            ensure_directory_exists, generate_timestamped_filename)

DEFAULT_PROMPT = "A futuristic cityscape with flying cars at sunset"

logging.basicConfig(level=logging.INFO)


def log_failure_reason(response: GenerateContentResponse) -> None:
    logging.error("No candidates returned from the API.")
    # Check prompt feedback for safety filtering
    if hasattr(response, "prompt_feedback") and response.prompt_feedback:
        feedback = response.prompt_feedback
        if hasattr(feedback, "block_reason"):
            logging.error("Prompt blocked: %s", feedback.block_reason)
        if hasattr(feedback, "safety_ratings") and feedback.safety_ratings:
            for rating in feedback.safety_ratings:
                logging.error("Safety rating: %s = %s", rating.category, rating.probability)
    # Log usage metadata if available
    if hasattr(response, "usage_metadata") and response.usage_metadata:
        logging.error("Usage metadata: %s", response.usage_metadata)
    # Log any other response properties that might give clues
    logging.error("Response type: %s", type(response))
    logging.error(
        "Response attributes: %s",
        [attr for attr in dir(response) if not attr.startswith("_")]
    )


def parse_command_line() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate an image using the Gemini API."
    )
    parser.add_argument(
        "--prompt", type=str, help="The prompt text for image generation."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("generated_gemini_image.png"),
        help="The output filename for the generated image.",
    )
    parser.add_argument(
        "--image",
        type=Path,
        action="append",
        help="Path to a reference image. Can be repeated.",
    )
    parser.add_argument(
        "--project-id",
        type=str,
        help="Google Cloud Project ID (defaults to value in conf.py).",
    )
    parser.add_argument(
        "--location",
        type=str,
        help="Google Cloud region (defaults to value in conf.py).",
    )
    parser.add_argument(
        "--scale",
        type=int,
        choices=[2, 4],
        help="Upscale factor: 2 or 4 (optional).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("."),
        help="Directory where generated files will be saved "
        "(default: current directory).",
    )
    return parser.parse_args()


class GeminiClient:
    """Client for generating images using Google Gemini API."""
    def __init__(
        self,
        project_id: Optional[str] = None,
        location: Optional[str] = None,
        output_dir: Optional[Path] = None,
    ):
        config = ConfigManager.get_config()

        self.project_id = project_id or config.project_id
        self.location = location or config.location
        self.output_dir = output_dir or config.default_output_dir

        # Create output directory if it doesn"t exist
        ensure_directory_exists(self.output_dir)

        self.client = genai.Client()
        # Initialize the Vertex AI client
        aiplatform.init(project=self.project_id, location=self.location)

    def generate_from_images(
        self, prompt_text: str, image_paths: List[Path]
    ) -> Optional[Path]:
        uploaded_files = self.upload_files(image_paths)

        response = self.client.models.generate_content(
            model="gemini-2.5-flash-image-preview",
            contents=[
                prompt_text,
                *uploaded_files
            ]
        )
        if not response.candidates:
            log_failure_reason(response)
            raise RuntimeError(f"Image generation failed, response: {response}")
        logging.info(
            "Generated image with %d candidates, finish_reason: %s, tokens: %d",
            len(response.candidates),
            response.candidates[0].finish_reason,
            response.usage_metadata.total_token_count if response.usage_metadata else 0
        )
        return save_response_image(response, self.output_dir)

    def upload_files(self, image_paths: List[Path]) -> List[Any]:
        uploaded_files = []
        for image_path in image_paths:
            validate_image_file(image_path)
            uploaded_file = self.client.files.upload(file=str(image_path))
            log_upload_info(image_path, uploaded_file)
            uploaded_files.append(uploaded_file)
        return uploaded_files

    def generate_hires_image_in_one_shot(
        self, prompt_text: str, image_paths: List[Path], scale: Optional[int] = None
    ) -> Optional[Path]:
        preview_image = self.generate_from_images(prompt_text, image_paths)

        if scale is not None and preview_image:
            upscaled_filename = self.output_dir / f"upscaled_{preview_image.name}"
            upscale_factor = f"x{scale}"
            upscale_image(
                preview_image, self.project_id, self.location,
                upscale_factor=upscale_factor
            ).save(str(upscaled_filename))
            return upscaled_filename

        return preview_image


def generate_from_images(
    prompt_text: str,
    image_paths: List[Path],
    project_id: str = DEFAULT_PROJECT_ID,
    location: str = DEFAULT_LOCATION,
    output_dir: Path = Path("."),
) -> Optional[Path]:
    client = GeminiClient(
        project_id=project_id, location=location, output_dir=output_dir
    )
    return client.generate_from_images(prompt_text, image_paths)


def save_response_image(
    response: GenerateContentResponse, output_dir: Path = Path(".")
) -> Optional[Path]:
    if not response.candidates:
        logging.warning("No candidates found in the API response.")
        raise RuntimeError("No candidates returned from the API.")

    candidate = response.candidates[0]
    if not candidate.content or not candidate.content.parts:
        logging.warning("No content parts found in the API response.")
        raise RuntimeError("No content parts in the candidate.")

    for part in candidate.content.parts:
        if part.text is not None:
            logging.info(part.text)
        elif part.inline_data is not None and part.inline_data.data is not None:
            image = Image.open(BytesIO(part.inline_data.data))
            filename = generate_timestamped_filename("generated")
            filepath = output_dir / filename
            image.save(str(filepath))
            return filepath
    logging.warning("No image found in the API response.")
    return None


if __name__ == "__main__":
    args = parse_command_line()
    prompt = args.prompt
    if not prompt:
        prompt = DEFAULT_PROMPT
    images = args.image if args.image else []

    # Pass command line arguments to GeminiClient, config provides defaults
    gemini = GeminiClient(
        project_id=getattr(args, "project_id"),
        location=getattr(args, "location"),
        output_dir=getattr(args, "output_dir")
    )

    hires_file = gemini.generate_hires_image_in_one_shot(prompt, images, scale=args.scale)
    if hires_file:
        if args.scale:
            logging.info("High Res Image saved to %s", hires_file)
        else:
            logging.info("Image saved to %s", hires_file)
    else:
        logging.error("Image generation failed.")
