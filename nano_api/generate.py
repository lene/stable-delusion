__author__ = 'Lene Preuss <lene.preuss@gmail.com>'
import argparse
import logging
import os
from datetime import datetime
from io import BytesIO
from typing import List, Optional

from google import genai
from google.genai import types
from google.cloud import aiplatform
from PIL import Image

from conf import DEFAULT_PROJECT_ID, DEFAULT_LOCATION
from upscale import upscale_image

logging.basicConfig(level=logging.INFO)

def parse_command_line() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate an image using the Gemini API.")
    parser.add_argument("--prompt", type=str, help="The prompt text for image generation.")
    parser.add_argument("--output", type=str, default="generated_gemini_image.png",
                        help="The output filename for the generated image.")
    parser.add_argument("--image", type=str, action="append",
                        help="Path to a reference image. Can be repeated.")
    parser.add_argument("--project-id", type=str,
                        help="Google Cloud Project ID (defaults to value in conf.py).")
    parser.add_argument("--location", type=str,
                        help="Google Cloud region (defaults to value in conf.py).")
    parser.add_argument("--scale", type=int, choices=[2, 4],
                        help="Upscale factor: 2 or 4 (optional).")
    return parser.parse_args()

class GeminiClient:
    def __init__(self, project_id: str = DEFAULT_PROJECT_ID,
                 location: str = DEFAULT_LOCATION):
        if not os.getenv("GEMINI_API_KEY"):
            raise ValueError("GEMINI_API_KEY environment variable is required but not set")

        self.project_id = project_id
        self.location = location

        self.client = genai.Client()
        # Initialize the Vertex AI client
        aiplatform.init(project=self.project_id, location=self.location)

    def multi_image_example(self, prompt_text: str, image_paths: List[str]) -> Optional[str]:
        uploaded_files = self.upload_files(image_paths)

        response = self.client.models.generate_content(
            model="gemini-2.5-flash-image-preview",
            contents=[
                prompt_text,
                *uploaded_files
            ]
        )
        logging.info(f"Generated image with {len(response.candidates)} candidates, "
                     f"finish_reason: {response.candidates[0].finish_reason}, "
                     f"tokens: {response.usage_metadata.total_token_count}")
        return save_response_image(response)

    def upload_files(self, image_paths):
        uploaded_files = []
        for image in image_paths:
            if not os.path.isfile(image):
                raise FileNotFoundError(f"Image file not found: {image}")
            uploaded_files.append(self.client.files.upload(file=image))
        return uploaded_files


    def generate_hires_image_in_one_shot(self, prompt_text: str, image_paths: List[str],
                                         scale: Optional[int] = None):
        preview_image = self.multi_image_example(prompt_text, image_paths)

        if scale is not None and preview_image:
            upscaled_filename = f"upscaled_{preview_image}"
            upscale_factor = f'x{scale}'
            upscale_image('preview_image.png', self.project_id, self.location,
                          upscale_factor=upscale_factor).save(upscaled_filename)
            return upscaled_filename

        return preview_image

def multi_image_example(prompt_text: str, image_paths: List[str],
                        project_id: str = DEFAULT_PROJECT_ID,
                        location: str = DEFAULT_LOCATION) -> Optional[str]:
    """Standalone function to generate image from prompt and reference images."""
    client = GeminiClient(project_id=project_id, location=location)
    return client.multi_image_example(prompt_text, image_paths)

def save_response_image(response: types.GenerateContentResponse) -> Optional[str]:
    for part in response.candidates[0].content.parts:
        if part.text is not None:
            logging.info(part.text)
        elif part.inline_data is not None:
            image = Image.open(BytesIO(part.inline_data.data))
            current_time = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
            filename = f"generated_{current_time}.png"
            image.save(filename)
            return filename
    logging.warning("No image found in the API response.")
    return None

if __name__ == "__main__":
    args = parse_command_line()
    prompt = args.prompt
    if not prompt:
        prompt = "A futuristic cityscape with flying cars at sunset"
    images = args.image if args.image else []

    # Pass command line arguments to GeminiClient, falling back to defaults if not provided
    project_id = getattr(args, 'project_id') or DEFAULT_PROJECT_ID
    location = getattr(args, 'location') or DEFAULT_LOCATION
    gemini = GeminiClient(project_id=project_id, location=location)

    hires_file = gemini.generate_hires_image_in_one_shot(prompt, images, scale=args.scale)
    if hires_file:
        if args.scale:
            logging.info(f"High Res Image saved to {hires_file}")
        else:
            logging.info(f"Image saved to {hires_file}")
    else:
        logging.error("Image generation failed.")
