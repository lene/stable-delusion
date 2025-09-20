__author__ = 'Lene Preuss <lene.preuss@gmail.com>'
import argparse
import logging
import os
from datetime import datetime
from io import BytesIO
from typing import List, Optional

from google import genai
from google.genai import types
from PIL import Image


logging.basicConfig(level=logging.INFO)

def parse_command_line() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate an image using the Gemini API.")
    parser.add_argument("--prompt", type=str, help="The prompt text for image generation.")
    parser.add_argument("--output", type=str, default="generated_gemini_image.png", help="The output filename for the generated image.")
    parser.add_argument("--image", type=str, action="append", help="Path to a reference image. Can be repeated.")
    return parser.parse_args()


def multi_image_example(prompt_text: str, image_paths: List[str]) -> Optional[str]:
    client = genai.Client()

    uploaded_files = []
    for image in image_paths:
        if not os.path.isfile(image):
            raise FileNotFoundError(f"Image file not found: {image}")
        uploaded_files.append(client.files.upload(file=image))

    # Create the prompt with text and multiple images
    response = client.models.generate_content(
        model="gemini-2.5-flash-image-preview",
        contents=[
            prompt_text,
            *uploaded_files
        ]
    )
    logging.info(response)
    return save_response_image(response)


def generate_gemini_image(prompt_text: str) -> Optional[str]:
    # Configure the client with the API key
    client = genai.Client()
    # Call the Gemini 2.5 Flash Image model to generate content
    response = client.models.generate_content(
        model="gemini-2.5-flash-image-preview",
        contents=[prompt_text]
    )
    return save_response_image(response)


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
    logging.warn("No image found in the API response.")
    return None

if __name__ == "__main__":
    api_key = os.getenv("GEMINI_API_KEY")  # no longer needed, genai reads it automatically
    args = parse_command_line()
    prompt = args.prompt
    if not prompt:
        prompt = "A futuristic cityscape with flying cars at sunset"
    images = args.image if args.image else []
    file = multi_image_example(prompt, images)
    if file:
        logging.info(f"Image saved to {file}")
    else:
        logging.error("Image generation failed.")
