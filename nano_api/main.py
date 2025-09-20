__author__ = 'Lene Preuss <lene.preuss@gmail.com>'
import argparse
import base64
import os
from fileinput import filename
from io import BytesIO

from google import genai
from google.genai import types
from PIL import Image


def parse_command_line():
    parser = argparse.ArgumentParser(description="Generate an image using the Gemini API.")
    parser.add_argument("--prompt", type=str, help="The prompt text for image generation.")
    parser.add_argument("--output", type=str, default="generated_gemini_image.png", help="The output filename for the generated image.")
    parser.add_argument("--reference-image", type=str, action="append", help="Path to a reference image. Can be repeated.")
    return parser.parse_args()


def multi_image_example(prompt_text):
    client = genai.Client()

    # Upload the first image
    image1_path = "path/to/image1.jpg"
    uploaded_file = client.files.upload(file=image1_path)

    # Prepare the second image as inline data
    image2_path = "path/to/image2.png"
    with open(image2_path, 'rb') as f:
        img2_bytes = f.read()

    # Create the prompt with text and multiple images
    response = client.models.generate_content(

        model="gemini-2.5-flash",
        contents=[
            "What is different between these two images?",
            uploaded_file,  # Use the uploaded file reference
            types.Part.from_bytes(
                data=img2_bytes,
                mime_type='image/png'
            )
        ]
    )

    print(response.text)

def generate_gemini_image(prompt_text):
    # Configure the client with the API key
    client = genai.Client()
    # Call the Gemini 2.5 Flash Image model to generate content
    response = client.models.generate_content(
        model="gemini-2.5-flash-image-preview",
        contents=[prompt_text]
    )

    for part in response.candidates[0].content.parts:
        if part.text is not None:
            print(part.text)
        elif part.inline_data is not None:
            image = Image.open(BytesIO(part.inline_data.data))
            filename = "generated_image.png"
            image.save(filename)
            return filename

    print("No image found in the API response.")
    return None


if __name__ == "__main__":
    api_key = os.getenv("GEMINI_API_KEY")  # no longer needed, genai reads it automatically
    args = parse_command_line()
    prompt = args.prompt
    if not prompt:
        prompt = "A futuristic cityscape with flying cars at sunset"
    generate_gemini_image(prompt)
