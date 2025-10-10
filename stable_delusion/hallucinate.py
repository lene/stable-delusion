"""
Image generation using Google Gemini 2.5 Flash Image Preview API.
Supports multi-image input, custom prompts, and automatic upscaling integration.
Provides both CLI interface and programmatic API for image generation workflows.
"""

__author__ = "Lene Preuss <lene.preuss@gmail.com>"

import argparse
import logging
import shutil
import sys
from io import BytesIO
from pathlib import Path
from typing import List, Optional, Tuple, TYPE_CHECKING

# Coloredlogs handled in setup_logging() from utils

from google.genai.types import GenerateContentResponse
from PIL import Image

from stable_delusion.exceptions import ImageGenerationError, FileOperationError
from stable_delusion import builders
from stable_delusion.client.gemini_client import GeminiClient
from stable_delusion.generate.generation_config import GenerationConfig
from stable_delusion.models.client_config import GeminiClientConfig
from stable_delusion.utils import (
    generate_timestamped_filename,
    setup_logging,
)

if TYPE_CHECKING:
    from stable_delusion.models.requests import GenerateImageRequest
    from stable_delusion.models.responses import GenerateImageResponse

DEFAULT_PROMPT = "A futuristic cityscape with flying cars at sunset"

# Coloredlogs will be configured in main() after parsing arguments


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
        "Response attributes: %s", [attr for attr in dir(response) if not attr.startswith("_")]
    )


def _validate_and_normalize_output_filename(filename: str) -> str:
    """
    Validate and normalize the output filename according to PNG requirements.

    Args:
        filename: The filename provided by user

    Returns:
        Normalized filename (basename without .png extension)

    Raises:
        SystemExit: If the file extension is not supported
    """
    if not filename:
        return filename

    # Convert to Path for easier extension handling
    path = Path(filename)

    # Get the extension (lowercase for comparison)
    extension = path.suffix.lower()

    if extension == ".png":
        # Strip .png extension but preserve directory path
        return str(path.with_suffix(""))
    if extension == "":
        # No extension - keep as is
        return filename

    # Any other extension is not supported
    print(
        f"Error: file type not supported for --output-filename: '{extension}'. "
        "Only PNG files are supported."
    )
    sys.exit(1)


def _setup_cli_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate an image using the Gemini API.")
    parser.add_argument(
        "-q", "--quiet", action="store_true", help="Quiet mode - only show warnings and errors."
    )
    parser.add_argument(
        "-d",
        "--debug",
        action="store_true",
        help="Debug mode - show all log messages including debug details.",
    )
    parser.add_argument("--prompt", type=str, help="The prompt text for image generation.")
    parser.add_argument(
        "--output-filename",
        type=Path,
        default=None,
        help="The output filename base (without timestamp/extension). "
        "If not specified, model-specific defaults are used "
        "(gemini: 'generated', seedream: 'seedream_image').",
    )
    parser.add_argument(
        "--image",
        type=Path,
        action="append",
        help="Path to a reference image. Can be repeated.",
    )
    parser.add_argument(
        "--gcp-project-id",
        type=str,
        help="Google Cloud Project ID (defaults to value in conf.py).",
    )
    parser.add_argument(
        "--gcp-location",
        type=str,
        help="Google Cloud region (defaults to value in conf.py).",
    )
    parser.add_argument(
        "--scale",
        type=int,
        choices=[2, 4],
        help="Upscale factor: 2 or 4 (optional, Gemini only).",
    )
    parser.add_argument(
        "--size",
        type=str,
        help="Image size for generation (optional, Seedream only). "
        "Can be '1K', '2K', '4K', or '{width}x{height}' "
        "where width is 1280-4096 and height is 720-4096. Examples: '2K', '1920x1080'.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("."),
        help="Directory where generated files will be saved " "(default: current directory).",
    )
    parser.add_argument(
        "--storage-type",
        type=str,
        choices=["local", "s3"],
        help="Storage backend: 'local' for local filesystem or 's3' for AWS S3 "
        "(overrides configuration file setting).",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["gemini", "seedream"],
        help="AI model to use for image generation: 'gemini' for Gemini 2.5 Flash "
        "or 'seedream' for SeeEdit Seedream 4.0 (defaults to 'gemini').",
    )

    # Authentication parameters
    parser.add_argument(
        "--gemini-api-key",
        type=str,
        help="Gemini API key (WARNING: visible in process list - prefer environment variable).",
    )

    # AWS S3 parameters
    parser.add_argument(
        "--aws-s3-bucket",
        type=str,
        help="AWS S3 bucket name (required when using S3 storage).",
    )
    parser.add_argument(
        "--aws-s3-region",
        type=str,
        help="AWS S3 region (required when using S3 storage).",
    )
    parser.add_argument(
        "--aws-access-key-id",
        type=str,
        help="AWS access key ID (WARNING: visible in process list - prefer environment variable).",
    )
    parser.add_argument(
        "--aws-secret-access-key",
        type=str,
        help="AWS secret access key (WARNING: visible in process list - "
        "prefer environment variable).",
    )

    # Flask/Application parameters
    parser.add_argument(
        "--upload-folder",
        type=Path,
        help="Directory for uploaded files (used by Flask API).",
    )
    parser.add_argument(
        "--default-output-dir",
        type=Path,
        help="Default output directory for generated images.",
    )
    parser.add_argument(
        "--flask-debug",
        action="store_true",
        help="Enable Flask debug mode.",
    )
    return parser


def _validate_s3_storage_arguments(
    args: argparse.Namespace, parser: argparse.ArgumentParser
) -> None:
    if args.storage_type == "s3":
        import os

        bucket = args.aws_s3_bucket or os.getenv("AWS_S3_BUCKET")
        region = args.aws_s3_region or os.getenv("AWS_S3_REGION")

        if not bucket or not region:
            parser.error(
                "When using --storage-type s3, bucket and region are required: "
                "--aws-s3-bucket and --aws-s3-region or set AWS_S3_BUCKET and "
                "AWS_S3_REGION environment variables. "
                "Credentials will be read from AWS credential chain "
                "(~/.aws/credentials, environment variables, IAM roles)."
            )


def _warn_about_sensitive_cli_parameters(args: argparse.Namespace) -> None:
    if args.gemini_api_key:
        logging.warning(
            "API key passed via command line is visible in process list. "
            "Consider using GEMINI_API_KEY environment variable instead."
        )

    if args.aws_access_key_id or args.aws_secret_access_key:
        logging.warning(
            "AWS credentials passed via command line are visible in process list. "
            "Consider using AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY "
            "environment variables instead."
        )


def parse_command_line() -> argparse.Namespace:
    parser = _setup_cli_argument_parser()
    args = parser.parse_args()

    # Load .env file before validation to ensure environment variables are available
    from dotenv import load_dotenv

    load_dotenv(override=False)

    _validate_s3_storage_arguments(args, parser)
    _warn_about_sensitive_cli_parameters(args)

    return args


def generate_from_images(
    prompt_text: str, image_paths: List[Path], config: Optional[GenerationConfig] = None
) -> Optional[Path]:
    if config is None:
        config = GenerationConfig()

    from stable_delusion.models.client_config import GCPConfig, StorageConfig

    client_config = GeminiClientConfig(
        gcp=GCPConfig(project_id=config.project_id, location=config.location),
        storage=StorageConfig(output_dir=config.output_dir, storage_type=config.storage_type),
    )
    client = GeminiClient(client_config)
    return client.generate_from_images(prompt_text, image_paths)


def save_response_image(
    response: GenerateContentResponse, output_dir: Path = Path(".")
) -> Optional[Path]:
    if not response.candidates:
        logging.warning("No candidates found in the API response.")
        raise ImageGenerationError(
            "No candidates returned from the API", api_response=str(response)
        )

    candidate = response.candidates[0]
    if not candidate.content or not candidate.content.parts:
        logging.warning("No content parts found in the API response.")
        raise ImageGenerationError("No content parts in the candidate", api_response=str(candidate))

    for part in candidate.content.parts:
        if part.text is not None:
            logging.debug("Response text: %s", part.text)
        elif part.inline_data is not None and part.inline_data.data is not None:
            image = Image.open(BytesIO(part.inline_data.data))
            filename = generate_timestamped_filename("generated")
            filepath = output_dir / filename
            image.save(str(filepath))
            return filepath
    logging.warning("No image found in the API response.")
    return None


def _process_cli_arguments() -> Tuple[str, List[Path], argparse.Namespace]:
    args = parse_command_line()
    prompt = args.prompt if args.prompt else DEFAULT_PROMPT
    images = args.image if args.image else []
    return prompt, images, args


def _create_cli_request_dto(
    prompt: str, images: List[Path], args: argparse.Namespace
) -> "GenerateImageRequest":
    from stable_delusion.models.requests import GenerateImageRequest

    # Validate and normalize output filename if provided (None means use model defaults)
    output_filename = getattr(args, "output_filename")
    if output_filename is not None:
        output_filename = Path(_validate_and_normalize_output_filename(str(output_filename)))

    return GenerateImageRequest(
        prompt=prompt,
        images=images,
        project_id=getattr(args, "gcp_project_id"),
        location=getattr(args, "gcp_location"),
        output_dir=getattr(args, "output_dir"),
        output_filename=output_filename,
        scale=getattr(args, "scale"),
        image_size=getattr(args, "size"),
        storage_type=getattr(args, "storage_type"),
        model=getattr(args, "model"),
    )


def _execute_image_generation(
    request_dto: "GenerateImageRequest",
) -> "GenerateImageResponse":
    service = builders.create_image_generation_service(
        project_id=request_dto.project_id,
        location=request_dto.location,
        output_dir=request_dto.output_dir,
        storage_type=request_dto.storage_type,
        model=request_dto.model,
    )
    return service.generate_image(request_dto)


def _handle_cli_custom_output(
    response: "GenerateImageResponse", request_dto: "GenerateImageRequest"
) -> None:
    if response.generated_file and request_dto.output_filename:
        # Generate timestamped filename with .png extension
        custom_filename = generate_timestamped_filename(
            str(request_dto.output_filename), extension="png"
        )

        logging.debug(
            "Custom output: attempting to rename %s to %s",
            response.generated_file,
            custom_filename,
        )
        logging.debug("Source file exists: %s", response.generated_file.exists())

        # If output_dir is specified, use it; otherwise use the same directory as the source file
        if request_dto.output_dir:
            custom_path = request_dto.output_dir / custom_filename
        else:
            custom_path = response.generated_file.parent / custom_filename

        logging.debug("Target path: %s", custom_path)

        try:
            # Ensure target directory exists
            custom_path.parent.mkdir(parents=True, exist_ok=True)

            # Move the file (handles cross-device links)
            shutil.move(str(response.generated_file), str(custom_path))
            response.image_config.generated_file = custom_path
            logging.debug("Successfully renamed to: %s", custom_path)
        except (OSError, FileNotFoundError, PermissionError, shutil.Error) as e:
            logging.error("Failed to rename file: %s", e)
            logging.debug(
                "Source path: %s (exists: %s)",
                response.generated_file,
                response.generated_file.exists(),
            )
            logging.debug("Target path: %s", custom_path)


def _log_generation_result(response: "GenerateImageResponse", args: argparse.Namespace) -> None:
    if response.generated_file:
        if args.scale:
            logging.info("High Res Image saved to %s", response.generated_file)
        else:
            logging.debug("Image saved to %s", response.generated_file)
    else:
        logging.error("Image generation failed.")


def main():
    try:
        prompt, images, args = _process_cli_arguments()

        # Configure coloredlogs based on quiet/debug flags
        setup_logging(quiet=args.quiet, debug=args.debug)
        request_dto = _create_cli_request_dto(prompt, images, args)
        response = _execute_image_generation(request_dto)
        _handle_cli_custom_output(response, request_dto)
        _log_generation_result(response, args)

    except (ImageGenerationError, FileOperationError) as e:
        logging.error("Image generation failed: %s", e)
    except Exception as e:  # pylint: disable=broad-exception-caught
        logging.error("Unexpected error during image generation: %s", e)


if __name__ == "__main__":
    main()
