"""
Image generation using Google Gemini 2.5 Flash Image Preview API.
Supports multi-image input, custom prompts, and automatic upscaling integration.
Provides both CLI interface and programmatic API for image generation workflows.
"""

__author__ = "Lene Preuss <lene.preuss@gmail.com>"

import argparse
import json
import logging
import shutil
import sys
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import List, Optional, Any, Tuple, TYPE_CHECKING

# Coloredlogs handled in setup_logging() from utils

from google import genai
from google.cloud import aiplatform
from google.genai.types import GenerateContentResponse
from PIL import Image

from stable_delusion.config import ConfigManager
from stable_delusion.config import DEFAULT_PROJECT_ID, DEFAULT_LOCATION, DEFAULT_GEMINI_MODEL
from stable_delusion.exceptions import ImageGenerationError, FileOperationError
from stable_delusion import builders
from stable_delusion.models.client_config import GeminiClientConfig
from stable_delusion.models.metadata import GenerationMetadata
from stable_delusion.upscale import upscale_image
from stable_delusion.utils import (
    log_upload_info,
    validate_image_file,
    ensure_directory_exists,
    generate_timestamped_filename,
    setup_logging,
)

if TYPE_CHECKING:
    from stable_delusion.models.client_config import AWSConfig
    from stable_delusion.models.requests import GenerateImageRequest
    from stable_delusion.models.responses import GenerateImageResponse

DEFAULT_PROMPT = "A futuristic cityscape with flying cars at sunset"

# Coloredlogs will be configured in main() after parsing arguments


@dataclass
class GenerationConfig:
    """Configuration for image generation parameters."""

    project_id: str = DEFAULT_PROJECT_ID
    location: str = DEFAULT_LOCATION
    output_dir: Path = Path(".")
    storage_type: Optional[str] = None


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


class GeminiClient:
    """Client for generating images using Google Gemini API."""

    def __init__(self, client_config: GeminiClientConfig) -> None:
        config = ConfigManager.get_config()

        self._validate_client_config(client_config)
        self._initialize_client_properties(config, client_config)

        # Apply temporary config overrides and initialize repositories
        original_values = self._store_original_config_values(config)
        self._apply_config_overrides(config, client_config)
        self._initialize_repositories()

        # Get effective storage type and restore original config
        assert client_config.storage is not None  # nosec  # Guaranteed by _validate_client_config
        effective_storage_type = client_config.storage.storage_type or config.storage_type
        self._restore_original_config_values(config, original_values)

        # Setup storage and initialize clients
        self._setup_storage(effective_storage_type)
        self._validate_and_initialize_clients(config)

    def _validate_client_config(self, client_config: GeminiClientConfig) -> None:
        if client_config.gcp is None:
            raise TypeError("GCP config is required but was None")
        if client_config.storage is None:
            raise TypeError("Storage config is required but was None")
        if client_config.aws is None:
            raise TypeError("AWS config is required but was None")
        if client_config.app is None:
            raise TypeError("App config is required but was None")

    def _initialize_client_properties(self, config, client_config: GeminiClientConfig) -> None:
        # Type assertions for MyPy - these are guaranteed by _validate_client_config
        assert client_config.gcp is not None  # nosec
        assert client_config.storage is not None  # nosec

        self.project_id = client_config.gcp.project_id or config.project_id
        self.location = client_config.gcp.location or config.location
        output_options = [
            client_config.storage.output_dir,
            client_config.storage.default_output_dir,
            config.default_output_dir,
        ]
        self.output_dir = next(option for option in output_options if option)

    def _store_original_config_values(self, config) -> dict:
        return {
            "storage_type": config.storage_type,
            "gemini_api_key": config.gemini_api_key,
            "s3_bucket": config.s3_bucket,
            "s3_region": config.s3_region,
            "aws_access_key_id": config.aws_access_key_id,
            "aws_secret_access_key": config.aws_secret_access_key,
            "upload_folder": config.upload_folder,
            "flask_debug": config.flask_debug,
        }

    def _apply_config_overrides(self, config, client_config: GeminiClientConfig) -> None:
        # Type assertions for MyPy - these are guaranteed by _validate_client_config
        assert client_config.storage is not None  # nosec
        assert client_config.gcp is not None  # nosec
        assert client_config.aws is not None  # nosec
        assert client_config.app is not None  # nosec

        overrides = [
            (client_config.storage.storage_type, "storage_type"),
            (client_config.gcp.gemini_api_key, "gemini_api_key"),
            (client_config.aws.s3_bucket, "s3_bucket"),
            (client_config.aws.s3_region, "s3_region"),
            (client_config.aws.aws_access_key_id, "aws_access_key_id"),
            (client_config.aws.aws_secret_access_key, "aws_secret_access_key"),
            (client_config.storage.upload_folder, "upload_folder"),
            (client_config.app.flask_debug, "flask_debug"),
        ]

        for override_value, config_attr in overrides:
            if override_value is not None:
                setattr(config, config_attr, override_value)

    def _initialize_repositories(self) -> None:
        self.image_repository = builders.create_image_repository()
        self.file_repository = builders.create_file_repository()
        self.metadata_repository = builders.create_metadata_repository()

    def _restore_original_config_values(self, config, original_values: dict) -> None:
        for key, value in original_values.items():
            setattr(config, key, value)

    def _setup_storage(self, effective_storage_type: str) -> None:
        if effective_storage_type == "local":
            ensure_directory_exists(self.output_dir)
        else:
            # For S3, create directory marker
            self.file_repository.create_directory(self.output_dir)

    def _validate_and_initialize_clients(self, config) -> None:
        if not config.gemini_api_key:
            from stable_delusion.exceptions import ConfigurationError

            raise ConfigurationError(
                "GEMINI_API_KEY environment variable is required but not set",
                config_key="GEMINI_API_KEY",
            )

        self.client = genai.Client()
        # Initialize the Vertex AI client
        aiplatform.init(project=self.project_id, location=self.location)

    def _create_generation_metadata(
        self, prompt_text: str, image_paths: List[Path], scale: Optional[int]
    ) -> GenerationMetadata:
        image_urls = [str(path) for path in image_paths]  # Convert to strings for hashing
        return GenerationMetadata(
            prompt=prompt_text,
            images=image_urls,
            generated_image="",  # Will be set after generation
            gcp_project_id=self.project_id,
            gcp_location=self.location,
            scale=scale,
            model=DEFAULT_GEMINI_MODEL,
        )

    def _check_existing_generation(self, metadata: GenerationMetadata) -> Optional[Path]:
        existing_metadata_key = self.metadata_repository.metadata_exists(
            metadata.content_hash or ""
        )

        if existing_metadata_key:
            logging.info(
                "Found existing generation with hash %s, reusing result",
                (metadata.content_hash or "unknown")[:8],
            )
            try:
                existing_metadata = self.metadata_repository.load_metadata(existing_metadata_key)

                # Return the existing generated image path
                if existing_metadata.generated_image:
                    # Convert S3 URL back to Path if needed
                    if existing_metadata.generated_image.startswith("s3://"):
                        # For S3 URLs, return the URL as a Path
                        return Path(existing_metadata.generated_image)
                    return Path(existing_metadata.generated_image)
            except (FileOperationError, ValueError, json.JSONDecodeError) as e:
                logging.warning(
                    "Failed to load existing metadata, proceeding with generation: %s", e
                )
        return None

    def _generate_new_image(self, prompt_text: str, image_paths: List[Path]):
        uploaded_files = self.upload_files(image_paths)

        response = self.client.models.generate_content(
            model=DEFAULT_GEMINI_MODEL,
            contents=[prompt_text, *uploaded_files],
        )
        if not response.candidates:
            log_failure_reason(response)
            raise ImageGenerationError(
                "Image generation failed - no candidates returned",
                prompt=prompt_text,
                api_response=str(response),
            )
        logging.info(
            "Generated image with %d candidates, finish_reason: %s, tokens: %d",
            len(response.candidates),
            response.candidates[0].finish_reason,
            response.usage_metadata.total_token_count if response.usage_metadata else 0,
        )
        return response

    def _save_generation_results(
        self, metadata: GenerationMetadata, generated_image_path: Path
    ) -> None:
        metadata.generated_image = str(generated_image_path)
        try:
            metadata_key = self.metadata_repository.save_metadata(metadata)
            logging.info("Saved generation metadata: %s", metadata_key)
        except FileOperationError as e:
            logging.warning("Failed to save metadata: %s", e)

    def generate_from_images(
        self, prompt_text: str, image_paths: List[Path], scale: Optional[int] = None
    ) -> Optional[Path]:
        # Create metadata for deduplication check
        temp_metadata = self._create_generation_metadata(prompt_text, image_paths, scale)

        # Check for existing generation with same inputs
        existing_path = self._check_existing_generation(temp_metadata)
        if existing_path:
            return existing_path

        # Proceed with new generation
        response = self._generate_new_image(prompt_text, image_paths)
        generated_image_path = self.save_response_image(response)

        if generated_image_path:
            self._save_generation_results(temp_metadata, generated_image_path)

        return generated_image_path

    def _log_prompt_feedback(self, response: GenerateContentResponse) -> None:
        if response.prompt_feedback and hasattr(response.prompt_feedback, "block_reason"):
            logging.warning("Reason: %s", response.prompt_feedback.block_reason)
        if response.prompt_feedback and response.prompt_feedback.safety_ratings:
            logging.warning("Safety Ratings for the blocked prompt:")
            for rating in response.prompt_feedback.safety_ratings:
                if rating.category and hasattr(rating.category, "name"):
                    logging.warning("  Category: %s", rating.category.name)
                if rating.probability and hasattr(rating.probability, "name"):
                    logging.warning("  Probability: %s", rating.probability.name)

    def _log_safety_ratings(self, candidate) -> None:
        logging.warning("Safety Ratings for the blocked candidate:")
        for rating in candidate.safety_ratings or []:
            if rating.category and hasattr(rating.category, "name"):
                logging.warning("  Category: %s", rating.category.name)
            if rating.probability and hasattr(rating.probability, "name"):
                logging.warning("  Probability: %s", rating.probability.name)

    def save_response_image(self, response: GenerateContentResponse) -> Optional[Path]:
        self._validate_response_has_candidates(response)
        candidate = self._get_and_validate_candidate(response)
        return self._extract_and_save_image_from_candidate(candidate)

    def _validate_response_has_candidates(self, response: GenerateContentResponse) -> None:
        """Validate response has candidates."""
        if not response.candidates:
            logging.warning("No candidates found in the API response.")
            logging.warning("Prompt was blocked.")
            self._log_prompt_feedback(response)
            raise ImageGenerationError(
                "No candidates returned from the API", api_response=str(response.model_dump_json())
            )

    def _get_and_validate_candidate(self, response: GenerateContentResponse):
        """Get first candidate and validate its content."""
        # _validate_response_has_candidates ensures candidates is not None
        if response.candidates is None:
            raise ImageGenerationError("Response validation failed: candidates is None")
        candidate = response.candidates[0]
        self._check_candidate_finish_reason(candidate)
        self._validate_candidate_content(candidate)
        return candidate

    def _check_candidate_finish_reason(self, candidate) -> None:
        """Check and log candidate finish reason."""
        if candidate.finish_reason and hasattr(candidate.finish_reason, "name"):
            logging.debug("Finish reason: %s", candidate.finish_reason.name)
            if "SAFETY" in candidate.finish_reason.name:
                logging.warning("Candidate was blocked due to safety policies.")
                self._log_safety_ratings(candidate)

    def _validate_candidate_content(self, candidate) -> None:
        """Validate candidate has content parts."""
        if not candidate.content or not candidate.content.parts:
            logging.warning("No content parts found in the API response.")
            raise ImageGenerationError(
                "No content parts in the candidate", api_response=str(candidate.model_dump_json())
            )

    def _extract_and_save_image_from_candidate(self, candidate) -> Optional[Path]:
        """Extract and save image from candidate content parts."""
        for part in candidate.content.parts:
            if part.text is not None:
                logging.debug("Response text: %s", part.text)
            elif part.inline_data is not None and part.inline_data.data is not None:
                return self._save_inline_image_data(part.inline_data.data)

        logging.warning("No image found in the API response.")
        return None

    def _save_inline_image_data(self, image_data: bytes) -> Path:
        """Save inline image data to file."""
        image = Image.open(BytesIO(image_data))
        filename = generate_timestamped_filename("generated")
        filepath = self.output_dir / filename
        return self.image_repository.save_image(image, filepath)

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
        preview_image = self.generate_from_images(prompt_text, image_paths, scale=scale)

        if scale is not None and preview_image:
            upscaled_filename = self.output_dir / f"upscaled_{preview_image.name}"
            upscale_factor = f"x{scale}"
            upscaled_image = upscale_image(
                preview_image, self.project_id, self.location, upscale_factor=upscale_factor
            )
            # Save upscaled image using image repository
            saved_path = self.image_repository.save_image(upscaled_image, upscaled_filename)
            return saved_path

        return preview_image


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
