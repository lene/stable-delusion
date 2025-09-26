"""
Image generation using Google Gemini 2.5 Flash Image Preview API.
Supports multi-image input, custom prompts, and automatic upscaling integration.
Provides both CLI interface and programmatic API for image generation workflows.
"""

__author__ = "Lene Preuss <lene.preuss@gmail.com>"

import argparse
import json
import logging
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import List, Optional, Any

from google import genai
from google.cloud import aiplatform
from google.genai.types import GenerateContentResponse
from PIL import Image

from nano_api.config import ConfigManager
from nano_api.conf import DEFAULT_PROJECT_ID, DEFAULT_LOCATION, DEFAULT_GEMINI_MODEL
from nano_api.exceptions import ImageGenerationError, FileOperationError
from nano_api.factories import RepositoryFactory
from nano_api.models.metadata import GenerationMetadata
from nano_api.models.client_config import GeminiClientConfig
from nano_api.upscale import upscale_image
from nano_api.utils import (log_upload_info, validate_image_file,
                            ensure_directory_exists, generate_timestamped_filename)

DEFAULT_PROMPT = "A futuristic cityscape with flying cars at sunset"

logging.basicConfig(level=logging.INFO)


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
        help="Upscale factor: 2 or 4 (optional).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("."),
        help="Directory where generated files will be saved "
        "(default: current directory).",
    )
    parser.add_argument(
        "--storage-type",
        type=str,
        choices=["local", "s3"],
        help="Storage backend: 'local' for local filesystem or 's3' for AWS S3 "
        "(overrides configuration file setting).",
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

    args = parser.parse_args()

    # Validation: if S3 storage is selected, require S3 credentials
    if args.storage_type == "s3":
        s3_params = [args.aws_s3_bucket, args.aws_s3_region,
                     args.aws_access_key_id, args.aws_secret_access_key]
        if not all(param is not None for param in s3_params):
            # Check if they're available in environment variables
            import os
            env_s3_params = [
                os.getenv("AWS_S3_BUCKET"),
                os.getenv("AWS_S3_REGION"),
                os.getenv("AWS_ACCESS_KEY_ID"),
                os.getenv("AWS_SECRET_ACCESS_KEY")
            ]
            if not all(param for param in env_s3_params):
                parser.error(
                    "When using --storage-type s3, all S3 parameters are required: "
                    "--aws-s3-bucket, --aws-s3-region, --aws-access-key-id, "
                    "--aws-secret-access-key or set corresponding environment variables."
                )

    # Security warnings for sensitive parameters
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

    return args


class GeminiClient:
    """Client for generating images using Google Gemini API."""

    def __init__(self, client_config: Optional[GeminiClientConfig] = None):
        config = ConfigManager.get_config()

        # Initialize client config if not provided
        if client_config is None:
            client_config = GeminiClientConfig()

        # Extract values with precedence: client_config > environment config
        # Note: client_config.gcp is guaranteed to be non-None by __post_init__
        assert client_config.gcp is not None  # nosec
        assert client_config.storage is not None  # nosec
        assert client_config.aws is not None  # nosec
        assert client_config.app is not None  # nosec

        self.project_id = client_config.gcp.project_id or config.project_id
        self.location = client_config.gcp.location or config.location
        output_options = [
            client_config.storage.output_dir,
            client_config.storage.default_output_dir,
            config.default_output_dir
        ]
        self.output_dir = next(option for option in output_options if option)

        # Store original config values before applying client overrides
        original_values = {
            'storage_type': config.storage_type,
            'gemini_api_key': config.gemini_api_key,
            's3_bucket': config.s3_bucket,
            's3_region': config.s3_region,
            'aws_access_key_id': config.aws_access_key_id,
            'aws_secret_access_key': config.aws_secret_access_key,
            'upload_folder': config.upload_folder,
            'flask_debug': config.flask_debug
        }

        # Apply client config overrides to global config (temporary for repository creation)
        if client_config.storage.storage_type is not None:
            config.storage_type = client_config.storage.storage_type
        if client_config.gcp.gemini_api_key is not None:
            config.gemini_api_key = client_config.gcp.gemini_api_key
        if client_config.aws.s3_bucket is not None:
            config.s3_bucket = client_config.aws.s3_bucket
        if client_config.aws.s3_region is not None:
            config.s3_region = client_config.aws.s3_region
        if client_config.aws.aws_access_key_id is not None:
            config.aws_access_key_id = client_config.aws.aws_access_key_id
        if client_config.aws.aws_secret_access_key is not None:
            config.aws_secret_access_key = client_config.aws.aws_secret_access_key
        if client_config.storage.upload_folder is not None:
            config.upload_folder = client_config.storage.upload_folder
        if client_config.app.flask_debug is not None:
            config.flask_debug = client_config.app.flask_debug
        # Initialize repositories with potentially overridden config
        self.image_repository = RepositoryFactory.create_image_repository()
        self.file_repository = RepositoryFactory.create_file_repository()
        self.metadata_repository = RepositoryFactory.create_metadata_repository()

        # Get the effective storage type
        effective_storage_type = (
            client_config.storage.storage_type or config.storage_type
        )

        # Restore original config values
        for key, value in original_values.items():
            setattr(config, key, value)

        # For local storage, create output directory if it doesn't exist
        if effective_storage_type == "local":
            ensure_directory_exists(self.output_dir)
        else:
            # For S3, create directory marker
            self.file_repository.create_directory(self.output_dir)

        self.client = genai.Client()
        # Initialize the Vertex AI client
        aiplatform.init(project=self.project_id, location=self.location)

    @classmethod
    def create_with_gcp(cls, project_id: Optional[str] = None,
                        location: Optional[str] = None,
                        output_dir: Optional[Path] = None) -> 'GeminiClient':
        """
        Convenience method to create GeminiClient with GCP configuration.

        Args:
            project_id: Google Cloud project ID
            location: Google Cloud region
            output_dir: Output directory for generated images

        Returns:
            Configured GeminiClient instance
        """
        from nano_api.models.client_config import GCPConfig, StorageConfig
        config = GeminiClientConfig(
            gcp=GCPConfig(project_id=project_id, location=location),
            storage=StorageConfig(output_dir=output_dir)
        )
        return cls(config)

    @classmethod
    def create_with_s3(cls, *,
                       project_id: Optional[str] = None,
                       location: Optional[str] = None,
                       aws_config: Optional['AWSConfig'] = None) -> 'GeminiClient':
        """
        Convenience method to create GeminiClient with S3 configuration.

        Args:
            project_id: Google Cloud project ID
            location: Google Cloud region
            aws_config: AWS configuration object with S3 settings

        Returns:
            Configured GeminiClient instance
        """
        from nano_api.models.client_config import GCPConfig, AWSConfig, StorageConfig

        if aws_config is None:
            aws_config = AWSConfig()

        config = GeminiClientConfig(
            gcp=GCPConfig(project_id=project_id, location=location),
            aws=aws_config,
            storage=StorageConfig(storage_type="s3")
        )
        return cls(config)

    def generate_from_images(
        self, prompt_text: str, image_paths: List[Path], scale: Optional[int] = None
    ) -> Optional[Path]:
        # Create metadata for deduplication check
        image_urls = [str(path) for path in image_paths]  # Convert to strings for hashing
        temp_metadata = GenerationMetadata(
            prompt=prompt_text,
            images=image_urls,
            generated_image="",  # Will be set after generation
            gcp_project_id=self.project_id,
            gcp_location=self.location,
            scale=scale,
            model=DEFAULT_GEMINI_MODEL
        )

        # Check for existing generation with same inputs
        existing_metadata_key = self.metadata_repository.metadata_exists(
            temp_metadata.content_hash or ""
        )

        if existing_metadata_key:
            logging.info(
                "Found existing generation with hash %s, reusing result",
                (temp_metadata.content_hash or "unknown")[:8]
            )
            try:
                existing_metadata = self.metadata_repository.load_metadata(existing_metadata_key)

                # Return the existing generated image path
                if existing_metadata.generated_image:
                    # Convert S3 URL back to Path if needed
                    if existing_metadata.generated_image.startswith('s3://'):
                        # For S3 URLs, return the URL as a Path
                        return Path(existing_metadata.generated_image)
                    return Path(existing_metadata.generated_image)
            except (FileOperationError, ValueError, json.JSONDecodeError) as e:
                logging.warning(
                    "Failed to load existing metadata, proceeding with generation: %s", e
                )

        # Proceed with new generation
        uploaded_files = self.upload_files(image_paths)

        response = self.client.models.generate_content(
            model=DEFAULT_GEMINI_MODEL,
            contents=[
                prompt_text,
                *uploaded_files
            ]
        )
        if not response.candidates:
            log_failure_reason(response)
            raise ImageGenerationError(
                "Image generation failed - no candidates returned",
                prompt=prompt_text,
                api_response=str(response)
            )
        logging.info(
            "Generated image with %d candidates, finish_reason: %s, tokens: %d",
            len(response.candidates),
            response.candidates[0].finish_reason,
            response.usage_metadata.total_token_count if response.usage_metadata else 0
        )

        # Save the new image and metadata
        generated_image_path = self.save_response_image(response)

        if generated_image_path:
            # Update metadata with generated image path and save it
            temp_metadata.generated_image = str(generated_image_path)
            try:
                metadata_key = self.metadata_repository.save_metadata(temp_metadata)
                logging.info("Saved generation metadata: %s", metadata_key)
            except FileOperationError as e:
                logging.warning("Failed to save metadata: %s", e)

        return generated_image_path

    def save_response_image(self, response: GenerateContentResponse) -> Optional[Path]:
        """Save response image using the configured image repository."""
        if not response.candidates:
            logging.warning("No candidates found in the API response.")
            raise ImageGenerationError(
                "No candidates returned from the API",
                api_response=str(response)
            )

        candidate = response.candidates[0]
        if not candidate.content or not candidate.content.parts:
            logging.warning("No content parts found in the API response.")
            raise ImageGenerationError(
                "No content parts in the candidate",
                api_response=str(candidate)
            )

        for part in candidate.content.parts:
            if part.text is not None:
                logging.info(part.text)
            elif part.inline_data is not None and part.inline_data.data is not None:
                image = Image.open(BytesIO(part.inline_data.data))
                filename = generate_timestamped_filename("generated")
                filepath = self.output_dir / filename
                # Use image repository to save the image
                saved_path = self.image_repository.save_image(image, filepath)
                return saved_path
        logging.warning("No image found in the API response.")
        return None

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
                preview_image, self.project_id, self.location,
                upscale_factor=upscale_factor
            )
            # Save upscaled image using image repository
            saved_path = self.image_repository.save_image(upscaled_image, upscaled_filename)
            return saved_path

        return preview_image


def generate_from_images(
    prompt_text: str,
    image_paths: List[Path],
    config: Optional[GenerationConfig] = None
) -> Optional[Path]:
    """Generate images from prompt and reference images with configuration.

    Args:
        prompt_text: Text prompt for image generation
        image_paths: List of reference image paths
        config: Generation configuration (uses defaults if None)

    Returns:
        Path to generated image file
    """
    if config is None:
        config = GenerationConfig()

    from nano_api.models.client_config import GCPConfig, StorageConfig
    client_config = GeminiClientConfig(
        gcp=GCPConfig(
            project_id=config.project_id,
            location=config.location
        ),
        storage=StorageConfig(
            output_dir=config.output_dir,
            storage_type=config.storage_type
        )
    )
    client = GeminiClient(client_config)
    return client.generate_from_images(prompt_text, image_paths)


def save_response_image(
    response: GenerateContentResponse, output_dir: Path = Path(".")
) -> Optional[Path]:
    if not response.candidates:
        logging.warning("No candidates found in the API response.")
        raise ImageGenerationError(
            "No candidates returned from the API",
            api_response=str(response)
        )

    candidate = response.candidates[0]
    if not candidate.content or not candidate.content.parts:
        logging.warning("No content parts found in the API response.")
        raise ImageGenerationError(
            "No content parts in the candidate",
            api_response=str(candidate)
        )

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
    from nano_api.models.client_config import GCPConfig, AWSConfig, StorageConfig, AppConfig
    client_config = GeminiClientConfig(
        gcp=GCPConfig(
            project_id=getattr(args, "gcp_project_id"),
            location=getattr(args, "gcp_location"),
            gemini_api_key=getattr(args, "gemini_api_key")
        ),
        aws=AWSConfig(
            s3_bucket=getattr(args, "aws_s3_bucket"),
            s3_region=getattr(args, "aws_s3_region"),
            aws_access_key_id=getattr(args, "aws_access_key_id"),
            aws_secret_access_key=getattr(args, "aws_secret_access_key")
        ),
        storage=StorageConfig(
            output_dir=getattr(args, "output_dir"),
            storage_type=getattr(args, "storage_type"),
            upload_folder=getattr(args, "upload_folder"),
            default_output_dir=getattr(args, "default_output_dir")
        ),
        app=AppConfig(
            flask_debug=getattr(args, "flask_debug")
        )
    )
    gemini = GeminiClient(client_config)

    hires_file = gemini.generate_hires_image_in_one_shot(prompt, images, scale=args.scale)
    if hires_file:
        if args.scale:
            logging.info("High Res Image saved to %s", hires_file)
        else:
            logging.info("Image saved to %s", hires_file)
    else:
        logging.error("Image generation failed.")
