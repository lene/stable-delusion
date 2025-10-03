"""
Concrete implementation of image generation service using SeeEdit Seedream 4.0.
Wraps the Seedream client functionality in a service interface.
"""

__author__ = "Lene Preuss <lene.preuss@gmail.com>"

import logging
from pathlib import Path
from typing import List, Optional

from stable_delusion.config import ConfigManager
from stable_delusion.models.requests import GenerateImageRequest
from stable_delusion.models.responses import GenerateImageResponse
from stable_delusion.models.client_config import GCPConfig, ImageGenerationConfig
from stable_delusion.repositories.interfaces import ImageRepository
from stable_delusion.services.interfaces import ImageGenerationService
from stable_delusion.seedream import SeedreamClient
from stable_delusion.exceptions import ConfigurationError


class SeedreamImageGenerationService(ImageGenerationService):
    """Concrete implementation of image generation using Seedream 4.0."""

    def __init__(
        self, seedream_client: SeedreamClient, image_repository: Optional[ImageRepository] = None
    ) -> None:
        self.client = seedream_client
        self.image_repository = image_repository

    @classmethod
    def create(
        cls,
        api_key: Optional[str] = None,
        output_dir: Optional[Path] = None,
        image_repository: Optional[ImageRepository] = None,
    ) -> "SeedreamImageGenerationService":
        logging.debug("Creating SeedreamImageGenerationService with output dir: %s", output_dir)

        try:
            if api_key:
                client = SeedreamClient(api_key)
            else:
                client = SeedreamClient.create_with_env_key()
        except Exception as e:
            logging.error("Failed to create Seedream client: %s", str(e))
            raise ConfigurationError(
                f"Failed to create Seedream client: {str(e)}", config_key="SEEDREAM_API_KEY"
            ) from e
        return cls(client, image_repository)

    def _log_generation_request(
        self, request: GenerateImageRequest, effective_output_dir: Path
    ) -> None:
        logging.info("Generating image with Seedream")
        logging.debug("Prompt: %s", request.prompt)
        logging.debug("Image count: %d, Output dir: %s", len(request.images), effective_output_dir)

    def _create_generation_response(
        self, request: GenerateImageRequest, generated_file: Optional[Path] = None
    ) -> GenerateImageResponse:
        config = ConfigManager.get_config()
        return GenerateImageResponse(
            image_config=ImageGenerationConfig(
                generated_file=generated_file,
                prompt=request.prompt,
                scale=request.scale,
                image_size=request.image_size,
                saved_files=request.images,
                output_dir=request.output_dir or config.default_output_dir,
            ),
            gcp_config=GCPConfig(project_id=request.project_id, location=request.location),
        )

    def generate_image(self, request: GenerateImageRequest) -> GenerateImageResponse:
        config = ConfigManager.get_config()
        effective_output_dir = request.output_dir or config.default_output_dir
        self._log_generation_request(request, effective_output_dir)

        try:
            # Handle image uploads to S3 if images are provided
            image_urls = []
            if request.images:
                logging.info("Uploading %d images to S3 for Seedream", len(request.images))
                image_urls = self.upload_images_to_s3(request.images)

            # Generate the image using Seedream with S3 URLs
            # Use custom output filename if provided, otherwise use default
            if request.output_filename:
                generated_file = self.client.generate_and_save(
                    prompt=request.prompt,
                    output_dir=effective_output_dir,
                    output_filename=str(request.output_filename),
                    image_urls=image_urls,  # Use S3 URLs instead of local paths
                    image_size=request.image_size or "2K",
                )
            else:
                generated_file = self.client.generate_and_save(
                    prompt=request.prompt,
                    output_dir=effective_output_dir,
                    image_urls=image_urls,  # Use S3 URLs instead of local paths
                    image_size=request.image_size or "2K",
                )
            logging.info("Image generation completed: %s", generated_file)

            # Upload generated image to S3 if using S3 storage
            final_path = self._upload_generated_image_to_s3(generated_file, config)

            return self._create_generation_response(request, final_path)

        except ConfigurationError as e:
            logging.error("Configuration error during image generation: %s", e)
            return self._create_generation_response(request)
        except Exception as e:  # pylint: disable=broad-exception-caught
            logging.error("Unexpected error during image generation: %s", e)
            return self._create_generation_response(request)

    def _upload_generated_image_to_s3(
        self, generated_file: Optional[Path], config
    ) -> Optional[Path]:
        """Upload generated image to S3 if storage type is S3."""
        if not generated_file:
            return None

        if config.storage_type == "s3" and self.image_repository:
            from PIL import Image
            from stable_delusion.repositories.s3_image_repository import S3ImageRepository

            # Only upload if we have an S3 repository
            if isinstance(self.image_repository, S3ImageRepository):
                logging.info("Uploading generated image to S3: %s", generated_file)
                try:
                    with Image.open(generated_file) as img:
                        # Upload to S3 and get the S3 path
                        s3_path = self.image_repository.save_image(img, generated_file)
                        logging.info("Generated image uploaded to S3: %s", s3_path)
                        return s3_path
                except (OSError, IOError) as e:
                    logging.error("Failed to upload generated image to S3: %s", e)
                    # Return local path as fallback
                    return generated_file

        # For local storage or if upload fails, return the local path
        return generated_file

    def _validate_s3_repository(self) -> None:
        if not self.image_repository:
            raise ConfigurationError(
                "Image repository not configured for S3 uploads", config_key="image_repository"
            )

        from stable_delusion.repositories.s3_image_repository import S3ImageRepository

        if not isinstance(self.image_repository, S3ImageRepository):
            raise ConfigurationError(
                "S3 storage required for Seedream image uploads. Use --storage-type s3",
                config_key="storage_type",
            )

    def _upload_single_image_to_s3(self, image_path: Path) -> str:
        logging.info("Uploading image to S3: %s", image_path)

        from PIL import Image
        from stable_delusion.utils import generate_timestamped_filename
        from stable_delusion.repositories.s3_file_repository import S3FileRepository

        # Upload input images using S3FileRepository which uses input/ prefix
        config = ConfigManager.get_config()
        file_repo = S3FileRepository(config)

        with Image.open(image_path) as img:
            s3_filename = generate_timestamped_filename(
                image_path.stem, image_path.suffix.lstrip(".")
            )
            s3_path = Path(s3_filename)  # Will be saved to input/ by S3FileRepository

            # Use file repository to save to input/ folder
            # We need to convert Image to bytes and upload directly
            import io

            img_bytes = io.BytesIO()
            img.save(img_bytes, format=img.format or "PNG")
            img_bytes.seek(0)

            # Create S3 key with input/ prefix
            from stable_delusion.repositories.s3_client import generate_s3_key, build_s3_url

            s3_key = generate_s3_key(str(s3_path), file_repo.key_prefix)
            file_repo.s3_client.put_object(
                Bucket=file_repo.bucket_name,
                Key=s3_key,
                Body=img_bytes.getvalue(),
                ContentType=f"image/{img.format or 'png'}".lower(),
            )

            s3_url = build_s3_url(file_repo.bucket_name, s3_key)

            # Fix URL normalization issue with Path objects
            if s3_url.startswith("https:/") and not s3_url.startswith("https://"):
                s3_url = s3_url.replace("https:/", "https://", 1)

            logging.info("Uploaded to S3: %s", s3_url)
            return s3_url

    def upload_images_to_s3(self, image_paths: List[Path]) -> List[str]:
        self._validate_s3_repository()
        uploaded_urls = []

        for image_path in image_paths:
            try:
                s3_url = self._upload_single_image_to_s3(image_path)
                uploaded_urls.append(s3_url)
            except Exception as e:
                logging.error("❌ Failed to upload %s to S3: %s", image_path, str(e))
                raise ConfigurationError(
                    f"Failed to upload image {image_path} to S3: {str(e)}", config_key="s3_upload"
                ) from e

        return uploaded_urls

    def upload_files(self, image_paths: List[Path]) -> List[str]:
        return self.upload_images_to_s3(image_paths)
