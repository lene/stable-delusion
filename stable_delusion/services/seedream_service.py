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
from stable_delusion.models.metadata import GenerationMetadata
from stable_delusion.repositories.interfaces import ImageRepository, MetadataRepository
from stable_delusion.services.interfaces import ImageGenerationService
from stable_delusion.seedream import SeedreamClient
from stable_delusion.exceptions import ConfigurationError, FileOperationError


class SeedreamImageGenerationService(ImageGenerationService):
    """Concrete implementation of image generation using Seedream 4.0."""

    def __init__(
        self,
        seedream_client: SeedreamClient,
        image_repository: Optional[ImageRepository] = None,
        metadata_repository: Optional[MetadataRepository] = None,
    ) -> None:
        self.client = seedream_client
        self.image_repository = image_repository
        self.metadata_repository = metadata_repository
        self._s3_hash_cache: Optional[dict] = None  # Cache for SHA-256 -> S3 key mappings

    @classmethod
    def create(
        cls,
        api_key: Optional[str] = None,
        output_dir: Optional[Path] = None,
        image_repository: Optional[ImageRepository] = None,
        metadata_repository: Optional[MetadataRepository] = None,
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
        return cls(client, image_repository, metadata_repository)

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

    def _build_seedream_api_params(
        self, request: GenerateImageRequest, image_urls: List[str]
    ) -> dict:
        api_params = {
            "model": self.client.model,
            "prompt": request.prompt,
            "size": request.image_size or "2K",
            "sequential_image_generation": "disabled",
            "response_format": "url",
            "watermark": False,
        }
        if request.images:
            api_params["image"] = image_urls
        return api_params

    def _create_generation_metadata(self, request: GenerateImageRequest) -> GenerationMetadata:
        image_urls = [str(path) for path in request.images]
        api_endpoint = "https://ark.cn-beijing.volces.com/api/v3/images/generations"
        api_params = self._build_seedream_api_params(request, image_urls)

        return GenerationMetadata(
            prompt=request.prompt,
            images=image_urls,
            generated_image="",
            gcp_project_id=None,
            gcp_location=None,
            scale=request.scale,
            model=self.client.model,
            api_endpoint=api_endpoint,
            api_model=self.client.model,
            api_params=api_params,
        )

    def _save_generation_metadata(
        self, metadata: GenerationMetadata, generated_image_path: Optional[Path]
    ) -> None:
        if not self.metadata_repository or not generated_image_path:
            return

        metadata.generated_image = str(generated_image_path)
        try:
            metadata_key = self.metadata_repository.save_metadata(metadata)
            logging.info("Saved generation metadata: %s", metadata_key)
        except FileOperationError as e:
            logging.warning("Failed to save metadata: %s", e)

    def _upload_input_images_to_s3(self, request: GenerateImageRequest) -> List[str]:
        if not request.images:
            return []
        logging.info("Uploading %d images to S3 for Seedream", len(request.images))
        return self.upload_images_to_s3(request.images)

    def _generate_with_seedream_client(
        self, request: GenerateImageRequest, effective_output_dir: Path, image_urls: List[str]
    ) -> Optional[Path]:
        if request.output_filename:
            generated_file = self.client.generate_and_save(
                prompt=request.prompt,
                output_dir=effective_output_dir,
                output_filename=str(request.output_filename),
                image_urls=image_urls,
                image_size=request.image_size or "2K",
            )
        else:
            generated_file = self.client.generate_and_save(
                prompt=request.prompt,
                output_dir=effective_output_dir,
                image_urls=image_urls,
                image_size=request.image_size or "2K",
            )
        logging.info("Image generation completed: %s", generated_file)
        return generated_file

    def _handle_generation_error(
        self, error: Exception, request: GenerateImageRequest
    ) -> GenerateImageResponse:
        if isinstance(error, ConfigurationError):
            logging.error("Configuration error during image generation: %s", error)
        else:
            logging.error("Unexpected error during image generation: %s", error)
        return self._create_generation_response(request)

    def generate_image(self, request: GenerateImageRequest) -> GenerateImageResponse:
        config = ConfigManager.get_config()
        effective_output_dir = request.output_dir or config.default_output_dir
        self._log_generation_request(request, effective_output_dir)

        metadata = self._create_generation_metadata(request)

        try:
            image_urls = self._upload_input_images_to_s3(request)
            generated_file = self._generate_with_seedream_client(
                request, effective_output_dir, image_urls
            )
            final_path = self._upload_generated_image_to_s3(generated_file, config)
            self._save_generation_metadata(metadata, final_path)
            return self._create_generation_response(request, final_path)
        except (ConfigurationError, Exception) as e:  # pylint: disable=broad-exception-caught
            return self._handle_generation_error(e, request)

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

    def _find_file_by_hash_in_s3(self, file_repo, file_hash: str) -> Optional[str]:
        """Find S3 file with matching hash using cached hash map."""
        # Build cache if not already built
        if self._s3_hash_cache is None:
            from stable_delusion.repositories.s3_client import build_s3_hash_cache

            self._s3_hash_cache = build_s3_hash_cache(
                file_repo.s3_client, file_repo.bucket_name, file_repo.key_prefix
            )

        # O(1) lookup in cache
        return self._s3_hash_cache.get(file_hash)

    def _convert_image_to_bytes(self, image_path: Path) -> tuple[bytes, str]:
        from PIL import Image
        import io

        with Image.open(image_path) as img:
            img_bytes = io.BytesIO()
            img_format = img.format or "PNG"
            img.save(img_bytes, format=img_format)
            return img_bytes.getvalue(), img_format

    def _check_for_duplicate_in_s3(self, file_repo, file_hash: str, config) -> Optional[str]:
        from stable_delusion.repositories.s3_client import build_https_s3_url

        existing_key = self._find_file_by_hash_in_s3(file_repo, file_hash)
        if existing_key:
            https_url = build_https_s3_url(file_repo.bucket_name, existing_key, config.s3_region)
            logging.info(
                "Skipping upload - file with same content already exists in S3: %s",
                https_url,
            )
            return https_url
        return None

    def _upload_image_bytes_to_s3(
        # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        file_repo,
        image_path: Path,
        img_bytes: bytes,
        file_hash: str,
        img_format: str,
        config,
    ) -> str:
        from stable_delusion.utils import generate_timestamped_filename
        from stable_delusion.repositories.s3_client import generate_s3_key, build_https_s3_url

        s3_filename = generate_timestamped_filename(image_path.stem, image_path.suffix.lstrip("."))
        s3_key = generate_s3_key(str(Path(s3_filename)), file_repo.key_prefix)

        file_repo.s3_client.put_object(
            Bucket=file_repo.bucket_name,
            Key=s3_key,
            Body=img_bytes,
            ContentType=f"image/{img_format.lower()}",
            Metadata={
                "original_filename": image_path.name,
                "uploaded_by": "stable-delusion",
                "sha256": file_hash,
            },
        )

        https_url = build_https_s3_url(file_repo.bucket_name, s3_key, config.s3_region)
        logging.info("Uploaded to S3: %s", https_url)
        return https_url

    def _upload_single_image_to_s3(self, image_path: Path) -> str:
        from stable_delusion.utils import calculate_file_sha256, optimize_image_size
        from stable_delusion.repositories.s3_file_repository import S3FileRepository

        config = ConfigManager.get_config()
        file_repo = S3FileRepository(config)

        optimized_path = optimize_image_size(image_path, max_size_mb=7.0)

        img_bytes, img_format = self._convert_image_to_bytes(optimized_path)
        file_hash = calculate_file_sha256(img_bytes)

        duplicate_url = self._check_for_duplicate_in_s3(file_repo, file_hash, config)
        if duplicate_url:
            if optimized_path != image_path:
                optimized_path.unlink(missing_ok=True)
            return duplicate_url

        result = self._upload_image_bytes_to_s3(
            file_repo, optimized_path, img_bytes, file_hash, img_format, config
        )

        if optimized_path != image_path:
            optimized_path.unlink(missing_ok=True)

        return result

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
