"""
Flask web API server for image generation services.
Provides REST endpoints for uploading images and generating new images with Gemini AI.
Supports multi-image input and custom output directories.
"""

__author__ = "Lene Preuss <lene.preuss@gmail.com>"

import json
from pathlib import Path
from typing import Tuple

from flask import Flask, jsonify, request, Response

from stable_delusion.config import ConfigManager
from stable_delusion.exceptions import (
    ValidationError,
    ImageGenerationError,
    UpscalingError,
    FileOperationError,
    ConfigurationError,
)
from stable_delusion.generate import DEFAULT_PROMPT
from stable_delusion.models.requests import GenerateImageRequest
from stable_delusion.models.responses import ErrorResponse, HealthResponse, APIInfoResponse
from stable_delusion.factories import ServiceFactory, RepositoryFactory
from stable_delusion.utils import create_error_response


# Lazy initialization to avoid config loading at import time
app = Flask(__name__)


class _AppState:
    """Application state container to avoid global variables."""

    def __init__(self):
        self.config = None
        self.upload_repository = None


_state = _AppState()


def get_config():
    if _state.config is None:
        _state.config = ConfigManager.get_config()
        app.config["UPLOAD_FOLDER"] = _state.config.upload_folder
    return _state.config


def get_upload_repository():
    if _state.upload_repository is None:
        _state.upload_repository = RepositoryFactory.create_upload_repository()
    return _state.upload_repository


@app.route("/health", methods=["GET"])
def health() -> Tuple[Response, int]:
    response = HealthResponse()
    return jsonify(response.to_dict()), 200


@app.route("/", methods=["GET"])
def api_info() -> Tuple[Response, int]:
    response = APIInfoResponse()
    return jsonify(response.to_dict()), 200


@app.route("/openapi.json", methods=["GET"])
def openapi_spec() -> Tuple[Response, int]:
    try:
        spec_path = Path(__file__).parent.parent / "openapi.json"
        with open(spec_path, "r", encoding="utf-8") as f:
            spec = json.load(f)
        return jsonify(spec), 200
    except FileNotFoundError:
        return create_error_response("OpenAPI specification not found", 404)


@app.route("/generate", methods=["POST"])
def generate() -> Tuple[Response, int]:  # pylint: disable=too-many-return-statements
    try:
        # Validate that images are provided
        if "images" not in request.files:
            error_response = ErrorResponse("Missing 'images' parameter")
            return jsonify(error_response.to_dict()), 400

        # Extract and save uploaded files using repository
        images = request.files.getlist("images")
        saved_files = get_upload_repository().save_uploaded_files(
            images, app.config["UPLOAD_FOLDER"]
        )

        # Create request DTO with validation
        config = get_config()
        request_dto = GenerateImageRequest(
            prompt=request.form.get("prompt") or DEFAULT_PROMPT,
            images=saved_files,
            project_id=request.form.get("project_id") or config.project_id,
            location=request.form.get("location") or config.location,
            output_dir=Path(request.form.get("output_dir") or config.default_output_dir),
            scale=int(request.form["scale"]) if request.form.get("scale") else None,
            image_size=request.form.get("size"),
            custom_output=request.form.get("output"),
            storage_type=request.form.get("storage_type"),
            model=request.form.get("model"),
        )

    except ValidationError as e:
        error_response = ErrorResponse(str(e))
        return jsonify(error_response.to_dict()), 400
    except ValueError as e:
        error_response = ErrorResponse(f"Invalid scale parameter: {e}")
        return jsonify(error_response.to_dict()), 400

    # Create image generation service
    try:
        service = ServiceFactory.create_image_generation_service(
            project_id=request_dto.project_id,
            location=request_dto.location,
            output_dir=request_dto.output_dir,
            storage_type=request_dto.storage_type,
            model=request_dto.model,
        )
    except (ConfigurationError, ValidationError) as e:
        error_response = ErrorResponse(str(e))
        return jsonify(error_response.to_dict()), 400

    # Generate image using service
    try:
        response_dto = service.generate_image(request_dto)
    except (ImageGenerationError, UpscalingError, FileOperationError) as e:
        error_response = ErrorResponse(f"Image generation failed: {e}")
        return jsonify(error_response.to_dict()), 500
    except Exception as e:  # pylint: disable=broad-exception-caught
        # Catch any other unexpected exceptions for API stability
        error_response = ErrorResponse(f"Unexpected error: {e}")
        return jsonify(error_response.to_dict()), 500

    # Handle custom output filename if provided
    if response_dto.generated_file and request_dto.custom_output and request_dto.output_dir:
        try:
            custom_path = request_dto.output_dir / request_dto.custom_output
            response_dto.generated_file.rename(custom_path)
            response_dto.image_config.generated_file = custom_path
        except OSError as e:
            error_response = ErrorResponse(f"Failed to rename output file: {e}")
            return jsonify(error_response.to_dict()), 500

    return jsonify(response_dto.to_dict()), 200


@app.route("/metadata/<hash_prefix>", methods=["GET"])
def get_metadata(hash_prefix: str) -> Tuple[Response, int]:
    try:
        metadata_repo = RepositoryFactory.create_metadata_repository()
        metadata_keys = metadata_repo.list_metadata_by_hash_prefix(hash_prefix)

        return (
            jsonify(
                {
                    "hash_prefix": hash_prefix,
                    "matching_metadata": len(metadata_keys),
                    "metadata_keys": metadata_keys[:10],  # Limit to first 10 for brevity
                }
            ),
            200,
        )

    except (FileOperationError, ConfigurationError) as e:
        error_response = ErrorResponse(f"Failed to query metadata: {str(e)}")
        return jsonify(error_response.to_dict()), 500


def main():
    """Main entry point for the Flask application."""
    # Use configuration for debug mode
    config = get_config()
    app.run(debug=config.flask_debug)


if __name__ == "__main__":
    main()
