"""
Flask web API server for image generation services.
Provides REST endpoints for uploading images and generating new images with Gemini AI.
Supports multi-image input and custom output directories.
"""

__author__ = "Lene Preuss <lene.preuss@gmail.com>"

import json
from pathlib import Path
from typing import Any, List, Tuple

# Coloredlogs handled in setup_logging() from utils
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
from stable_delusion import builders
from stable_delusion.utils import create_error_response, setup_logging


# Lazy initialization to avoid config loading at import time
app = Flask(__name__)


class _AppState:
    """Application state container to avoid global variables."""

    def __init__(self):
        self.config = None
        self.file_repository = None


_state = _AppState()


def get_config():
    if _state.config is None:
        _state.config = ConfigManager.get_config()
        app.config["UPLOAD_FOLDER"] = _state.config.upload_folder
    return _state.config


def get_file_repository():
    if _state.file_repository is None:
        _state.file_repository = builders.create_file_repository()
    return _state.file_repository


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


def _validate_and_save_uploaded_files() -> List[Path]:
    if "images" not in request.files:
        raise ValueError("Missing 'images' parameter")

    images = request.files.getlist("images")
    return get_file_repository().save_uploaded_files(images, app.config["UPLOAD_FOLDER"])


def _create_request_dto(saved_files: List[Path]) -> GenerateImageRequest:
    config = get_config()
    scale = None
    if request.form.get("scale"):
        try:
            scale = int(request.form["scale"])
        except ValueError as e:
            raise ValidationError(f"Invalid scale parameter: {e}", field="scale") from e

    return GenerateImageRequest(
        prompt=request.form.get("prompt") or DEFAULT_PROMPT,
        images=saved_files,
        project_id=request.form.get("project_id") or config.project_id,
        location=request.form.get("location") or config.location,
        output_dir=Path(request.form.get("output_dir") or config.default_output_dir),
        scale=scale,
        image_size=request.form.get("size"),
        output_filename=request.form.get("output_filename"),
        storage_type=request.form.get("storage_type"),
        model=request.form.get("model"),
    )


def _create_generation_service(request_dto: GenerateImageRequest) -> Any:
    return builders.create_image_generation_service(
        project_id=request_dto.project_id,
        location=request_dto.location,
        output_dir=request_dto.output_dir,
        storage_type=request_dto.storage_type,
        model=request_dto.model,
    )


def _handle_custom_output_filename(response_dto: Any, request_dto: GenerateImageRequest) -> None:
    if response_dto.generated_file and request_dto.output_filename and request_dto.output_dir:
        custom_path = request_dto.output_dir / request_dto.output_filename
        response_dto.generated_file.rename(custom_path)
        response_dto.image_config.generated_file = custom_path


def _handle_generation_errors(error: Exception) -> Tuple[Response, int]:
    if isinstance(error, (ValidationError, ConfigurationError)):
        error_response = ErrorResponse(str(error))
        return jsonify(error_response.to_dict()), 400
    if isinstance(error, (ImageGenerationError, UpscalingError, FileOperationError)):
        error_response = ErrorResponse(f"Image generation failed: {error}")
        return jsonify(error_response.to_dict()), 500
    if isinstance(error, ValueError):
        error_msg = str(error)
        if "Missing 'images' parameter" in error_msg or "Invalid scale parameter" in error_msg:
            error_response = ErrorResponse(error_msg)
            return jsonify(error_response.to_dict()), 400
        error_response = ErrorResponse(f"Unexpected error: {error}")
        return jsonify(error_response.to_dict()), 500
    if isinstance(error, OSError):
        error_response = ErrorResponse(f"Failed to rename output file: {error}")
        return jsonify(error_response.to_dict()), 500
    error_response = ErrorResponse(f"Unexpected error: {error}")
    return jsonify(error_response.to_dict()), 500


@app.route("/generate", methods=["POST"])
def generate() -> Tuple[Response, int]:
    try:
        saved_files = _validate_and_save_uploaded_files()
        request_dto = _create_request_dto(saved_files)
        service = _create_generation_service(request_dto)
        response_dto = service.generate_image(request_dto)
        _handle_custom_output_filename(response_dto, request_dto)
        return jsonify(response_dto.to_dict()), 200
    except Exception as e:  # pylint: disable=broad-exception-caught
        return _handle_generation_errors(e)


@app.route("/metadata/<hash_prefix>", methods=["GET"])
def get_metadata(hash_prefix: str) -> Tuple[Response, int]:
    try:
        metadata_repo = builders.create_metadata_repository()
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
    """Main entry point for the stable-delusion application."""
    import sys

    # Check for quiet and debug flags
    quiet_mode = "-q" in sys.argv or "--quiet" in sys.argv
    debug_mode = "-d" in sys.argv or "--debug" in sys.argv

    # Setup coloredlogs for better console output
    setup_logging(quiet=quiet_mode, debug=debug_mode)

    # Check if --version is requested
    if len(sys.argv) > 1 and sys.argv[1] in ["--version", "-V"]:
        from stable_delusion import __version__

        print(f"stable-delusion {__version__}")
        return

    # Check if --help is requested or if there are CLI arguments for image generation
    cli_flags = {"--help", "-h", "--image", "--prompt", "-q", "--quiet", "-d", "--debug"}
    if len(sys.argv) > 1 and sys.argv[1] in cli_flags:
        # Delegate to the CLI interface in generate.py
        from stable_delusion.generate import main as generate_main

        generate_main()
        return

    # Default behavior: start Flask server
    config = get_config()
    app.run(debug=config.flask_debug)


if __name__ == "__main__":
    main()
