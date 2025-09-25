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

from nano_api.config import ConfigManager
from nano_api.exceptions import (ValidationError, ImageGenerationError,
                                 UpscalingError, FileOperationError,
                                 ConfigurationError)
from nano_api.generate import DEFAULT_PROMPT
from nano_api.models.requests import GenerateImageRequest
from nano_api.models.responses import (ErrorResponse, HealthResponse,
                                       APIInfoResponse)
from nano_api.factories import ServiceFactory, RepositoryFactory
from nano_api.utils import create_error_response


config = ConfigManager.get_config()
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = config.upload_folder

# Initialize upload repository
upload_repository = RepositoryFactory.create_upload_repository()


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
        saved_files = upload_repository.save_uploaded_files(
            images, app.config["UPLOAD_FOLDER"]
        )

        # Create request DTO with validation
        request_dto = GenerateImageRequest(
            prompt=request.form.get("prompt") or DEFAULT_PROMPT,
            images=saved_files,
            project_id=request.form.get("project_id") or config.project_id,
            location=request.form.get("location") or config.location,
            output_dir=Path(
                request.form.get("output_dir") or config.default_output_dir
            ),
            scale=int(request.form["scale"]) if request.form.get("scale") else None,
            custom_output=request.form.get("output")
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
            output_dir=request_dto.output_dir
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
    if (response_dto.generated_file and request_dto.custom_output
            and request_dto.output_dir):
        try:
            custom_path = request_dto.output_dir / request_dto.custom_output
            response_dto.generated_file.rename(custom_path)
            response_dto.generated_file = custom_path
        except OSError as e:
            error_response = ErrorResponse(f"Failed to rename output file: {e}")
            return jsonify(error_response.to_dict()), 500

    return jsonify(response_dto.to_dict()), 200


if __name__ == "__main__":
    # Use configuration for debug mode
    app.run(debug=config.flask_debug)
