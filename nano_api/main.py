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
from werkzeug.utils import secure_filename

from nano_api.config import ConfigManager
from nano_api.exceptions import (ValidationError, ImageGenerationError,
                                 UpscalingError, FileOperationError,
                                 ConfigurationError)
from nano_api.generate import GeminiClient, DEFAULT_PROMPT
from nano_api.models.requests import GenerateImageRequest
from nano_api.models.responses import (GenerateImageResponse, ErrorResponse,
                                       HealthResponse, APIInfoResponse)
from nano_api.utils import create_error_response, get_current_timestamp


config = ConfigManager.get_config()
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = config.upload_folder


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

        # Extract and save uploaded files
        images = request.files.getlist("images")
        saved_files = []

        # Save uploaded files with utility functions
        for image in images:
            timestamp = get_current_timestamp("compact")
            filename = secure_filename(
                image.filename or f"uploaded_image_{timestamp}"
            )
            filepath = app.config["UPLOAD_FOLDER"] / filename
            image.save(str(filepath))
            saved_files.append(filepath)

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

    # Create Gemini client with provided parameters
    try:
        client = GeminiClient(
            project_id=request_dto.project_id,
            location=request_dto.location,
            output_dir=request_dto.output_dir
        )
    except (ConfigurationError, ValidationError) as e:
        error_response = ErrorResponse(str(e))
        return jsonify(error_response.to_dict()), 400

    # Generate image with optional upscaling
    try:
        generated_file = client.generate_hires_image_in_one_shot(
            request_dto.prompt, request_dto.images, scale=request_dto.scale
        )
    except (ImageGenerationError, UpscalingError, FileOperationError) as e:
        error_response = ErrorResponse(f"Image generation failed: {e}")
        return jsonify(error_response.to_dict()), 500
    except Exception as e:  # pylint: disable=broad-exception-caught
        # Catch any other unexpected exceptions for API stability
        error_response = ErrorResponse(f"Unexpected error: {e}")
        return jsonify(error_response.to_dict()), 500

    # Handle custom output filename if provided
    if generated_file and request_dto.custom_output and request_dto.output_dir:
        try:
            custom_path = request_dto.output_dir / request_dto.custom_output
            generated_file.rename(custom_path)
            generated_file = custom_path
        except OSError as e:
            error_response = ErrorResponse(f"Failed to rename output file: {e}")
            return jsonify(error_response.to_dict()), 500

    # Create response DTO
    response_dto = GenerateImageResponse(
        generated_file=generated_file,
        prompt=request_dto.prompt,
        project_id=request_dto.project_id or config.project_id,
        location=request_dto.location or config.location,
        scale=request_dto.scale,
        saved_files=saved_files,
        output_dir=request_dto.output_dir or config.default_output_dir
    )

    return jsonify(response_dto.to_dict()), 200


if __name__ == "__main__":
    # Use configuration for debug mode
    app.run(debug=config.flask_debug)
