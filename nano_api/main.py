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

from nano_api.generate import GeminiClient, DEFAULT_PROMPT
from nano_api.utils import (create_error_response, get_current_timestamp,
                            validate_scale_parameter, get_project_config)


app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = Path("uploads")
app.config["UPLOAD_FOLDER"].mkdir(exist_ok=True)


@app.route("/health", methods=["GET"])
def health() -> Tuple[Response, int]:
    return jsonify({
        "status": "healthy",
        "service": "NanoAPIClient",
        "version": "1.0.0"
    }), 200


@app.route("/", methods=["GET"])
def api_info() -> Tuple[Response, int]:
    return jsonify({
        "name": "NanoAPIClient API",
        "description": "Flask web API for image generation using Google Gemini AI",
        "version": "1.0.0",
        "endpoints": {
            "/": "API information",
            "/health": "Health check",
            "/generate": "Generate images from prompt and reference images",
            "/openapi.json": "OpenAPI specification"
        }
    }), 200


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
    # Get prompt parameter (use default if not provided)
    prompt = request.form.get("prompt") or DEFAULT_PROMPT

    # Validate that images are provided
    if "images" not in request.files:
        return create_error_response("Missing 'images' parameter")

    # Get optional parameters with defaults using utility function
    project_id, location = get_project_config(request.form)
    output_dir = Path(request.form.get("output_dir", "."))

    # Parse scale parameter using utility function
    try:
        scale = validate_scale_parameter(request.form.get("scale"))
    except ValueError as e:
        return create_error_response(str(e))

    # Parse custom output filename
    custom_output = request.form.get("output")

    images = request.files.getlist("images")
    saved_files = []

    # Save uploaded files with utility functions
    for image in images:
        timestamp = get_current_timestamp("compact")
        filename = secure_filename(image.filename or f"uploaded_image_{timestamp}")
        filepath = app.config["UPLOAD_FOLDER"] / filename
        image.save(str(filepath))
        saved_files.append(filepath)

    # Create Gemini client with provided parameters
    try:
        client = GeminiClient(
            project_id=project_id,
            location=location,
            output_dir=output_dir
        )
    except ValueError as e:
        return create_error_response(str(e), 400)

    # Generate image with optional upscaling
    try:
        generated_file = client.generate_hires_image_in_one_shot(
            prompt, saved_files, scale=scale
        )
    except (RuntimeError, OSError, ValueError) as e:
        return create_error_response(f"Image generation failed: {e}", 500)

    # Handle custom output filename if provided
    if generated_file and custom_output:
        try:
            custom_path = output_dir / custom_output
            generated_file.rename(custom_path)
            generated_file = custom_path
        except OSError as e:
            return create_error_response(f"Failed to rename output file: {e}", 500)

    return jsonify({
        "message": "Image generated successfully",
        "prompt": prompt,
        "project_id": project_id,
        "location": location,
        "scale": scale,
        "saved_files": [str(f) for f in saved_files],
        "generated_file": str(generated_file) if generated_file else None,
        "output_dir": str(output_dir),
        "upscaled": scale is not None
    }), 200


if __name__ == "__main__":
    import os
    # Only enable debug mode in development, never in production
    debug_mode = os.getenv("FLASK_DEBUG", "False").lower() in ("true", "1", "yes")
    app.run(debug=debug_mode)
