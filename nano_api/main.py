"""
Flask web API server for image generation services.
Provides REST endpoints for uploading images and generating new images with Gemini AI.
Supports multi-image input and custom output directories.
"""

__author__ = "Lene Preuss <lene.preuss@gmail.com>"

from pathlib import Path
from typing import Tuple

from flask import Flask, jsonify, request, Response
from werkzeug.utils import secure_filename

from nano_api.generate import generate_from_images
from nano_api.utils import create_error_response, get_current_timestamp


app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = Path("uploads")
app.config["UPLOAD_FOLDER"].mkdir(exist_ok=True)


@app.route("/generate", methods=["POST"])
def generate() -> Tuple[Response, int]:
    # Validate required parameters
    prompt = request.form.get("prompt")
    if not prompt:
        return create_error_response("Missing 'prompt' parameter")

    if "images" not in request.files:
        return create_error_response("Missing 'images' parameter")

    # Get optional output directory parameter
    output_dir = Path(request.form.get("output_dir", "."))
    images = request.files.getlist("images")
    saved_files = []

    # Save uploaded files with utility functions
    for image in images:
        timestamp = get_current_timestamp("compact")
        filename = secure_filename(image.filename or f"uploaded_image_{timestamp}")
        filepath = app.config["UPLOAD_FOLDER"] / filename
        image.save(str(filepath))
        saved_files.append(filepath)

    generated_file = generate_from_images(
        prompt, saved_files, output_dir=output_dir
    )

    return jsonify({
        "message": "Files uploaded successfully",
        "prompt": prompt,
        "saved_files": [str(f) for f in saved_files],
        "generated_file": str(generated_file) if generated_file else None,
        "output_dir": str(output_dir)
    }), 200


if __name__ == "__main__":
    import os
    # Only enable debug mode in development, never in production
    debug_mode = os.getenv("FLASK_DEBUG", "False").lower() in ("true", "1", "yes")
    app.run(debug=debug_mode)
