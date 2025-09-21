__author__ = 'Lene Preuss <lene.preuss@gmail.com>'

from pathlib import Path

from flask import Flask, jsonify, request
from werkzeug.utils import secure_filename

from generate import generate_from_images


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = Path('uploads')
app.config['UPLOAD_FOLDER'].mkdir(exist_ok=True)


@app.route('/generate', methods=['POST'])
def generate():
    # Get the prompt parameter
    prompt = request.form.get('prompt')
    if not prompt:
        return jsonify({"error": "Missing 'prompt' parameter"}), 400

    # Get the uploaded files
    if 'images' not in request.files:
        return jsonify({"error": "Missing 'images' parameter"}), 400

    # Get optional output directory parameter
    output_dir = Path(request.form.get('output_dir', '.'))

    images = request.files.getlist('images')
    saved_files = []

    # Save uploaded files
    for image in images:
        filename = secure_filename(image.filename)
        filepath = app.config['UPLOAD_FOLDER'] / filename
        image.save(str(filepath))
        saved_files.append(filepath)

    generated_file = generate_from_images(
        prompt, saved_files, output_dir=output_dir
    )

    # Example response
    return jsonify({
        "message": "Files uploaded successfully",
        "prompt": prompt,
        "saved_files": [str(f) for f in saved_files],
        "generated_file": str(generated_file) if generated_file else None,
        "output_dir": str(output_dir)
    })


if __name__ == '__main__':
    app.run(debug=True)
