__author__ = 'Lene Preuss <lene.preuss@gmail.com>'
import os

from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename

from generate import multi_image_example


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.route('/generate', methods=['POST'])
def generate():
    # Get the prompt parameter
    prompt = request.form.get('prompt')
    if not prompt:
        return jsonify({"error": "Missing 'prompt' parameter"}), 400

    # Get the uploaded files
    if 'images' not in request.files:
        return jsonify({"error": "Missing 'images' parameter"}), 400

    images = request.files.getlist('images')
    saved_files = []

    # Save uploaded files
    for image in images:
        filename = secure_filename(image.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        image.save(filepath)
        saved_files.append(filepath)

    generated_file = multi_image_example(prompt, saved_files)

    # Example response
    return jsonify({
        "message": "Files uploaded successfully",
        "prompt": prompt,
        "saved_files": saved_files,
        "generated_file": generated_file
    })

if __name__ == '__main__':
    app.run(debug=True)