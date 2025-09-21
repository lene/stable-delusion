# CLI client and web server to easily use nano banana image editing model

## Installation

```bash
poetry install
```

## Setup

Set your Gemini API key as an environment variable:
```bash
export GEMINI_API_KEY="your-api-key-here"
```

## Usage

### CLI

#### Basic usage
```bash
$ poetry run python nano_api/generate.py \
    --prompt "please make the women in the provided image look affectionately at each other" \
    --image samples/base.png
```

#### Advanced usage with all parameters
```bash
$ poetry run python nano_api/generate.py \
    --prompt "a futuristic cityscape with flying cars" \
    --image samples/base.png \
    --image samples/reference.png \
    --output custom_output.png \
    --output-dir ./generated \
    --project-id my-gcp-project \
    --location us-central1 \
    --scale 4
```

#### Command line parameters
- `--prompt`: Text prompt for image generation (optional, defaults to sample prompt)
- `--image`: Path to reference image(s), can be used multiple times
- `--output`: Output filename (default: "generated_gemini_image.png")
- `--output-dir`: Directory where generated files will be saved (default: current directory)
- `--project-id`: Google Cloud Project ID (defaults to value in conf.py)
- `--location`: Google Cloud region (defaults to value in conf.py)
- `--scale`: Upscale factor, 2 or 4 (optional, enables automatic upscaling)

### Web server

#### Start the server
```bash
$ poetry run python nano_api/main.py
```

#### Make a request to the web API
```bash
# Basic request
$ curl -X POST \
    -F "prompt=please make the women in the provided image look affectionately at each other" \
    -F "images=@samples/base_2.png" \
    http://127.0.0.1:5000/generate

# Request with custom output directory
$ curl -X POST \
    -F "prompt=create a sunset landscape" \
    -F "images=@samples/base.png" \
    -F "output_dir=./api_generated" \
    http://127.0.0.1:5000/generate

# Multiple images
$ curl -X POST \
    -F "prompt=blend these images creatively" \
    -F "images=@samples/image1.png" \
    -F "images=@samples/image2.png" \
    -F "output_dir=./results" \
    http://127.0.0.1:5000/generate
```

#### API Parameters
- `prompt`: Text prompt for image generation (required)
- `images`: Image file(s) to upload (required, can be multiple)
- `output_dir`: Directory where generated files will be saved (optional, default: ".")

#### API Response
```json
{
    "message": "Files uploaded successfully",
    "prompt": "your prompt text",
    "saved_files": ["/path/to/uploaded/file1.png", "/path/to/uploaded/file2.png"],
    "generated_file": "/path/to/generated_image.png",
    "output_dir": "/custom/output/directory"
}
```

### Upscale generated images

#### Setup for upscaling
Preliminaries to get permissions sorted out:
```bash
$ gcloud init
$ gcloud auth login
$ gcloud auth application-default login
$ gcloud services enable aiplatform.googleapis.com
```

#### Upscale a specific image
```bash
$ poetry run python nano_api/upscale.py \
    --input generated_image.png \
    --scale 4 \
    --project-id my-gcp-project \
    --location us-central1
```

#### Upscale parameters
- `--input`: Input image file to upscale (required)
- `--scale`: Upscale factor, 2 or 4 (default: 2)
- `--project-id`: Google Cloud Project ID (defaults to value in conf.py)
- `--location`: Google Cloud region (defaults to value in conf.py)

## Features

- **Image Generation**: Generate images from text prompts using Gemini 2.5 Flash Image Preview
- **Multi-image Support**: Use multiple reference images for generation
- **Automatic Upscaling**: Optional 2x or 4x upscaling using Google Cloud Vertex AI
- **Flexible Output**: Specify custom output directories and filenames
- **Error Handling**: Comprehensive error logging and diagnostic information
- **Web API**: RESTful API for integration with other applications
- **Command Line Interface**: Full-featured CLI for batch processing and automation

## Error Handling

The application provides detailed error logging when image generation fails:
- Safety filter violations with specific categories and probability levels
- API response diagnostics including token usage and finish reasons
- File upload details with metadata (size, MIME type, expiration times)
- Comprehensive error messages for troubleshooting

## Testing

Run the comprehensive test suite:
```bash
# Run all tests
$ poetry run pytest tests/ -v

# Run specific test categories
$ poetry run pytest tests/unit/ -v
$ poetry run pytest tests/integration/ -v
```