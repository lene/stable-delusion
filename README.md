# stable-delusion - AI-powered image generation and editing assistant

## Features

- **Image Generation**: Generate images from text prompts using Gemini 2.5 Flash Image Preview
- **Multi-image Support**: Use multiple reference images for generation
- **Automatic Upscaling**: Optional 2x or 4x upscaling using Google Cloud Vertex AI
- **Flexible Output**: Specify custom output directories and filenames
- **Storage Options**: Local filesystem or AWS S3 storage backends
- **Error Handling**: Comprehensive error logging and diagnostic information
- **Web API**: RESTful API for integration with other applications
- **Command Line Interface**: Full-featured CLI for batch processing and automation

## Installation

### From PyPI (Recommended for Users)

```bash
pip install stable-delusion
```

## Setup

### Configuration

The application uses environment variables for configuration. You can set these either through:
1. **`.env` file** (recommended) - Copy `.env.example` to `.env` and customize
2. **Environment variables** - Set directly in your shell (overrides .env values)

#### Option 1: Using .env File (Recommended)

```bash
# Copy the example file and edit it
cp .env.example .env

# Edit .env with your actual values
# At minimum, you need to set GEMINI_API_KEY
```

#### Option 2: Using Environment Variables

```bash
# Required: Gemini API key for image generation
export GEMINI_API_KEY="your-api-key-here"

# Optional: Flask debug mode (development only)
export FLASK_DEBUG="true"  # Enable debug mode in development
# export FLASK_DEBUG="false"  # Disable debug mode (default/production)
```

**Security Notes**:
- `FLASK_DEBUG` is disabled by default for security reasons. Only enable it in
  development environments, never in production.
- **Never commit your `.env` file to version control** - it contains sensitive
  information!
- The `.env` file is already in `.gitignore` to prevent accidental commits.

### AWS S3 Configuration (Optional)

The application supports storing generated images in AWS S3 instead of the local
filesystem. You can configure S3 either in your `.env` file or via environment
variables:

**Using .env file:**
```bash
# Add these to your .env file
STORAGE_TYPE=s3
AWS_S3_BUCKET=your-s3-bucket-name
AWS_S3_REGION=us-east-1
AWS_PROFILE=your-aws-profile  # or use direct credentials
```

**Using environment variables:**
```bash
# S3 Storage Configuration
export STORAGE_TYPE="s3"                    # Use "s3" for AWS S3, "local" for filesystem (default)
export AWS_S3_BUCKET="your-s3-bucket-name" # S3 bucket name for image storage
export AWS_S3_REGION="us-east-1"           # AWS region where your bucket is located

# AWS Credentials (use one of the following methods)
# Method 1: Environment variables
export AWS_ACCESS_KEY_ID="your-access-key"
export AWS_SECRET_ACCESS_KEY="your-secret-key"

# Method 2: AWS CLI profiles (recommended)
# Configure with: aws configure --profile your-profile
export AWS_PROFILE="your-profile"

# Method 3: IAM roles (for EC2/Lambda deployment)
# No additional configuration needed if running on AWS with proper IAM roles
```

**S3 Setup Requirements:**
1. Create an S3 bucket in your desired AWS region
2. Ensure your AWS credentials have the following permissions for the bucket:
   - `s3:PutObject` - Upload generated images
   - `s3:GetObject` - Download images (if needed)
   - `s3:DeleteObject` - Clean up old images
   - `s3:ListBucket` - List bucket contents

## Usage

### CLI

#### Basic usage
```bash
$ poetry run python stable_delusion/generate.py \
    --prompt "please make the women in the provided image look affectionately at each other" \
    --image samples/base.png
```

#### Advanced usage with all parameters
```bash
$ poetry run python stable_delusion/generate.py \
    --prompt "a futuristic cityscape with flying cars" \
    --image samples/base.png \
    --image samples/reference.png \
    --output-filename custom_output.png \
    --output-dir ./generated \
    --project-id my-gcp-project \
    --location us-central1 \
    --scale 4
```

#### S3 storage examples
```bash
# Use S3 storage (requires S3 environment variables to be set)
$ poetry run python stable_delusion/generate.py \
    --prompt "a beautiful landscape" \
    --image samples/base.png \
    --storage-type s3 \
    --output-dir generated-images

# Force local storage (override S3 configuration)
$ poetry run python stable_delusion/generate.py \
    --prompt "a city at night" \
    --image samples/base.png \
    --storage-type local \
    --output-dir ./local-output
```

#### Command line parameters
- `--prompt`: Text prompt for image generation (optional, defaults to sample
  prompt)
- `--image`: Path to reference image(s), can be used multiple times
- `--output-filename`: Output filename (default: "generated_gemini_image.png")
- `--output-dir`: Directory where generated files will be saved (default:
  current directory)
- `--project-id`: Google Cloud Project ID (defaults to value in conf.py)
- `--location`: Google Cloud region (defaults to value in conf.py)
- `--scale`: Upscale factor, 2 or 4 (optional, enables automatic upscaling)
- `--storage-type`: Storage backend - "local" for filesystem or "s3" for AWS S3
  (overrides configuration)

### Web server

#### Start the server
```bash
$ poetry run python stable_delusion/main.py
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

# S3 storage examples
# Save to S3 (requires S3 environment variables to be set)
$ curl -X POST \
    -F "prompt=a mountain landscape" \
    -F "images=@samples/base.png" \
    -F "storage_type=s3" \
    -F "output_dir=generated-images" \
    http://127.0.0.1:5000/generate

# Force local storage (override S3 configuration)
$ curl -X POST \
    -F "prompt=a city skyline" \
    -F "images=@samples/base.png" \
    -F "storage_type=local" \
    -F "output_dir=./local-results" \
    http://127.0.0.1:5000/generate
```

#### API Parameters

**Content-Type**: `multipart/form-data`

Parameters are sent as form fields (not JSON):

- `prompt`: Text prompt for image generation (required)
- `images`: Image file(s) to upload (required, can be multiple files)
- `output_dir`: Directory where generated files will be saved (optional,
  default: ".")
- `storage_type`: Storage backend - "local" for filesystem or "s3" for AWS S3
  (optional, uses configuration default)

#### API Response

**Content-Type**: `application/json`
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
$ poetry run python stable_delusion/upscale.py \
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

## Error Handling

The application provides detailed error logging when image generation fails:
- Safety filter violations with specific categories and probability levels
- API response diagnostics including token usage and finish reasons
- File upload details with metadata (size, MIME type, expiration times)
- Comprehensive error messages for troubleshooting

## Development

For development guidelines, code quality tools, CI/CD pipeline details, and
security best practices, see [doc/Development.md](doc/Development.md).

## Documentation

- **[Development Guide](doc/Development.md)** - Development practices, CI/CD
  pipeline, and code quality tools
- **[Architecture](doc/ARCHITECTURE.md)** - System architecture and design
  patterns
- **[API Demo](doc/API_DEMO.md)** - API endpoint examples and usage
- **[Changelog](CHANGELOG.md)** - Version history and release notes