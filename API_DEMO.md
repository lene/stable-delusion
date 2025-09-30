# NanoAPIClient Enhanced API Demo

The Flask API has been updated to support all CLI functionality. Here are examples of the new capabilities:

## API Endpoints

### 1. Health Check
```bash
curl http://localhost:5000/health
```

Response:
```json
{
  "status": "healthy",
  "service": "NanoAPIClient",
  "version": "1.0.0"
}
```

### 2. API Information
```bash
curl http://localhost:5000/
```

Response:
```json
{
  "name": "NanoAPIClient API",
  "description": "Flask web API for image generation using Google Gemini AI",
  "version": "1.0.0",
  "endpoints": {
    "/": "API information",
    "/health": "Health check",
    "/generate": "Generate images from prompt and reference images",
    "/openapi.json": "OpenAPI specification"
  }
}
```

### 3. OpenAPI Specification
```bash
curl http://localhost:5000/openapi.json
```

Returns the complete OpenAPI 3.0.3 specification for the API.

### 4. Enhanced Image Generation

#### Basic Generation (equivalent to CLI default)
```bash
curl -X POST http://localhost:5000/generate \
  -F "images=@reference1.jpg" \
  -F "images=@reference2.png"
```

#### With Custom Prompt
```bash
curl -X POST http://localhost:5000/generate \
  -F "prompt=A futuristic cityscape at sunset" \
  -F "images=@reference1.jpg"
```

#### With Custom Project Configuration
```bash
curl -X POST http://localhost:5000/generate \
  -F "prompt=Generate a landscape" \
  -F "project_id=my-custom-project" \
  -F "location=us-west1" \
  -F "images=@reference1.jpg"
```

#### With Upscaling
```bash
curl -X POST http://localhost:5000/generate \
  -F "prompt=Generate artwork" \
  -F "scale=4" \
  -F "images=@reference1.jpg" \
  -F "images=@reference2.jpg"
```

#### With Custom Output Settings
```bash
curl -X POST http://localhost:5000/generate \
  -F "prompt=Create an image" \
  -F "output=my_custom_image.png" \
  -F "output_dir=./output" \
  -F "images=@reference1.jpg"
```

#### Complete Example with All Parameters
```bash
curl -X POST http://localhost:5000/generate \
  -F "prompt=A beautiful mountain landscape with aurora" \
  -F "project_id=my-project-123" \
  -F "location=us-central1" \
  -F "scale=2" \
  -F "output=mountain_aurora.png" \
  -F "output_dir=./generated_images" \
  -F "images=@mountain_ref.jpg" \
  -F "images=@aurora_ref.png"
```

## API Response Format

The enhanced `/generate` endpoint returns detailed information:

```json
{
  "message": "Image generated successfully",
  "prompt": "A beautiful mountain landscape with aurora",
  "project_id": "my-project-123",
  "location": "us-central1",
  "scale": 2,
  "saved_files": [
    "uploads/mountain_ref_20240101-120000.jpg",
    "uploads/aurora_ref_20240101-120001.png"
  ],
  "generated_file": "./generated_images/mountain_aurora.png",
  "output_dir": "./generated_images",
  "upscaled": true
}
```

## Parameter Mapping: CLI vs API

| CLI Parameter | API Parameter | Description | Default |
|---------------|---------------|-------------|---------|
| `--prompt` | `prompt` | Text prompt for generation | Uses DEFAULT_PROMPT if not provided |
| `--image` (multiple) | `images` | Reference images | Required (at least one) |
| `--project-id` | `project_id` | Google Cloud Project ID | From conf.py |
| `--location` | `location` | Google Cloud region | From conf.py |
| `--scale` | `scale` | Upscale factor (2 or 4) | None (no upscaling) |
| `--output-filename` | `output_filename` | Custom output filename | Timestamp-based |
| `--output-dir` | `output_dir` | Output directory | Current directory |

## Error Handling

The API provides detailed error messages:

```json
{
  "error": "Scale must be 2 or 4"
}
```

Common error scenarios:
- Missing images: "Missing 'images' parameter"
- Invalid scale: "Scale must be 2 or 4"
- Invalid scale format: "Scale must be an integer"
- API key issues: "GEMINI_API_KEY environment variable is required but not set"
- File operation errors: "Failed to rename output file: ..."

## OpenAPI Specification

The API includes a complete OpenAPI 3.0.3 specification available at `/openapi.json`, which includes:

- Complete endpoint documentation
- Request/response schemas
- Parameter validation rules
- Example requests and responses
- Error response formats

This specification can be used with tools like Swagger UI, Postman, or code generators.