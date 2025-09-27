"""
Configuration constants for Google Cloud Project settings.
Defines default project ID and location for Gemini API and Vertex AI services.
"""

__author__ = "Lene Preuss <lene.preuss@gmail.com>"

# --- Configuration ---
# Replace with your Google Cloud Project ID and region
DEFAULT_PROJECT_ID = "gen-lang-client-0216779332"
DEFAULT_LOCATION = "us-central1"

# Valid scale factors for image upscaling
VALID_SCALE_FACTORS = [2, 4]

# Model configuration
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash-image-preview"
# DEFAULT_SEEDREAM_MODEL = "bytedance/seedream-v4-edit"
DEFAULT_SEEDREAM_MODEL = "seedream-4-0-250828"
DEFAULT_UPSCALE_MODEL = "imagegeneration@002"

# Supported models for image generation
SUPPORTED_MODELS = ["gemini", "seedream"]
