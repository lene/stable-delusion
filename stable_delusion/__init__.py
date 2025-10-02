"""
stable-delusion - AI-powered image generation and editing assistant.
Provides CLI and web API interfaces for Google Gemini and Seedream image generation.
Includes automatic upscaling capabilities via Google Vertex AI.
"""

__author__ = "Lene Preuss <lene.preuss@gmail.com>"

try:
    import importlib.metadata

    __version__ = importlib.metadata.version("stable-delusion")
except importlib.metadata.PackageNotFoundError:
    # Fallback for development environment
    __version__ = "0.1.0-dev"
