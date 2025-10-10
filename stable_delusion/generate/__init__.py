"""
Image generation module for stable_delusion.
"""

# Import everything from the parent hallucinate.py module to maintain backward compatibility
# This is done by directly importing from the .py file
import importlib.util
import logging
import sys
from pathlib import Path

# Also import modules that are used by hallucinate.py for mocking/patching compatibility
from google import genai
from google.cloud import aiplatform
from PIL import Image

from stable_delusion import builders
from stable_delusion.client.gemini_client import GeminiClient, log_failure_reason

# Load hallucinate.py as a separate module to avoid circular imports
generate_py_path = Path(__file__).parent.parent / "hallucinate.py"
spec = importlib.util.spec_from_file_location("_generate_module", generate_py_path)
if spec and spec.loader:
    _generate_module = importlib.util.module_from_spec(spec)
    sys.modules["_generate_module"] = _generate_module
    spec.loader.exec_module(_generate_module)

    # Re-export all public functions and constants from hallucinate.py
    DEFAULT_PROMPT = _generate_module.DEFAULT_PROMPT
    GenerationConfig = _generate_module.GenerationConfig
    generate_from_images = _generate_module.generate_from_images
    save_response_image = _generate_module.save_response_image
    parse_command_line = _generate_module.parse_command_line
    main = _generate_module.main

    # Export private functions that are used in tests
    _validate_and_normalize_output_filename = (
        _generate_module._validate_and_normalize_output_filename
    )
    _create_cli_request_dto = _generate_module._create_cli_request_dto
    _handle_cli_custom_output = _generate_module._handle_cli_custom_output
    _log_generation_result = _generate_module._log_generation_result
    _execute_image_generation = _generate_module._execute_image_generation
    _process_cli_arguments = _generate_module._process_cli_arguments

__all__ = [
    "GeminiClient",
    "log_failure_reason",
    "DEFAULT_PROMPT",
    "GenerationConfig",
    "generate_from_images",
    "save_response_image",
    "parse_command_line",
    "main",
    "genai",
    "aiplatform",
    "Image",
    "builders",
    "logging",
]
