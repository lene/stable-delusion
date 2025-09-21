import pytest
import os
import argparse
from unittest.mock import patch, MagicMock
from io import BytesIO

import sys
sys.path.append('nano_api')

from nano_api.generate import (
    GeminiClient, parse_command_line, save_response_image,
    generate_from_images
)
from nano_api.conf import DEFAULT_PROJECT_ID, DEFAULT_LOCATION


class TestGeminiClient:
    def test_init_missing_api_key(self):
        """Test GeminiClient initialization without GEMINI_API_KEY."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError,
                             match="GEMINI_API_KEY environment variable is required"):
                GeminiClient()

    @patch('nano_api.generate.genai.Client')
    @patch('nano_api.generate.aiplatform.init')
    def test_init_with_defaults(self, mock_init, mock_client):
        """Test GeminiClient initialization with default parameters."""
        with patch.dict(os.environ, {'GEMINI_API_KEY': 'test-key'}):
            client = GeminiClient()
            assert client.project_id == DEFAULT_PROJECT_ID
            assert client.location == DEFAULT_LOCATION
            mock_init.assert_called_once_with(project=DEFAULT_PROJECT_ID,
                                            location=DEFAULT_LOCATION)

    @patch('nano_api.generate.genai.Client')
    @patch('nano_api.generate.aiplatform.init')
    def test_init_with_custom_params(self, mock_init, mock_client):
        """Test GeminiClient initialization with custom parameters."""
        custom_project = "custom-project"
        custom_location = "custom-location"

        with patch.dict(os.environ, {'GEMINI_API_KEY': 'test-key'}):
            client = GeminiClient(project_id=custom_project,
                                location=custom_location)
            assert client.project_id == custom_project
            assert client.location == custom_location
            mock_init.assert_called_once_with(project=custom_project,
                                            location=custom_location)

    @patch('nano_api.generate.genai.Client')
    @patch('nano_api.generate.aiplatform.init')
    def test_upload_files_success(self, mock_init, mock_client):
        """Test successful file upload."""
        with patch.dict(os.environ, {'GEMINI_API_KEY': 'test-key'}):
            client = GeminiClient()
            mock_upload = MagicMock()
            client.client.files.upload = mock_upload

            with patch('os.path.isfile', return_value=True):
                image_paths = ['test1.png', 'test2.png']
                result = client.upload_files(image_paths)

                assert mock_upload.call_count == 2
                assert len(result) == 2

    @patch('nano_api.generate.genai.Client')
    @patch('nano_api.generate.aiplatform.init')
    def test_upload_files_not_found(self, mock_init, mock_client):
        """Test file upload with missing file."""
        with patch.dict(os.environ, {'GEMINI_API_KEY': 'test-key'}):
            client = GeminiClient()

            with patch('os.path.isfile', return_value=False):
                image_paths = ['missing.png']
                with pytest.raises(FileNotFoundError,
                                 match="Image file not found: missing.png"):
                    client.upload_files(image_paths)

    @patch('nano_api.generate.genai.Client')
    @patch('nano_api.generate.aiplatform.init')
    def test_generate_hires_image_without_scale(self, mock_init, mock_client):
        """Test generate_hires_image_in_one_shot without scaling."""
        with patch.dict(os.environ, {'GEMINI_API_KEY': 'test-key'}):
            client = GeminiClient()

            with patch.object(client, 'generate_from_images',
                            return_value='test_image.png'):
                result = client.generate_hires_image_in_one_shot("prompt",
                                                               ["image.png"])
                assert result == 'test_image.png'

    @patch('nano_api.generate.genai.Client')
    @patch('nano_api.generate.aiplatform.init')
    @patch('nano_api.generate.upscale_image')
    def test_generate_hires_image_with_scale(self, mock_upscale, mock_init,
                                           mock_client):
        """Test generate_hires_image_in_one_shot with scaling."""
        with patch.dict(os.environ, {'GEMINI_API_KEY': 'test-key'}):
            client = GeminiClient()
            mock_upscaled_image = MagicMock()
            mock_upscale.return_value = mock_upscaled_image

            with patch.object(client, 'generate_from_images',
                            return_value='test_image.png'):
                result = client.generate_hires_image_in_one_shot("prompt",
                                                               ["image.png"],
                                                               scale=4)

                mock_upscale.assert_called_once_with('test_image.png',
                                                   client.project_id,
                                                   client.location,
                                                   upscale_factor='x4')
                mock_upscaled_image.save.assert_called_once()
                assert result == 'upscaled_test_image.png'

    @patch('nano_api.generate.genai.Client')
    @patch('nano_api.generate.aiplatform.init')
    def test_generate_from_images_no_candidates(self, mock_init, mock_client):
        """Test generate_from_images when API returns no candidates."""
        with patch.dict(os.environ, {'GEMINI_API_KEY': 'test-key'}):
            # Mock response with no candidates
            mock_response = MagicMock()
            mock_response.candidates = []

            mock_client_instance = MagicMock()
            mock_client_instance.models.generate_content.return_value = mock_response
            mock_client_instance.files.upload.return_value = MagicMock()
            mock_client.return_value = mock_client_instance

            client = GeminiClient()

            with patch.object(client, 'log_failure_reason') as mock_log_failure:
                with patch('os.path.isfile', return_value=True):
                    result = client.generate_from_images("test prompt", ["image.png"])

                    assert result is None
                    mock_log_failure.assert_called_once_with(mock_response)

    @patch('nano_api.generate.genai.Client')
    @patch('nano_api.generate.aiplatform.init')
    def test_log_failure_reason_with_prompt_feedback(self, mock_init, mock_client):
        """Test log_failure_reason with prompt feedback and safety ratings."""
        with patch.dict(os.environ, {'GEMINI_API_KEY': 'test-key'}):
            client = GeminiClient()

            # Mock response with prompt feedback
            mock_response = MagicMock()
            mock_feedback = MagicMock()
            mock_feedback.block_reason = "SAFETY"

            # Mock safety ratings
            mock_rating1 = MagicMock()
            mock_rating1.category = "HATE_SPEECH"
            mock_rating1.probability = "HIGH"
            mock_rating2 = MagicMock()
            mock_rating2.category = "VIOLENCE"
            mock_rating2.probability = "MEDIUM"

            mock_feedback.safety_ratings = [mock_rating1, mock_rating2]
            mock_response.prompt_feedback = mock_feedback
            mock_response.usage_metadata = MagicMock()

            with patch('nano_api.generate.logging.error') as mock_log_error:
                client.log_failure_reason(mock_response)

                # Verify logging calls
                mock_log_error.assert_any_call("No candidates returned from the API.")
                mock_log_error.assert_any_call("Prompt blocked: SAFETY")
                mock_log_error.assert_any_call("Safety rating: HATE_SPEECH = HIGH")
                mock_log_error.assert_any_call("Safety rating: VIOLENCE = MEDIUM")

    @patch('nano_api.generate.genai.Client')
    @patch('nano_api.generate.aiplatform.init')
    def test_log_failure_reason_minimal_response(self, mock_init, mock_client):
        """Test log_failure_reason with minimal response data."""
        with patch.dict(os.environ, {'GEMINI_API_KEY': 'test-key'}):
            client = GeminiClient()

            # Mock response with minimal data
            mock_response = MagicMock()
            mock_response.prompt_feedback = None
            mock_response.usage_metadata = None

            with patch('nano_api.generate.logging.error') as mock_log_error:
                client.log_failure_reason(mock_response)

                # Should still log basic information
                mock_log_error.assert_any_call("No candidates returned from the API.")
                assert mock_log_error.call_count >= 3  # Basic logs + type + attributes

    @patch('nano_api.generate.genai.Client')
    @patch('nano_api.generate.aiplatform.init')
    def test_upload_files_enhanced_logging(self, mock_init, mock_client):
        """Test upload_files with enhanced logging of file details."""
        with patch.dict(os.environ, {'GEMINI_API_KEY': 'test-key'}):
            # Mock uploaded file with all attributes
            mock_uploaded_file = MagicMock()
            mock_uploaded_file.name = "test_uploaded_file.png"
            mock_uploaded_file.mime_type = "image/png"
            mock_uploaded_file.size_bytes = 1024
            mock_uploaded_file.create_time.strftime.return_value = "2024-01-01 12:00:00"
            mock_uploaded_file.expiration_time.strftime.return_value = "2024-01-02 12:00:00"
            mock_uploaded_file.uri = "gs://test-bucket/test_file"

            mock_client_instance = MagicMock()
            mock_client_instance.files.upload.return_value = mock_uploaded_file
            mock_client.return_value = mock_client_instance

            client = GeminiClient()

            with patch('os.path.isfile', return_value=True):
                with patch('nano_api.generate.logging.info') as mock_log_info:
                    result = client.upload_files(["test.png"])

                    assert len(result) == 1
                    assert result[0] == mock_uploaded_file

                    # Verify enhanced logging
                    expected_log = ("Uploaded file: test.png -> name=test_uploaded_file.png, "
                                  "mime_type=image/png, size_bytes=1024, "
                                  "create_time=2024-01-01 12:00:00, "
                                  "expiration_time=2024-01-02 12:00:00, "
                                  "uri=gs://test-bucket/test_file")
                    mock_log_info.assert_called_with(expected_log)

    @patch('nano_api.generate.genai.Client')
    @patch('nano_api.generate.aiplatform.init')
    @patch('nano_api.generate.upscale_image')
    def test_generate_hires_uses_actual_filename(self, mock_upscale, mock_init, mock_client):
        """Test that generate_hires_image_in_one_shot uses actual preview filename, not hardcoded."""
        with patch.dict(os.environ, {'GEMINI_API_KEY': 'test-key'}):
            client = GeminiClient()

            # Mock upscaling
            mock_upscaled_image = MagicMock()
            mock_upscale.return_value = mock_upscaled_image

            # Mock generate_from_images to return a specific filename
            actual_filename = "generated_2024-01-01-15:30:45.png"
            with patch.object(client, 'generate_from_images', return_value=actual_filename):
                result = client.generate_hires_image_in_one_shot("test prompt",
                                                               ["image.png"],
                                                               scale=2)

                # Verify upscale_image is called with the actual filename, not hardcoded
                mock_upscale.assert_called_once_with(actual_filename,
                                                   client.project_id,
                                                   client.location,
                                                   upscale_factor='x2')

                expected_result = f"upscaled_{actual_filename}"
                assert result == expected_result


class TestParseCommandLine:
    def test_parse_command_line_defaults(self):
        """Test parse_command_line with no arguments."""
        with patch('sys.argv', ['generate.py']):
            args = parse_command_line()
            assert args.prompt is None
            assert args.output == "generated_gemini_image.png"
            assert args.image is None
            assert args.project_id is None
            assert args.location is None
            assert args.scale is None

    def test_parse_command_line_all_args(self):
        """Test parse_command_line with all arguments."""
        test_args = ['generate.py', '--prompt', 'test prompt',
                    '--output', 'test_output.png',
                    '--image', 'image1.png', '--image', 'image2.png',
                    '--project-id', 'test-project',
                    '--location', 'test-location',
                    '--scale', '4']

        with patch('sys.argv', test_args):
            args = parse_command_line()
            assert args.prompt == 'test prompt'
            assert args.output == 'test_output.png'
            assert args.image == ['image1.png', 'image2.png']
            assert args.project_id == 'test-project'
            assert args.location == 'test-location'
            assert args.scale == 4

    def test_parse_command_line_invalid_scale(self):
        """Test parse_command_line with invalid scale value."""
        test_args = ['generate.py', '--scale', '3']

        with patch('sys.argv', test_args):
            with pytest.raises(SystemExit):
                parse_command_line()


class TestSaveResponseImage:
    def test_save_response_image_success(self):
        """Test successful image saving from response."""
        mock_response = MagicMock()
        mock_candidate = MagicMock()
        mock_part = MagicMock()
        mock_part.text = None
        mock_part.inline_data = MagicMock()
        mock_part.inline_data.data = b'fake_image_data'

        mock_candidate.content.parts = [mock_part]
        mock_response.candidates = [mock_candidate]

        with patch('nano_api.generate.Image.open') as mock_open:
            with patch('nano_api.generate.datetime') as mock_datetime:
                mock_datetime.now.return_value.strftime.return_value = '2024-01-01-12:00:00'
                mock_image = MagicMock()
                mock_open.return_value = mock_image

                result = save_response_image(mock_response)

                assert result == 'generated_2024-01-01-12:00:00.png'
                mock_image.save.assert_called_once_with('generated_2024-01-01-12:00:00.png')

    def test_save_response_image_no_image(self):
        """Test response with no image data."""
        mock_response = MagicMock()
        mock_candidate = MagicMock()
        mock_part = MagicMock()
        mock_part.text = "No image generated"
        mock_part.inline_data = None

        mock_candidate.content.parts = [mock_part]
        mock_response.candidates = [mock_candidate]

        result = save_response_image(mock_response)
        assert result is None

    def test_save_response_image_text_part(self):
        """Test response with text part logging."""
        mock_response = MagicMock()
        mock_candidate = MagicMock()
        mock_part = MagicMock()
        mock_part.text = "Generated successfully"
        mock_part.inline_data = None

        mock_candidate.content.parts = [mock_part]
        mock_response.candidates = [mock_candidate]

        with patch('nano_api.generate.logging.info') as mock_log:
            result = save_response_image(mock_response)
            mock_log.assert_called_with("Generated successfully")
            assert result is None


class TestGenerateFromImages:
    @patch('nano_api.generate.GeminiClient')
    def test_generate_from_images_defaults(self, mock_client_class):
        """Test standalone generate_from_images function with defaults."""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        mock_client.generate_from_images.return_value = 'test_result.png'

        result = generate_from_images("test prompt", ["image.png"])

        mock_client_class.assert_called_once_with(project_id=DEFAULT_PROJECT_ID,
                                                location=DEFAULT_LOCATION)
        mock_client.generate_from_images.assert_called_once_with("test prompt",
                                                              ["image.png"])
        assert result == 'test_result.png'

    @patch('nano_api.generate.GeminiClient')
    def test_generate_from_images_custom_params(self, mock_client_class):
        """Test standalone generate_from_images function with custom parameters."""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        mock_client.generate_from_images.return_value = 'test_result.png'

        custom_project = "custom-project"
        custom_location = "custom-location"

        result = generate_from_images("test prompt", ["image.png"],
                                   project_id=custom_project,
                                   location=custom_location)

        mock_client_class.assert_called_once_with(project_id=custom_project,
                                                location=custom_location)
        assert result == 'test_result.png'