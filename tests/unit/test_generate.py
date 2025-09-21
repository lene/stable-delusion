import pytest
import os
import argparse
from unittest.mock import patch, MagicMock
from io import BytesIO

import sys
sys.path.append('nano_api')

from nano_api.generate import (
    GeminiClient, parse_command_line, save_response_image,
    multi_image_example
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

            with patch.object(client, 'multi_image_example',
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

            with patch.object(client, 'multi_image_example',
                            return_value='test_image.png'):
                result = client.generate_hires_image_in_one_shot("prompt",
                                                               ["image.png"],
                                                               scale=4)

                mock_upscale.assert_called_once_with('preview_image.png',
                                                   client.project_id,
                                                   client.location,
                                                   upscale_factor='x4')
                mock_upscaled_image.save.assert_called_once()
                assert result == 'upscaled_test_image.png'


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


class TestMultiImageExample:
    @patch('nano_api.generate.GeminiClient')
    def test_multi_image_example_defaults(self, mock_client_class):
        """Test standalone multi_image_example function with defaults."""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        mock_client.multi_image_example.return_value = 'test_result.png'

        result = multi_image_example("test prompt", ["image.png"])

        mock_client_class.assert_called_once_with(project_id=DEFAULT_PROJECT_ID,
                                                location=DEFAULT_LOCATION)
        mock_client.multi_image_example.assert_called_once_with("test prompt",
                                                              ["image.png"])
        assert result == 'test_result.png'

    @patch('nano_api.generate.GeminiClient')
    def test_multi_image_example_custom_params(self, mock_client_class):
        """Test standalone multi_image_example function with custom parameters."""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        mock_client.multi_image_example.return_value = 'test_result.png'

        custom_project = "custom-project"
        custom_location = "custom-location"

        result = multi_image_example("test prompt", ["image.png"],
                                   project_id=custom_project,
                                   location=custom_location)

        mock_client_class.assert_called_once_with(project_id=custom_project,
                                                location=custom_location)
        assert result == 'test_result.png'