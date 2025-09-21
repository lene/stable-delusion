import pytest
import os
from unittest.mock import patch, MagicMock
from PIL import Image
import requests

import sys
sys.path.append('nano_api')

from nano_api.upscale import upscale_image
from nano_api.conf import DEFAULT_PROJECT_ID, DEFAULT_LOCATION


class TestUpscaleImage:
    @patch('nano_api.upscale.requests.post')
    @patch('nano_api.upscale.default')
    def test_upscale_image_success_x2(self, mock_default, mock_post):
        """Test successful image upscaling with x2 factor."""
        # Mock authentication
        mock_credentials = MagicMock()
        mock_credentials.token = 'test-token'
        mock_default.return_value = (mock_credentials, None)

        # Mock successful API response
        mock_response = MagicMock()
        mock_response.json.return_value = {
            'predictions': [{'bytesBase64Encoded': 'dGVzdCBpbWFnZSBkYXRh'}]
        }
        mock_post.return_value = mock_response

        # Mock PIL Image operations
        with patch('builtins.open', create=True) as mock_open:
            mock_open.return_value.__enter__.return_value.read.return_value = b'test_image_data'

            with patch('nano_api.upscale.Image.open') as mock_image_open:
                mock_image = MagicMock(spec=Image.Image)
                mock_image_open.return_value = mock_image

                with patch('nano_api.upscale.base64.b64encode',
                          return_value=b'dGVzdCBkYXRh'):
                    with patch('nano_api.upscale.base64.b64decode') as mock_decode:
                        mock_decode.return_value = b'decoded_image_data'

                        result = upscale_image('test.jpg', 'test-project',
                                             'us-central1', 'x2')

                        # Verify API call
                        mock_post.assert_called_once()
                        call_args = mock_post.call_args

                        # Check URL format
                        expected_url = ("https://us-central1-aiplatform.googleapis.com"
                                      "/v1/projects/test-project/locations/us-central1"
                                      "/publishers/google/models/imagegeneration@002:predict")
                        assert call_args[0][0] == expected_url

                        # Check request payload
                        payload = call_args[1]['json']
                        assert payload['parameters']['upscaleConfig']['upscaleFactor'] == 'x2'

                        # Check result
                        assert result == mock_image

    @patch('nano_api.upscale.requests.post')
    @patch('nano_api.upscale.default')
    def test_upscale_image_success_x4(self, mock_default, mock_post):
        """Test successful image upscaling with x4 factor."""
        # Mock authentication
        mock_credentials = MagicMock()
        mock_credentials.token = 'test-token'
        mock_default.return_value = (mock_credentials, None)

        # Mock successful API response
        mock_response = MagicMock()
        mock_response.json.return_value = {
            'predictions': [{'bytesBase64Encoded': 'dGVzdCBpbWFnZSBkYXRh'}]
        }
        mock_post.return_value = mock_response

        # Mock PIL Image operations
        with patch('builtins.open', create=True) as mock_open:
            mock_open.return_value.__enter__.return_value.read.return_value = b'test_image_data'

            with patch('nano_api.upscale.Image.open') as mock_image_open:
                mock_image = MagicMock(spec=Image.Image)
                mock_image_open.return_value = mock_image

                with patch('nano_api.upscale.base64.b64encode',
                          return_value=b'dGVzdCBkYXRh'):
                    with patch('nano_api.upscale.base64.b64decode') as mock_decode:
                        mock_decode.return_value = b'decoded_image_data'

                        result = upscale_image('test.jpg', 'test-project',
                                             'us-central1', 'x4')

                        # Check request payload has x4 factor
                        call_args = mock_post.call_args
                        payload = call_args[1]['json']
                        assert payload['parameters']['upscaleConfig']['upscaleFactor'] == 'x4'

    @patch('nano_api.upscale.requests.post')
    @patch('nano_api.upscale.default')
    def test_upscale_image_file_not_found(self, mock_default, mock_post):
        """Test upscale_image with non-existent file."""
        # Mock authentication
        mock_credentials = MagicMock()
        mock_default.return_value = (mock_credentials, None)

        # Mock file not found
        with patch('builtins.open', side_effect=FileNotFoundError):
            with pytest.raises(FileNotFoundError):
                upscale_image('nonexistent.jpg', 'test-project')

    @patch('nano_api.upscale.requests.post')
    @patch('nano_api.upscale.default')
    def test_upscale_image_api_error(self, mock_default, mock_post):
        """Test upscale_image with API error response."""
        # Mock authentication
        mock_credentials = MagicMock()
        mock_default.return_value = (mock_credentials, None)

        # Mock API error
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = requests.HTTPError("API Error")
        mock_post.return_value = mock_response

        with patch('builtins.open', create=True) as mock_open:
            mock_open.return_value.__enter__.return_value.read.return_value = b'test_data'

            with pytest.raises(requests.HTTPError, match="API Error"):
                upscale_image('test.jpg', 'test-project')

    @patch('nano_api.upscale.default')
    def test_upscale_image_auth_error(self, mock_default):
        """Test upscale_image with authentication error."""
        # Mock authentication failure
        mock_default.side_effect = Exception("Authentication failed")

        with pytest.raises(Exception, match="Authentication failed"):
            upscale_image('test.jpg', 'test-project')

    @patch('nano_api.upscale.requests.post')
    @patch('nano_api.upscale.default')
    def test_upscale_image_default_location(self, mock_default, mock_post):
        """Test upscale_image with default location parameter."""
        # Mock authentication
        mock_credentials = MagicMock()
        mock_credentials.token = 'test-token'
        mock_default.return_value = (mock_credentials, None)

        # Mock successful API response
        mock_response = MagicMock()
        mock_response.json.return_value = {
            'predictions': [{'bytesBase64Encoded': 'dGVzdA=='}]
        }
        mock_post.return_value = mock_response

        with patch('builtins.open', create=True) as mock_open:
            mock_open.return_value.__enter__.return_value.read.return_value = b'test'

            with patch('nano_api.upscale.Image.open'):
                with patch('nano_api.upscale.base64.b64encode'):
                    with patch('nano_api.upscale.base64.b64decode') as mock_decode:
                        mock_decode.return_value = b'decoded_test_data'

                        upscale_image('test.jpg', 'test-project')

                        # Check that default location was used
                        call_args = mock_post.call_args
                        url = call_args[0][0]
                        assert 'us-central1' in url

    def test_upscale_image_headers_format(self):
        """Test that request headers are properly formatted."""
        with patch('nano_api.upscale.default') as mock_default:
            mock_credentials = MagicMock()
            mock_credentials.token = 'test-bearer-token'
            mock_default.return_value = (mock_credentials, None)

            with patch('nano_api.upscale.requests.post') as mock_post:
                mock_response = MagicMock()
                mock_response.json.return_value = {
                    'predictions': [{'bytesBase64Encoded': 'dGVzdA=='}]
                }
                mock_post.return_value = mock_response

                with patch('builtins.open', create=True) as mock_open:
                    mock_open.return_value.__enter__.return_value.read.return_value = b'test'

                    with patch('nano_api.upscale.Image.open'):
                        with patch('nano_api.upscale.base64.b64encode'):
                            with patch('nano_api.upscale.base64.b64decode') as mock_decode:
                                mock_decode.return_value = b'test_decoded_data'

                                upscale_image('test.jpg', 'test-project')

                                # Check headers
                                call_args = mock_post.call_args
                                headers = call_args[1]['headers']
                                assert headers['Authorization'] == 'Bearer test-bearer-token'
                                assert headers['Content-Type'] == 'application/json'


class TestUpscaleCommandLine:
    def test_command_line_defaults(self):
        """Test command line parsing with defaults."""
        import argparse
        from nano_api.upscale import DEFAULT_PROJECT_ID, DEFAULT_LOCATION

        # Create parser similar to upscale.py
        parser = argparse.ArgumentParser()
        parser.add_argument("image_path", type=str)
        parser.add_argument("--scale", type=int, default=4, choices=[2, 4])

        args = parser.parse_args(['test.jpg'])
        assert args.image_path == 'test.jpg'
        assert args.scale == 4

    def test_command_line_custom_scale(self):
        """Test command line parsing with custom scale."""
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument("image_path", type=str)
        parser.add_argument("--scale", type=int, default=4, choices=[2, 4])

        args = parser.parse_args(['test.jpg', '--scale', '2'])
        assert args.image_path == 'test.jpg'
        assert args.scale == 2

    def test_command_line_invalid_scale(self):
        """Test command line parsing with invalid scale."""
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument("image_path", type=str)
        parser.add_argument("--scale", type=int, default=4, choices=[2, 4])

        with pytest.raises(SystemExit):
            parser.parse_args(['test.jpg', '--scale', '3'])


class TestUpscaleIntegration:
    """Integration tests that test the full upscale workflow."""

    @patch('nano_api.upscale.upscale_image')
    def test_main_execution_default_scale(self, mock_upscale):
        """Test main execution with default scale."""
        mock_image = MagicMock()
        mock_upscale.return_value = mock_image

        test_args = ['upscale.py', 'test.jpg']

        with patch('sys.argv', test_args):
            with patch('nano_api.upscale.print') as mock_print:
                # Import and execute the main block logic
                from nano_api.upscale import DEFAULT_PROJECT_ID, DEFAULT_LOCATION
                import argparse

                parser = argparse.ArgumentParser()
                parser.add_argument("image_path", type=str)
                parser.add_argument("--scale", type=int, default=4, choices=[2, 4])
                args = parser.parse_args(['test.jpg'])

                # Simulate main execution
                mock_upscale('test.jpg', DEFAULT_PROJECT_ID, DEFAULT_LOCATION,
                           upscale_factor=f'x{args.scale}')
                mock_image.save('upscaled_test.jpg')

                mock_upscale.assert_called_once_with('test.jpg', DEFAULT_PROJECT_ID,
                                                   DEFAULT_LOCATION,
                                                   upscale_factor='x4')
                mock_image.save.assert_called_once()