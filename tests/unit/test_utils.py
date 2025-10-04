"""
Tests for utility functions in stable_delusion.utils.
"""

import tempfile
from pathlib import Path
from unittest.mock import patch
from PIL import Image
import pytest

from stable_delusion.utils import (
    optimize_image_size,
    _get_file_size_mb,
    _convert_to_jpeg_with_quality,
    _find_optimal_jpeg_quality,
)
from stable_delusion.exceptions import FileOperationError


class TestImageOptimization:
    """Tests for image size optimization functions."""

    def test_get_file_size_mb(self):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as f:
            f.write(b"x" * (2 * 1024 * 1024))
            temp_path = Path(f.name)

        try:
            size_mb = _get_file_size_mb(temp_path)
            assert 1.9 < size_mb < 2.1
        finally:
            temp_path.unlink()

    def test_convert_to_jpeg_with_quality(self):
        img = Image.new("RGB", (100, 100), color="red")

        jpeg_bytes_high = _convert_to_jpeg_with_quality(img, 95)
        jpeg_bytes_low = _convert_to_jpeg_with_quality(img, 50)

        assert isinstance(jpeg_bytes_high, bytes)
        assert isinstance(jpeg_bytes_low, bytes)
        assert len(jpeg_bytes_high) > len(jpeg_bytes_low)

    def test_convert_rgba_to_jpeg(self):
        img = Image.new("RGBA", (100, 100), color=(255, 0, 0, 128))

        jpeg_bytes = _convert_to_jpeg_with_quality(img, 95)

        assert isinstance(jpeg_bytes, bytes)
        assert len(jpeg_bytes) > 0

    def test_find_optimal_jpeg_quality_small_image(self):
        img = Image.new("RGB", (100, 100), color="blue")

        jpeg_bytes = _find_optimal_jpeg_quality(img, max_size_mb=1.0)

        size_mb = len(jpeg_bytes) / (1024 * 1024)
        assert size_mb < 1.0

    def test_find_optimal_jpeg_quality_large_target(self):
        img = Image.new("RGB", (500, 500), color="green")

        jpeg_bytes = _find_optimal_jpeg_quality(img, max_size_mb=10.0)

        size_mb = len(jpeg_bytes) / (1024 * 1024)
        assert size_mb < 10.0

    def test_optimize_image_size_below_threshold(self):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as f:
            img = Image.new("RGB", (100, 100), color="red")
            img.save(f.name, format="JPEG", quality=95)
            temp_path = Path(f.name)

        try:
            result_path = optimize_image_size(temp_path, max_size_mb=7.0)
            assert result_path == temp_path
        finally:
            temp_path.unlink(missing_ok=True)

    def test_optimize_image_size_above_threshold(self):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as f:
            img = Image.new("RGB", (5000, 5000), color="blue")
            img.save(f.name, format="PNG", compress_level=0)
            temp_path = Path(f.name)

        try:
            original_size_mb = _get_file_size_mb(temp_path)
            assert original_size_mb > 7.0

            result_path = optimize_image_size(temp_path, max_size_mb=7.0)

            assert result_path != temp_path
            assert result_path.exists()
            assert result_path.suffix == ".jpg"

            optimized_size_mb = _get_file_size_mb(result_path)
            assert optimized_size_mb < 7.0
            assert optimized_size_mb < original_size_mb

            result_path.unlink()
        finally:
            temp_path.unlink(missing_ok=True)

    def test_optimize_image_size_preserves_content(self):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as f:
            img = Image.new("RGB", (2000, 2000), color=(100, 150, 200))
            img.save(f.name, format="PNG")
            temp_path = Path(f.name)

        try:
            result_path = optimize_image_size(temp_path, max_size_mb=5.0)

            if result_path != temp_path:
                with Image.open(result_path) as optimized_img:
                    assert optimized_img.size == (2000, 2000)
                    assert optimized_img.mode == "RGB"

                result_path.unlink()
        finally:
            temp_path.unlink(missing_ok=True)

    def test_optimize_image_size_custom_threshold(self):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as f:
            img = Image.new("RGB", (1500, 1500), color="yellow")
            img.save(f.name, format="PNG")
            temp_path = Path(f.name)

        try:
            result_path = optimize_image_size(temp_path, max_size_mb=1.0)

            if result_path != temp_path:
                optimized_size_mb = _get_file_size_mb(result_path)
                assert optimized_size_mb < 1.0
                result_path.unlink()
        finally:
            temp_path.unlink(missing_ok=True)

    def test_optimize_image_size_webp_input(self):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".webp") as f:
            img = Image.new("RGB", (2000, 2000), color="purple")
            img.save(f.name, format="WEBP")
            temp_path = Path(f.name)

        try:
            original_size_mb = _get_file_size_mb(temp_path)

            if original_size_mb > 7.0:
                result_path = optimize_image_size(temp_path, max_size_mb=7.0)
                assert result_path.suffix == ".jpg"
                optimized_size_mb = _get_file_size_mb(result_path)
                assert optimized_size_mb < 7.0
                result_path.unlink()
        finally:
            temp_path.unlink(missing_ok=True)

    def test_optimize_image_size_handles_write_error(self):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as f:
            img = Image.new("RGB", (5000, 5000), color="red")
            img.save(f.name, format="PNG", compress_level=0)
            temp_path = Path(f.name)

        try:
            original_size_mb = _get_file_size_mb(temp_path)
            if original_size_mb <= 5.0:
                pytest.skip("Test image not large enough to trigger optimization")

            with patch("pathlib.Path.write_bytes", side_effect=OSError("Disk full")):
                with pytest.raises(FileOperationError, match="Failed to save optimized image"):
                    optimize_image_size(temp_path, max_size_mb=5.0)
        finally:
            temp_path.unlink(missing_ok=True)

    def test_optimize_image_size_invalid_image(self):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as f:
            f.write(b"not an image" * 10 * 1024 * 1024)
            temp_path = Path(f.name)

        try:
            file_size_mb = _get_file_size_mb(temp_path)
            if file_size_mb > 7.0:
                with pytest.raises(FileOperationError):
                    optimize_image_size(temp_path, max_size_mb=7.0)
            else:
                result = optimize_image_size(temp_path, max_size_mb=7.0)
                assert result == temp_path
        finally:
            temp_path.unlink(missing_ok=True)

    def test_optimize_image_size_very_small_image(self):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as f:
            img = Image.new("RGB", (100, 100), color="red")
            img.save(f.name, format="JPEG", quality=95)
            temp_path = Path(f.name)

        try:
            size_mb = _get_file_size_mb(temp_path)
            assert size_mb < 1.0

            result_path = optimize_image_size(temp_path, max_size_mb=7.0)
            assert result_path == temp_path
        finally:
            temp_path.unlink(missing_ok=True)

    def test_optimize_image_size_grayscale_image(self):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as f:
            img = Image.new("L", (3000, 3000), color=128)
            img.save(f.name, format="PNG", compress_level=0)
            temp_path = Path(f.name)

        try:
            original_size_mb = _get_file_size_mb(temp_path)

            if original_size_mb > 7.0:
                result_path = optimize_image_size(temp_path, max_size_mb=7.0)

                assert result_path != temp_path
                assert result_path.suffix == ".jpg"

                with Image.open(result_path) as optimized_img:
                    assert optimized_img.mode == "RGB"

                optimized_size_mb = _get_file_size_mb(result_path)
                assert optimized_size_mb < 7.0

                result_path.unlink()
        finally:
            temp_path.unlink(missing_ok=True)

    def test_optimize_image_size_png_with_alpha(self):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as f:
            img = Image.new("RGBA", (3000, 3000), color=(255, 100, 50, 200))
            img.save(f.name, format="PNG", compress_level=0)
            temp_path = Path(f.name)

        try:
            original_size_mb = _get_file_size_mb(temp_path)

            if original_size_mb > 7.0:
                result_path = optimize_image_size(temp_path, max_size_mb=7.0)

                assert result_path != temp_path
                assert result_path.suffix == ".jpg"

                with Image.open(result_path) as optimized_img:
                    assert optimized_img.mode == "RGB"

                optimized_size_mb = _get_file_size_mb(result_path)
                assert optimized_size_mb < 7.0

                result_path.unlink()
        finally:
            temp_path.unlink(missing_ok=True)

    @pytest.mark.filterwarnings("ignore::PIL.Image.DecompressionBombWarning")
    def test_optimize_image_size_extreme_case(self):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as f:
            img = Image.new("RGB", (10000, 10000), color="white")
            img.save(f.name, format="PNG", compress_level=0)
            temp_path = Path(f.name)

        try:
            original_size_mb = _get_file_size_mb(temp_path)
            assert original_size_mb > 7.0

            result_path = optimize_image_size(temp_path, max_size_mb=7.0)

            if result_path != temp_path:
                optimized_size_mb = _get_file_size_mb(result_path)
                assert optimized_size_mb < 7.0

                result_path.unlink()
        finally:
            temp_path.unlink(missing_ok=True)

    def test_optimize_image_temp_file_cleanup_on_success(self):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as f:
            img = Image.new("RGB", (5000, 5000), color="blue")
            img.save(f.name, format="PNG", compress_level=0)
            temp_path = Path(f.name)

        try:
            original_size_mb = _get_file_size_mb(temp_path)

            if original_size_mb > 7.0:
                result_path = optimize_image_size(temp_path, max_size_mb=7.0)

                assert result_path != temp_path
                assert result_path.exists()

                result_path.unlink()

                assert not result_path.exists()
        finally:
            temp_path.unlink(missing_ok=True)
