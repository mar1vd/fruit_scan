from PIL import Image
import numpy as np
import pytest

from src.preprocessing import prepare_image


def test_prepare_image_returns_rgb_batch():
    image = Image.new("RGB", (250, 120), color=(255, 0, 0))
    result = prepare_image(image)
    assert result.shape == (1, 100, 100, 3)
    assert result.dtype == np.float32
    assert result.min() >= 0
    assert result.max() <= 255


def test_prepare_image_converts_grayscale_to_rgb():
    image = Image.new("L", (80, 80), color=128)
    result = prepare_image(image)
    assert result.shape == (1, 100, 100, 3)


def test_prepare_image_rejects_non_pil_input():
    with pytest.raises(TypeError):
        prepare_image("not an image")
