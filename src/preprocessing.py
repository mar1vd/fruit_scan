"""Image preprocessing shared by training/inference."""
from PIL import Image, ImageOps
import numpy as np

IMAGE_SIZE = (100, 100)


def prepare_image(image: Image.Image) -> np.ndarray:
    """Return a float32 RGB batch shaped (1, 100, 100, 3), values in [0, 255].

    The model includes a Rescaling layer, so do not divide by 255 here.
    """
    if not isinstance(image, Image.Image):
        raise TypeError("image must be a PIL.Image.Image")
    image = ImageOps.exif_transpose(image).convert("RGB")
    image = image.resize(IMAGE_SIZE)
    array = np.asarray(image, dtype=np.float32)
    return np.expand_dims(array, axis=0)
