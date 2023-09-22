import math
from typing import Union

import numpy as np
from PIL import Image


def resize(
    w: int, h: int, expected_height: int, image_min_width: int, image_max_width: int
):
    """Resize image to expected height while maintaining aspect ratio

    Args:
        w (int): Width of image
        h (int): Height of image
        expected_height (int): Expected height of image
        image_min_width (int): Minimum width of image
        image_max_width (int): Maximum width of image

    Returns:
            tuple: (new_w, expected_height)
    """
    new_w = int(expected_height * float(w) / float(h))
    round_to = 10
    new_w = math.ceil(new_w / round_to) * round_to
    new_w = max(new_w, image_min_width)
    new_w = min(new_w, image_max_width)

    return new_w, expected_height


def process_image(
    image: Image.Image, image_height: int, image_min_width: int, image_max_width: int
):
    """Process image to feed into model

    Args:
        image (PIL.Image.Image): Image
        image_height (int): Expected height of image
        image_min_width (int): Minimum width of image
        image_max_width (int): Maximum width of image

    Returns:
        np.ndarray: Processed image
    """
    img = image.convert("RGB")

    w, h = img.size
    new_w, image_height = resize(w, h, image_height, image_min_width, image_max_width)

    img = img.resize((new_w, image_height), Image.LANCZOS)

    img = np.asarray(img).transpose(2, 0, 1)
    img = img / 255
    return img


def process_input(
    image: Union[Image.Image, np.ndarray, str],
    image_height: int,
    image_min_width: int,
    image_max_width: int,
) -> np.ndarray:
    """Process input to feed into model

    Args:
        image (np.ndarray or PIL.Image.Image or str): Image
        image_height (int): Expected height of image
        image_min_width (int): Minimum width of image
        image_max_width (int): Maximum width of image

    Returns:
        np.ndarray: Processed image
    """
    img = process_image(image, image_height, image_min_width, image_max_width)
    img = img[np.newaxis, ...]
    img = img.astype(np.float32)
    return img
