import numpy as np
import typing as t

from .types import RGBAImage, Color
from .moves import Move


def static_cost(move: t.Union[Move, t.Type[Move]], block_area: int, canvas_area: int) -> int:
    return round(move.cost() * canvas_area / block_area)


def block_similarity(img_part: RGBAImage, block_color: Color, alpha: float = 0.005) -> float:
    int_img = img_part.astype(np.int64)
    block_color = np.array(block_color, copy=False).astype(np.int64)
    return float(alpha * np.sum(np.sqrt(np.sum((int_img - block_color) ** 2, axis=-1))))


def image_similarity(left_img: RGBAImage, right_img: RGBAImage, alpha: float = 0.005) -> float:
    int_left = left_img.astype(np.int64)
    int_right = right_img.astype(np.int64)
    return float(alpha * np.sum(np.sqrt(np.sum((int_left - int_right) ** 2, axis=-1))))
