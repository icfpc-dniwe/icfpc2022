import numpy as np
import typing as t

from .types import RGBAImage, Color
from .moves import Move


def static_cost(move: Move, block_area: int, canvas_area: int) -> int:
    return round(move.cost() * canvas_area / block_area)


def block_similarity(img_part: RGBAImage, block_color: Color, alpha: float = 0.005) -> float:
    return float(alpha * np.sum(np.sqrt(np.sum((img_part - block_color) ** 2, axis=-1))))


def image_similarity(left_img: RGBAImage, right_img: RGBAImage, alpha: float = 0.005) -> float:
    return float(alpha * np.sum(np.sqrt(np.sum((left_img - right_img) ** 2, axis=-1))))
