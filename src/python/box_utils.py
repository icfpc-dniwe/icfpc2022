import numpy as np
import typing as t
from .types import Box, RGBAImage


def box_size(box: Box) -> int:
    x_min, y_min, x_max, y_max = box
    return (x_max - x_min) * (y_max - y_min)


def box_wh(box: Box) -> t.Tuple[int, int]:
    x_min, y_min, x_max, y_max = box
    return x_max - x_min, y_max - y_min


def get_part(img: RGBAImage, box: Box) -> RGBAImage:
    return img[box[1]:box[3], box[0]:box[2]]
