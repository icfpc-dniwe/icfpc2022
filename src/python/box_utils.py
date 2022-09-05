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


def can_merge(left_box: Box, right_box: Box) -> bool:
    left_x_min, left_y_min, left_x_max, left_y_max = left_box
    right_x_min, right_y_min, right_x_max, right_y_max = right_box
    if left_x_min != right_x_min:
        if left_y_min != right_y_min or left_y_max != right_y_max:
            return False
        if left_x_min < right_x_min:
            return left_x_max == right_x_min
        else:
            return right_x_max == left_x_min
    else:
        if left_x_max != right_x_max:
            return False
        if left_y_min < right_y_min:
            return left_y_max == right_y_min
        else:
            return right_y_max == left_y_min
