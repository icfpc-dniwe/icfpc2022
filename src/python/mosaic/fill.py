import numpy as np
import cv2
from enum import Enum, auto
import typing as t

from ..moves import Color
from ..scoring import static_cost, block_similarity
from ..image_utils import get_area
from ..box_utils import box_size, get_part
from ..types import RGBAImage, LabelImage, Color, Box


class ExpandOrientation(Enum):
    Top = auto()
    Right = auto()
    Bottom = auto()
    Left = auto()

    def next(self) -> 'ExpandOrientation':
        if self is self.Top:
            return self.Right
        elif self is self.Right:
            return self.Bottom
        elif self is self.Bottom:
            return self.Left
        else:
            return self.Top


def expand(old_bbox: np.ndarray, orientation: ExpandOrientation) -> np.ndarray:
    new_bbox = old_bbox.copy()
    if orientation is ExpandOrientation.Top:
        new_bbox[3] += 1
    elif orientation is ExpandOrientation.Right:
        new_bbox[2] += 1
    elif orientation is ExpandOrientation.Bottom:
        new_bbox[1] -= 1
    elif orientation is ExpandOrientation.Left:
        new_bbox[0] -= 1
    return new_bbox


def expand_pixel(
        img: RGBAImage,
        point: t.Tuple[int, int],
        default_color: Color = (0, 0, 0, 0),
        min_block_area: int = 9
) -> t.Optional[t.Tuple[Box, Color]]:
    canvas_area = get_area(img)
    bbox = np.array([point[0], point[1], point[0] + 1, point[1] + 1], dtype=np.int64)
    cur_orientation = ExpandOrientation.Top

    def color_cost(block_area):
        return static_cost(Color, block_area, canvas_area)

    def can_expand(orientation: ExpandOrientation) -> bool:
        if orientation is ExpandOrientation.Top and bbox[3] >= img.shape[0]:
            return False
        elif orientation is ExpandOrientation.Right and bbox[2] >= img.shape[1]:
            return False
        elif orientation is ExpandOrientation.Bottom and bbox[1] <= 0:
            return False
        elif orientation is ExpandOrientation.Left and bbox[0] <= 0:
            return False
        return True

    def try_expand(new_orientation: ExpandOrientation) -> bool:
        if can_expand(new_orientation):
            new_bbox = expand(bbox, new_orientation)
            new_color = np.mean(img[new_bbox[1]:new_bbox[3], new_bbox[0]:new_bbox[2]], axis=(0, 1))
            return block_similarity(get_part(img, new_bbox), new_color)\
                   < block_similarity(get_part(img, new_bbox), default_color)
        return False

    while True:
        old_orientation = cur_orientation
        cur_orientation = cur_orientation.next()
        while (not try_expand(cur_orientation)) and (cur_orientation is not old_orientation):
            cur_orientation = cur_orientation.next()
        if cur_orientation is old_orientation:
            break
        bbox = expand(bbox, cur_orientation)
    if box_size(bbox) < min_block_area:
        return None
    else:
        color = np.mean(img[bbox[1]:bbox[3], bbox[0]:bbox[2]], axis=(0, 1))
        return bbox, color


def process_image(img: RGBAImage) -> t.Tuple[LabelImage, t.Dict[int, Color]]:
    pass
