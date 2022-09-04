import numpy as np
import cv2
from enum import Enum, auto
import typing as t

from ..moves import ColorMove
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

    @staticmethod
    def get_all() -> t.Sequence['ExpandOrientation']:
        return [
            ExpandOrientation.Top,
            ExpandOrientation.Right,
            ExpandOrientation.Bottom,
            ExpandOrientation.Left
        ]


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
        point: t.Tuple[int, int],
        canvas_height: int,
        canvas_width: int,
        expand_cost_fn: t.Callable[[Box], t.Optional[float]],
        tol: float = 1,
        min_block_area: int = 9
) -> t.Optional[Box]:
    bbox = np.array([point[0], point[1], point[0] + 1, point[1] + 1], dtype=np.int64)

    def can_expand(orientation: ExpandOrientation) -> bool:
        if orientation is ExpandOrientation.Top and bbox[3] >= canvas_height:
            return False
        elif orientation is ExpandOrientation.Right and bbox[2] >= canvas_width:
            return False
        elif orientation is ExpandOrientation.Bottom and bbox[1] <= 0:
            return False
        elif orientation is ExpandOrientation.Left and bbox[0] <= 0:
            return False
        return True

    def try_expand(new_orientation: ExpandOrientation) -> t.Optional[float]:
        if can_expand(new_orientation):
            new_bbox = expand(bbox, new_orientation)
            return expand_cost_fn(new_bbox)
        return None

    all_orientations = ExpandOrientation.get_all()
    cur_cost = None
    while True:
        costs = list(filter(lambda x: x[1] is not None, enumerate(map(try_expand, all_orientations))))
        if len(costs) < 1:
            break
        cost_idx = np.argmin([el[1] for el in costs])
        if cur_cost is None:
            cur_cost = costs[cost_idx][1]
        else:
            if costs[cost_idx][1] < cur_cost:
                cur_cost = costs[cost_idx][1]
            else:
                break
        cur_orientation = all_orientations[costs[cost_idx][0]]
        bbox = expand(bbox, cur_orientation)
    if box_size(bbox) < min_block_area:
        return None
    else:
        return bbox
