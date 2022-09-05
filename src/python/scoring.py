import numpy as np
import typing as t

from .box_utils import box_size
from .types import RGBAImage, Color, Box
from .moves import Move, get_block_size


def relative_static_cost(move: t.Union[Move, t.Type[Move]], block_area: int) -> float:
    if block_area == 0:
        return 10000
    return move.cost() / block_area


def static_cost(move: t.Union[Move, t.Type[Move]], block_area: int, canvas_area: int) -> int:
    if block_area == 0:
        return 10000
    return round(move.cost() * canvas_area / block_area)


def block_similarity(img_part: RGBAImage, block_color: Color, alpha: float = 0.005) -> float:
    int_img = img_part.astype(np.int64)
    block_color = np.array(block_color, copy=False).astype(np.int64)
    return float(alpha * np.sum(np.sqrt(np.sum((int_img - block_color) ** 2, axis=-1))))


def image_similarity(left_img: RGBAImage, right_img: RGBAImage, alpha: float = 0.005) -> float:
    int_left = left_img.astype(np.int64)
    int_right = right_img.astype(np.int64)
    return float(alpha * np.sum(np.sqrt(np.sum((int_left - int_right) ** 2, axis=-1))))


def score_program_agaist_nothing(
        source_img: RGBAImage,
        canvas_before: RGBAImage,
        canvas_after: RGBAImage,
        program: t.List[Move],
        box: t.Optional[Box] = None,
        use_relative: bool = False
) -> t.Tuple[float, float]:
    do_nothing_cost = image_similarity(source_img, canvas_before)
    after_similarity = image_similarity(source_img, canvas_after)
    if use_relative:
        program_cost = sum(
            [relative_static_cost(cur_move, get_block_size(cur_move)) for cur_move in program]
        )
    else:
        h, w = source_img.shape[:2]
        img_area = h * w
        program_cost = sum(
            [static_cost(cur_move, get_block_size(cur_move), img_area) for cur_move in program]
        )
    return do_nothing_cost, after_similarity + program_cost
