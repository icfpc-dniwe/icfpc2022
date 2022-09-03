import numpy as np
import typing as t
from .types import Orientation


def add_line_cut(block_id: str, orientation: Orientation, offset: int) -> str:
    return f'cut [{block_id}] {orientation} [{offset}]'


def add_point_cut_move(block_id: str, offset_point: t.Union[np.ndarray, t.Tuple[int, int]]) -> str:
    return f'cut [{block_id}] [{offset_point[0]},{offset_point[1]}]'


def add_color_move(block_id: str, color: t.Union[np.ndarray, t.Tuple[int, int, int, int]]) -> str:
    return f'color [{block_id}] [{color[0]},{color[1]},{color[2]},{color[3]}]'


def add_merge_move(left_block_id: str, right_block_id: str) -> str:
    return f'merge [{left_block_id}] [{right_block_id}]'
