import numpy as np
import typing as t

from ..moves import Color as ColorMove
from ..moves import Move
from ..types import RGBAImage


def produce_program(img: RGBAImage, block_info: t.Sequence[t.Any]) -> t.List[Move]:
    moves = []
    for cur_block in block_info:
        block_id = cur_block['blockId']
        x_min, y_min = cur_block['bottomLeft']
        x_max, y_max = cur_block['topRight']
        cur_color = np.mean(img[y_min:y_max, x_min:x_max], axis=(0, 1))
        moves.append(ColorMove(block_id, cur_color))
    return moves
