import numpy as np
import typing as t

from ..block import Block
from ..moves import ColorMove as ColorMove
from ..moves import Move
from ..types import RGBAImage, Box, Color
from ..box_utils import get_part, box_size
from ..scoring import image_similarity, block_similarity, static_cost


def produce_program(img: RGBAImage, blocks: t.Sequence[Block]) -> t.List[Move]:
    h, w = img.shape[:2]
    img_area = h * w

    def move_cost(box: Box, old_part: RGBAImage, new_color: Color) -> float:
        return static_cost(ColorMove, box_size(box), img_area) \
               + block_similarity(get_part(img, box), new_color) \
               - image_similarity(get_part(img, box), old_part)

    moves = []
    for cur_block in blocks:
        block_id = cur_block.block_id
        x_min, y_min, x_max, y_max = cur_block.box
        block_img = cur_block.img_part
        cur_box = (x_min, y_min, x_max, y_max)
        cur_color = np.mean(img[y_min:y_max, x_min:x_max], axis=(0, 1))
        if move_cost(cur_box, block_img, cur_color) < 0:
            moves.append(ColorMove(block_id, cur_color))
    return moves
