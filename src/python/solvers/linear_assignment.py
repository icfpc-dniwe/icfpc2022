import numpy as np
from scipy.optimize import linear_sum_assignment
import typing as t

from ..block import Block, create_canvas
from ..moves import ColorMove, Merge, Move, Swap
from ..types import RGBAImage, Box, Color
from ..box_utils import get_part, box_size, can_merge
from ..scoring import image_similarity, block_similarity, static_cost, score_program_agaist_nothing, relative_static_cost


def get_swap_cost(img: RGBAImage, canvas: RGBAImage, left_block: Block, right_block: Block) -> float:
    return relative_static_cost(Swap, max(box_size(left_block.box), box_size(right_block.box)))\
           + image_similarity(get_part(img, left_block.box), get_part(canvas, right_block.box))\
           + image_similarity(get_part(img, right_block.box), get_part(canvas, left_block.box))\
           - image_similarity(get_part(img, left_block.box), get_part(canvas, left_block.box))\
           - image_similarity(get_part(img, right_block.box), get_part(canvas, right_block.box))


def produce_program(
        img: RGBAImage,
        blocks: t.Sequence[Block]
) -> t.Tuple[RGBAImage, t.List[Move]]:
    default_canvas = create_canvas(blocks, *img.shape[:2])
    cur_canvas = default_canvas.copy()
    moves = []

    cost_matrix = np.eye(len(blocks), dtype=np.float32) * 100000
    for left_idx in range(0, len(blocks)-1):
        for right_idx in range(left_idx + 1, len(blocks)):
            cost = get_swap_cost(img, default_canvas, blocks[left_idx], blocks[right_idx])
            cost_matrix[left_idx, right_idx] = cost
            cost_matrix[right_idx, left_idx] = cost

    row_assignments, col_assignments = linear_sum_assignment(cost_matrix)
    for cur_row, cur_col in zip(row_assignments, col_assignments):
        moves.append(Swap(blocks[cur_row].block_id, blocks[cur_col].block_id,
                          max(box_size(blocks[cur_row].box), box_size(blocks[cur_col].box))))
    return cur_canvas, moves
