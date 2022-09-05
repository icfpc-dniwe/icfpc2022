import numpy as np
from scipy.optimize import linear_sum_assignment
import typing as t

from ..block import Block, create_canvas
from ..moves import ColorMove, Merge, Move, Swap
from ..types import RGBAImage, Box, Color
from ..box_utils import get_part, box_size, can_merge
from ..image_utils import get_area
from ..scoring import image_similarity, block_similarity, static_cost, score_program_agaist_nothing, relative_static_cost


def get_swap_cost(img: RGBAImage, left_block: Block, right_block: Block) -> float:
    return static_cost(Swap, max(box_size(left_block.box), box_size(right_block.box)), get_area(img))\
           + image_similarity(get_part(img, left_block.box), right_block.img_part)\
           + image_similarity(get_part(img, right_block.box), left_block.img_part)\
           - image_similarity(get_part(img, left_block.box), left_block.img_part)\
           - image_similarity(get_part(img, right_block.box), right_block.img_part)


def produce_program(
        img: RGBAImage,
        blocks: t.Sequence[Block]
) -> t.Tuple[RGBAImage, t.List[Move]]:
    moves = []
    cur_canvas = create_canvas(blocks, *img.shape[:2])
    cost_matrix = np.zeros((len(blocks), len(blocks)), dtype=np.float32) + 1
    for left_idx in range(0, len(blocks)-1):
        for right_idx in range(left_idx + 1, len(blocks)):
            cost = get_swap_cost(img, blocks[left_idx], blocks[right_idx])
            cost_matrix[left_idx, right_idx] = cost
            cost_matrix[right_idx, left_idx] = cost

    row_assignments, col_assignments = linear_sum_assignment(cost_matrix)
    total_cost = cost_matrix[row_assignments, col_assignments].sum()
    print('total assignment cost:', total_cost)
    swapped = set()
    new_blocks = []
    for cur_row, cur_col in zip(row_assignments, col_assignments):
        if cur_row in swapped or cur_col in swapped:
            continue
        swapped.add(cur_row)
        swapped.add(cur_col)
        cur_cost = cost_matrix[cur_row, cur_col]
        if cur_cost < 0:
            moves.append(Swap(blocks[cur_row].block_id, blocks[cur_col].block_id,
                              max(box_size(blocks[cur_row].box), box_size(blocks[cur_col].box))))
            # left_box = blocks[cur_row].box
            # right_box = blocks[cur_col].box
            # cur_canvas[]
    # cur_canvas = create_canvas(new_blocks, *img.shape[:2])
    return cur_canvas, moves
