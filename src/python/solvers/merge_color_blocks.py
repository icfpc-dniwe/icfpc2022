import numpy as np
import typing as t

from ..block import Block
from ..moves import ColorMove, Merge, Move
from ..types import RGBAImage, Box, Color
from ..box_utils import get_part, box_size, can_merge
from ..scoring import image_similarity, block_similarity, static_cost


def blocks_are_neightbors(left_block: Block, right_block: Block) -> bool:
    left_box = left_block.box
    right_box = right_block.box
    return can_merge(left_box, right_box)


def find_neighbors(blocks: t.Sequence[Block]) -> t.Dict[int, t.Tuple[int, int, int, int]]:
    """
    :param blocks: sequence of Block
    :return: a dictionary block_idx -> (top_neighbor_idx, right_neighbor_idx, bottom_neighbor_idx, left_neighbor_idx)
    """
    neighbors_dict = {cur_idx: [-1, -1, -1, -1] for cur_idx in range(len(blocks))}
    # 1 -- top; 2 -- right; 3 -- bottom; 4 -- left
    for left_idx, left_block in enumerate(blocks):
        for right_idx, right_block in enumerate(blocks):
            if left_idx == right_idx:
                continue
            if blocks_are_neightbors(left_block, right_block):
                left_x_min, left_y_min, left_x_max, left_y_max = left_block.box
                right_x_min, right_y_min, right_x_max, right_y_max = right_block.box
                if left_x_min == right_x_min:
                    if left_y_min < right_y_min:
                        neighbors_dict[left_idx][2] = right_idx
                        neighbors_dict[right_idx][0] = left_idx
                    else:
                        neighbors_dict[left_idx][0] = right_idx
                        neighbors_dict[right_idx][2] = left_idx
                else:
                    if left_x_min < right_x_min:
                        neighbors_dict[left_idx][1] = right_idx
                        neighbors_dict[right_idx][3] = left_idx
                    else:
                        neighbors_dict[left_idx][3] = right_idx
                        neighbors_dict[right_idx][1] = left_idx
    return neighbors_dict


def produce_program(
        img: RGBAImage,
        blocks: t.Sequence[Block],
        default_canvas: t.Optional[RGBAImage] = None,
        global_block_id: t.Optional[int] = None
) -> t.List[Move]:
    pass
