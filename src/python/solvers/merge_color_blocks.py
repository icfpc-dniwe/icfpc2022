import numpy as np
import typing as t

from ..block import Block
from ..moves import ColorMove, Merge, Move
from ..types import RGBAImage, Box, Color
from ..box_utils import get_part, box_size
from ..scoring import image_similarity, block_similarity, static_cost


def blocks_are_neightbors(left_block: Block, right_block: Block) -> bool:
    left_box = left_block.box
    right_box = right_block.box


def find_neighbors(blocks: t.Sequence[Block]) -> t.Dict[int, t.Tuple[int, int, int, int]]:
    neighbors = np.eye(len(blocks), dtype=bool)
    for left_idx, left_block in enumerate(blocks):
        for right_idx, right_block in enumerate(blocks):
            if left_idx == right_idx:
                continue
            pass


def produce_program(
        img: RGBAImage,
        blocks: t.Sequence[Block],
        default_canvas: t.Optional[RGBAImage] = None,
        global_block_id: t.Optional[int] = None
) -> t.List[Move]:
    pass
