import numpy as np
import typing as t

from .types import Box, RGBAImage
from .image_utils import revert_y


class Block:
    def __init__(self, block_id: str, box: Box, img_part: RGBAImage):
        self.block_id = block_id
        self.box = box
        self.img_part = img_part


def from_json(json_block: t.Dict[str, t.Any], canvas: t.Optional[RGBAImage] = None) -> Block:
    x_min, y_min = json_block['bottomLeft']
    x_max, y_max = json_block['topRight']
    block_w = x_max - x_min
    block_h = y_max - y_min
    if 'pngBottomLeftPoint' in json_block:
        if canvas is None:
            raise ValueError(f'Found a block with a "pngBottomLeftPoint" but no canvas given')
        img_x_min, img_y_min = json_block['pngBottomLeftPoint']
        img_part = canvas[img_y_min:img_y_min+block_h, img_x_min:img_x_min+block_w]
    else:
        color = json_block['color']
        img_part = np.zeros((block_h, block_w, 4), dtype=np.uint8) + color
    return Block(json_block['blockId'], (x_min, y_min, x_max, y_max), img_part)


def create_canvas(blocks: t.Iterable[Block], height: int, width: int, reverse: bool = False) -> RGBAImage:
    canvas = np.zeros((height, width, 4), dtype=np.uint8)
    for cur_block in blocks:
        box = cur_block.box
        canvas[box[1]:box[3], box[0]:box[2]] = cur_block.img_part
    if reverse:
        canvas = revert_y(canvas)
    return canvas
