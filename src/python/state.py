from copy import copy
from typing import Tuple, List, Optional, Union

import numpy as np

from python.scoring import image_similarity
from python.types import Box, RGBAImage, Color, BlockId


class BoxMk2:
    def __init__(self, box: Union[Box, Tuple[int, int, int, int]]):
        self.__x_min, self.__y_min, self.__x_max, self.__y_max = box

        if self.__x_min >= self.__x_max or self.__y_min >= self.__y_max:
            raise Exception("BoxMk2 creation error, cause coordinates")

    @property
    def size(self) -> int:
        return (self.__x_max - self.__x_min) * (self.__y_max - self.__y_min)

    @property
    def wh(self) -> Tuple[int, int]:
        return self.__x_max - self.__x_min, self.__y_max - self.__y_min

    @property
    def box(self) -> Box:
        return np.ndarray([self.__x_min, self.__y_min, self.__x_max, self.__y_max])

    def is_in(self, x: int, y: int) -> bool:
        return self.__x_min <= x < self.__x_max and self.__y_min <= y < self.__y_max

    def image_part(self, img: RGBAImage) -> RGBAImage:
        return img[self.__y_min:self.__y_max, self.__x_min:self.__x_max]

    def average_color(self, img: RGBAImage) -> Color:
        return np.mean(self.image_part(img), axis=(0, 1))

    def set_color(self, img: RGBAImage, color: Color) -> None:
        img[self.__y_min:self.__y_max, self.__x_min:self.__x_max] = color

    def lcut_vertical(self, x: int) -> Tuple['BoxMk2', 'BoxMk2']:
        left_box = BoxMk2((self.__x_min, self.__y_min, x, self.__x_max))
        right_box = BoxMk2((x, self.__y_min, self.__x_max, self.__y_max))
        return left_box, right_box

    def lcut_horizontal(self, y: int) -> Tuple['BoxMk2', 'BoxMk2']:
        bottom_box = BoxMk2((self.__x_min, self.__y_min, self.__x_max, y))
        top_box = BoxMk2((self.__x_min, y, self.__x_max, self.__y_max))
        return bottom_box, top_box

    def pcut(self, x: int, y: int) -> Tuple['BoxMk2', 'BoxMk2', 'BoxMk2', 'BoxMk2']:
        bottom_left_box = BoxMk2((self.__x_min, self.__y_min, x, y))
        bottom_right_box = BoxMk2((x, self.__y_min, self.__x_max, y))
        top_right_box = BoxMk2((x, y, self.__x_max, self.__y_max))
        top_left_box = BoxMk2((self.__x_min, y, x, self.__x_max))
        return bottom_left_box, bottom_right_box, top_right_box, top_left_box

    def merge(self, box_mk2): # -> BoxMk2:
        if not self.mergable():
            raise Exception("BoxMk2s are not mergeble!")
        x_min_1, y_min_1, x_max_1, y_max_1 = self.box
        x_min_2, y_min_2, x_max_2, y_max_2 = box_mk2.box
        merged_box = BoxMk2((
            min(x_min_1, x_min_2),
            min(y_min_1, y_min_2),
            max(x_max_1, x_max_2),
            max(y_max_1, y_max_2),
        ))
        return merged_box

    def borders(self, box_mk2) -> bool:
        pass

    def mergable(self, box_mk2) -> bool:
        x_min_1, y_min_1, x_max_1, y_max_1 = self.box
        x_min_2, y_min_2, x_max_2, y_max_2 = box_mk2.box
        if (x_min_1, y_min_1, x_max_1, y_min_1) == (x_min_2, y_max_2, x_max_2, y_max_2):
            return True
        if (x_min_2, y_min_2, x_max_2, y_min_2) == (x_min_1, y_max_1, x_max_1, y_max_1):
            return True
        if (x_min_1, y_min_1, x_min_1, y_max_1) == (x_max_2, y_min_2, x_max_2, y_max_2):
            return True
        if (x_min_2, y_min_2, x_min_2, y_max_2) == (x_max_1, y_min_1, x_max_1, y_max_1):
            return True
        return False


class Block:
    def __init__(self, block_id: Union[BlockId, int], box: BoxMk2):
        self.__block_id = str(block_id)
        self.__box = box

    @property
    def block_id(self) -> BlockId:
        return self.__block_id

    @property
    def box(self) -> BoxMk2:
        return self.__box

    def is_in(self, x: int, y: int) -> bool:
        return self.__box.is_in(x, y)

    def image_part(self, img: RGBAImage) -> RGBAImage:
        return self.__box.image_part(img)

    def average_color(self, img: RGBAImage) -> Color:
        return self.__box.average_color(img)

    def set_color(self, img: RGBAImage, color: Color) -> None:
        return self.__box.set_color(img, color)

    def lcut_vertical(self, x: int) -> Tuple['Block', 'Block']:
        left_box, right_box = self.__box.lcut_vertical(x)
        left_block = Block(f'{self.__block_id}.0', left_box)
        right_block = Block(f'{self.__block_id}.1', right_box)
        return left_block, right_block

    def lcut_horizontal(self, y: int) -> Tuple['Block', 'Block']:
        bottom_box, top_box = self.__box.lcut_horizontal(y)
        bottom_block = Block(f'{self.__block_id}.0', bottom_box)
        top_block = Block(f'{self.__block_id}.1', top_box)
        return bottom_block, top_block

    def pcut(self, x: int, y: int) -> Tuple['Block', 'Block', 'Block', 'Block']:
        bottom_left_box, bottom_right_box, top_right_box, top_left_box = self.__box.pcut(x, y)
        bottom_left_block = Block(f'{self.__block_id}.0', bottom_left_box)
        bottom_right_block = Block(f'{self.__block_id}.1', bottom_right_box)
        top_right_block = Block(f'{self.__block_id}.2', top_right_box)
        top_left_block = Block(f'{self.__block_id}.3', top_left_box)
        return bottom_left_block, bottom_right_block, top_right_block, top_left_block

    def merge(self, block, global_block_id: str) -> 'Block'':
        merged_box = self.__box.merge(block.box)
        merged_block = Block(f'{global_block_id}', merged_box)
        return merged_block


def parse_blocks(text: str) -> List[Block]:
    pass # TODO implement


class State:
    def __init__(self,
                 target_image: RGBAImage,
                 initial_image: Optional[RGBAImage] = None,
                 blocks: Optional[List[Block]] = None,
                 global_block_id: Optional[int] = 0,
                 cur_image: Optional[RGBAImage] = None, # only for self.copy()
    ):
        self.__target_image = target_image.copy()
        self.__global_block_id = global_block_id

        if self.__initial_image is None:
            self.__initial_image = np.full_like(target_image, 255)
        else:
            self.__initial_image = initial_image.copy()

        if cur_image is None:
            self.__cur_image = cur_image
        else:
            self.__cur_image = self.__initial_image.copy()

        if self.__blocks is None:
            w, h = self.wh
            initial_box = BoxMk2((0, 0, w, h))
            self.__blocks = [Block(self.__global_block_id, initial_box)]
            self.__global_block_id += 1
        else:
            self.__blocks = blocks

    @property
    def wh(self) -> Tuple[int, int]:
        return self.__target_image.shape[1], self.__target_image.shape[0]

    def copy(self):
        return State(
            target_image=self.__target_image,
            initial_image=self.__initial_image,
            blocks=copy(self.__blocks),
            global_block_id=self.__global_block_id,
            cur_image=self.__cur_image.copy()
        )

    def target_image(self) -> RGBAImage:
        return self.__target_image.copy()

    def initial_image(self) -> RGBAImage:
        return self.__initial_image.copy()

    def cur_image(self) -> RGBAImage:
        return self.__cur_image.copy()

    def blocks(self) -> List[Block]:
        return copy(self.__blocks)

    def global_block_id(self) -> str:
        return self.__global_block_id

    def block_at(self, x: int, y: int) -> Block:
        for block in self.__blocks:
            if block.is_in(x, y):
                return block
        raise Exception("Blocks are corrupt!!!")

    def block_by_id(self, block_id: BlockId) -> Block:
        for block in self.__blocks:
            if block.block_id == block_id:
                return block
        raise Exception("No such block_id")

    def similarity(self) -> float:
        return image_similarity(self.__cur_image, self.__target_image)

    def lcut_veritical(self, x: int, y: int) -> Tuple[BlockId, BlockId]:
        self.validate_cut(x, y)
        block_to_cut = self.block_at(x, y)
        left_block, right_block = block_to_cut.lcut_vertical(x)
        self.__blocks.remove(block_to_cut)
        self.__blocks.append(left_block)
        self.__blocks.append(right_block)
        return left_block.block_id, right_block.block_id

    def lcut_horizontal(self, x: int, y: int) -> Tuple[BlockId, BlockId]:
        self.validate_cut(x, y)
        block_to_cut = self.block_at(x, y)
        bottom_block, top_block = block_to_cut.lcut_horizontal(y)
        self.__blocks.remove(block_to_cut)
        self.__blocks.append(bottom_block)
        self.__blocks.append(top_block)
        return bottom_block.block_id, top_block.block_id

    def pcut(self, x: int, y: int) -> Tuple[BlockId, BlockId, BlockId, BlockId]:
        self.validate_cut(x, y)
        block_to_cut = self.block_at(x, y)
        bottom_left_block, bottom_right_block, top_right_block, top_left_block = block_to_cut.pcut(x, y)
        self.__blocks.remove(block_to_cut)
        self.__blocks.append(bottom_left_block)
        self.__blocks.append(bottom_right_block)
        self.__blocks.append(top_right_block)
        self.__blocks.append(top_left_block)
        return bottom_left_block.block_id, bottom_right_block.block_id, top_right_block.block_id, top_left_block.block_id

    def merge(self, block_id_1: BlockId, block_id_2: BlockId) -> BlockId:
        block_1 = self.block_by_id(block_id_1)
        block_2 = self.block_by_id(block_id_2)
        merged_block = block_1.merge(block_2, str(self.__global_block_id))
        self.__global_block_id += 1
        self.__blocks.remove(block_1)
        self.__blocks.remove(block_2)
        self.__blocks.append(merged_block)
        return merged_block.block_id

    def swap(self, block_id_1: BlockId, block_id_2: BlockId):
        pass

    def color(self, block_id: BlockId, color: Optional[Color]=None):
        block = self.block_by_id(block_id)
        if color is None:
            color = block.average_color(self.__target_image)
        block.set_color(self.__cur_image, color)


    def validate_cut(self, x: int, y: int) -> None:
        w, h = self.wh
        assert isinstance(x, int)
        assert isinstance(y, int)
        assert 0 < x < w - 1 and 0 < y < h - 1
