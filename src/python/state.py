from copy import copy
from typing import Tuple, List, Optional, Union

import numpy as np

from python.types import Box, RGBAImage


class BoxMk2:
    def __init__(self, box: t.Union[Box, Tuple[int, int, int, int]]):
        self.x_min, self.y_min, self.x_max, self.y_max = box

    @property
    def size(self) -> int:
        return (self.x_max - self.x_min) * (self.y_max - self.y_min)

    @property
    def wh(self) -> Tuple[int, int]:
        return self.x_max - self.x_min, self.y_max - self.y_min

    def image_part(self, img: RGBAImage) -> RGBAImage:
        return img[self.y_min:self.y_max, self.x_min:self.x_max]

    @property
    def box(self) -> Box:
        return np.ndarray([self.x_min, self.y_min, self.x_max, self.y_max])


class Block:
    def __init__(self, block_id: Union[str, int], box: BoxMk2):
        self.__block_id = str(block_id)
        self.__box = box

    @property
    def block_id(self) -> str:
        return self.__block_id


    @property
    def box(self) -> BoxMk2:
        return self.__box


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
            initial_box = BoxMk2([0, 0, self.__target_image.shape[1], self.__target_image.shape[0]])
            self.__blocks = [Block(self.__global_block_id, initial_box)]
            self.__global_block_id += 1
        else:
            self.__blocks = blocks

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

    def similarity(self) -> int:
        pass # TODO implement

    # def lcut(self):
    #     pass # TODO implement
    #
    # def pcut(self):
    #     pass # TODO implement
    #
    # def merge(self):
    #     pass # TODO implement
    #
    # def swap(self):
    #     pass # TODO implement
    #
    # def color(self):
    #     pass # TODO implement
