from copy import copy
from pathlib import Path
from typing import Tuple, List, Optional, Union

import cv2
import numpy as np

from python.image_utils import load_image
from python.moves import Move, LineCut, PointCut, Merge, ColorMove
from python.scoring import image_similarity
from python.types import Box, RGBAImage, Color, BlockId, Orientation


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
        return [self.__x_min, self.__y_min, self.__x_max, self.__y_max]

    def is_in(self, x: int, y: int) -> bool:
        return self.__x_min <= x < self.__x_max and self.__y_min <= y < self.__y_max

    def image_part(self, img: RGBAImage) -> RGBAImage:
        return img[self.__y_min:self.__y_max, self.__x_min:self.__x_max]

    def average_color(self, img: RGBAImage) -> Color:
        return np.mean(self.image_part(img), axis=(0, 1))

    def set_color(self, img: RGBAImage, color: Color) -> None:
        img[self.__y_min:self.__y_max, self.__x_min:self.__x_max] = color

    def lcut_vertical(self, x: int) -> Tuple['BoxMk2', 'BoxMk2']:
        left_box = BoxMk2((self.__x_min, self.__y_min, x, self.__y_max))
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
        top_left_box = BoxMk2((self.__x_min, y, x, self.__y_max))
        return bottom_left_box, bottom_right_box, top_right_box, top_left_box

    def merge(self, box_mk2: 'BoxMk2') -> 'BoxMk2':
        if not self.mergable(box_mk2):
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

    def borders(self, box_mk2: 'BoxMk2') -> bool:
        pass

    def mergable(self, box_mk2: 'BoxMk2') -> bool:
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

    @property
    def size(self) -> int:
        return self.__box.size

    @property
    def wh(self) -> Tuple[int, int]:
        return self.__box.wh

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

    def merge(self, block: 'Block', global_block_id: str) -> 'Block':
        merged_box = self.__box.merge(block.box)
        merged_block = Block(f'{global_block_id}', merged_box)
        return merged_block

    def mergable(self, block: 'Block') -> bool:
        return self.__box.mergable(block.box)


def parse_blocks(text: str) -> List[Block]:
    raise NotImplementedError


def get_biggest_block(blocks: List[Block]) -> Optional[Block]:
    max_size = -1
    cur_block = None
    for block in blocks:
        if block.size > max_size:
            max_size = block.size
            cur_block = block
    return cur_block


def get_smallest_block(blocks: List[Block]) -> Optional[Block]:
    min_size = 99999999
    cur_block = None
    for block in blocks:
        if block.size < min_size:
            min_size = block.size
            cur_block = block
    return cur_block


class State:
    def __init__(self,
                 target_image: RGBAImage,
                 initial_image: Optional[RGBAImage] = None,
                 blocks: Optional[List[Block]] = None,
                 global_block_id: Optional[int] = 0,
                 moves: Optional[List[Move]] = None,
                 cur_image: Optional[RGBAImage] = None, # only for self.copy()
    ):
        self.__target_image = target_image.copy()
        self.__global_block_id = global_block_id

        if initial_image is None:
            # self.__initial_image = np.full_like(self.__target_image, 255)
            self.__initial_image = np.zeros_like(self.__target_image) + 255
        else:
            self.__initial_image = initial_image.copy()

        if cur_image is None:
            self.__cur_image = self.__initial_image.copy()
        else:
            self.__cur_image = cur_image

        if blocks is None:
            w, h = self.wh
            initial_box = BoxMk2((0, 0, w, h))
            self.__blocks = [Block(self.__global_block_id, initial_box)]
            self.__global_block_id += 1
        else:
            self.__blocks = blocks

        if moves is None:
            self.__moves = []
        else:
            self.__moves = moves

    @property
    def wh(self) -> Tuple[int, int]:
        return self.__target_image.shape[1], self.__target_image.shape[0]

    def copy(self):
        return State(
            target_image=self.__target_image,
            initial_image=self.__initial_image,
            blocks=copy(self.__blocks),
            global_block_id=self.__global_block_id,
            moves=self.__moves,
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

    def block_ids(self) -> List[BlockId]:
        return [b.block_id for b in self.__blocks]

    def global_block_id(self) -> str:
        return self.__global_block_id

    def moves(self) -> List[Move]:
        return copy(self.__moves)

    def block_at(self, x: int, y: int) -> Block:
        for block in self.__blocks:
            if block.is_in(x, y):
                return block
        raise Exception("Blocks are corrupt!!!")

    def block_by_id(self, block_id: BlockId) -> Block:
        for block in self.__blocks:
            if block.block_id == block_id:
                return block
        raise Exception(f"No such block_id {block_id}")

    def similarity(self) -> float:
        return image_similarity(self.__cur_image, self.__target_image)

    def moves_cost(self) -> float:
        raise NotImplementedError

    def total_cost(self) -> float:
        return self.similarity() + self.moves_cost()

    def lcut_veritical(self, x: int, y: int) -> Tuple[Move, Tuple[BlockId, BlockId]]:
        self.validate_cut(x, y)
        block_to_cut = self.block_at(x, y)
        left_block, right_block = block_to_cut.lcut_vertical(x)
        self.__blocks.remove(block_to_cut)
        self.__blocks.append(left_block)
        self.__blocks.append(right_block)
        move = LineCut(block_to_cut.block_id, Orientation.X, x)
        self.__moves.append(move)
        return move, (left_block.block_id, right_block.block_id)

    def lcut_horizontal(self, x: int, y: int) -> Tuple[Move, Tuple[BlockId, BlockId]]:
        self.validate_cut(x, y)
        block_to_cut = self.block_at(x, y)
        bottom_block, top_block = block_to_cut.lcut_horizontal(y)
        self.__blocks.remove(block_to_cut)
        self.__blocks.append(bottom_block)
        self.__blocks.append(top_block)
        move = LineCut(block_to_cut.block_id, Orientation.Y, y)
        self.__moves.append(move)
        return move, (bottom_block.block_id, top_block.block_id)

    def pcut(self, x: int, y: int) -> Tuple[Move, Tuple[BlockId, BlockId, BlockId, BlockId]]:
        self.validate_cut(x, y)
        block_to_cut = self.block_at(x, y)
        bottom_left_block, bottom_right_block, top_right_block, top_left_block = block_to_cut.pcut(x, y)
        self.__blocks.remove(block_to_cut)
        self.__blocks.append(bottom_left_block)
        self.__blocks.append(bottom_right_block)
        self.__blocks.append(top_right_block)
        self.__blocks.append(top_left_block)
        move = PointCut(block_to_cut.block_id, (x, y))
        self.__moves.append(move)
        return move, (bottom_left_block.block_id, bottom_right_block.block_id, top_right_block.block_id, top_left_block.block_id)

    def merge(self, block_id_1: BlockId, block_id_2: BlockId) -> Tuple[Move, Tuple[BlockId]]:
        block_1 = self.block_by_id(block_id_1)
        block_2 = self.block_by_id(block_id_2)
        merged_block = block_1.merge(block_2, str(self.__global_block_id))
        self.__global_block_id += 1
        self.__blocks.remove(block_1)
        self.__blocks.remove(block_2)
        self.__blocks.append(merged_block)
        move = Merge(block_id_1, block_id_2)
        self.__moves.append(move)
        return move, (merged_block.block_id,)

    def swap(self, block_id_1: BlockId, block_id_2: BlockId):
        raise NotImplementedError

    def color(self, block_id: BlockId, color: Optional[Color]=None) -> Tuple[Move, Tuple]:
        block = self.block_by_id(block_id)
        if color is None:
            color = block.average_color(self.__target_image)
        block.set_color(self.__cur_image, color)
        move = ColorMove(block_id, color)
        self.__moves.append(move)
        return move, ()

    def validate_cut(self, x: int, y: int) -> None:
        w, h = self.wh
        assert isinstance(x, int)
        assert isinstance(y, int)
        assert 0 < x < w - 1 and 0 < y < h - 1

    def color_rect_and_remerge(self, box: Union[Box, BoxMk2], color: Optional[Color] = None, no_merge: bool = False) -> Tuple[List[Move], Tuple[BlockId]]:
        if not isinstance(box, BoxMk2):
            box = BoxMk2(box)

        x_min, y_min, x_max, y_max = box.box
        moves = []
        last_bid = None
        block_1 = self.block_at(x_min, y_min)
        block_2 = self.block_at(x_max-1, y_max-1)
        # w, h = self.wh
        w, h = block_1.wh

        if block_1.box.box == box.box:
            self.color(block_1.block_id, color)

        elif block_1.block_id == block_2.block_id:
            if x_min == 0 and y_min == 0:
                move_1, (bl_bid, br_bid, tr_bid, tl_bid) = self.pcut(x_max, y_max)
                move_2, __ = self.color(bl_bid, color)
                # move_3, (m1_bid,) = self.merge(bl_bid, br_bid)
                # move_4, (m2_bid,) = self.merge(tr_bid, tl_bid)
                # move_5, (last_bid,) = self.merge(m1_bid, m2_bid)
                # moves += [move_1, move_2, move_3, move_4, move_5]

            elif x_max == w and y_min == 0:
                move_1, (bl_bid, br_bid, tr_bid, tl_bid) = self.pcut(x_min, y_max)
                move_2, __ = self.color(br_bid, color)
                moves, (last_bid,) = self.merge_all()
                # move_3, (m1_bid,) = self.merge(bl_bid, br_bid)
                # move_4, (m2_bid,) = self.merge(tr_bid, tl_bid)
                # move_5, (last_bid,) = self.merge(m1_bid, m2_bid)
                # moves += [move_1, move_2, move_3, move_4, move_5]

            elif x_max == w and y_max == h:
                move_1, (bl_bid, br_bid, tr_bid, tl_bid) = self.pcut(x_min, y_min)
                move_2, __ = self.color(tr_bid, color)
                moves, (last_bid,) = self.merge_all()
                # move_3, (m1_bid,) = self.merge(bl_bid, br_bid)
                # move_4, (m2_bid,) = self.merge(tr_bid, tl_bid)
                # move_5, (last_bid,) = self.merge(m1_bid, m2_bid)
                # moves += [move_1, move_2, move_3, move_4, move_5]

            elif x_min == 0 and y_max == h:
                move_1, (bl_bid, br_bid, tr_bid, tl_bid) = self.pcut(x_max, y_min)
                move_2, __ = self.color(tl_bid, color)
                moves, (last_bid,) = self.merge_all()
                # move_3, (m1_bid,) = self.merge(bl_bid, br_bid)
                # move_4, (m2_bid,) = self.merge(tr_bid, tl_bid)
                # move_5, (last_bid,) = self.merge(m1_bid, m2_bid)
                # moves += [move_1, move_2, move_3, move_4, move_5]

            elif x_min == 0 and x_max == w:
                raise NotImplementedError

            elif y_min == 0 and y_max == h:
                raise NotImplementedError

            elif x_min == 0:
                if abs(y_min - h/2) > abs(y_max - h/2):
                    move_1, (bl_bid, br_bid, tr_bid, tl_bid) = self.pcut(x_max, y_min)
                    move_2, (bottom_bid, top_bid) = self.lcut_horizontal(1, y_max)
                    move_3, __ = self.color(bottom_bid, color)
                else:
                    move_1, (bl_bid, br_bid, tr_bid, tl_bid) = self.pcut(x_max, y_max)
                    move_2, (bottom_bid, top_bid) = self.lcut_horizontal(1, y_min)
                    move_3, __ = self.color(top_bid, color)
                # move_4, (m1_bid,) = self.merge(bottom_bid, top_bid)
                # move_5, (m2_bid,) = self.merge(m1_bid, tr_bid)
                # move_6, (m3_bid,) = self.merge(br_bid, bl_bid)
                # move_7, (last_bid,) = self.merge(m2_bid, m3_bid)
                # moves += [move_1, move_2, move_3, move_4, move_5, move_6, move_7]

            elif y_min == 0:
                if abs(x_min - w/2) > abs(x_max - w/2):
                    move_1, (bl_bid, br_bid, tr_bid, tl_bid) = self.pcut(x_min, y_max)
                    move_2, (left_bid, right_bid) = self.lcut_veritical(x_max, 1)
                    move_3, __ = self.color(left_bid, color)
                else:
                    move_1, (bl_bid, br_bid, tr_bid, tl_bid) = self.pcut(x_max, y_max)
                    move_2, (left_bid, right_bid) = self.lcut_veritical(x_min, 1)
                    move_3, __ = self.color(right_bid, color)
                # move_4, (m1_bid,) = self.merge(left_bid, right_bid)
                # move_5, (m2_bid,) = self.merge(m1_bid, br_bid)
                # move_6, (m3_bid,) = self.merge(tr_bid, tl_bid)
                # move_7, (last_bid,) = self.merge(m2_bid, m3_bid)
                # moves += [move_1, move_2, move_3, move_4, move_5, move_6, move_7]

            elif x_max == w:
                if abs(y_min - h/2) > abs(y_max - h/2):
                    move_1, (bl_bid, br_bid, tr_bid, tl_bid) = self.pcut(x_min, y_min)
                    move_2, (bottom_bid, top_bid) = self.lcut_horizontal(w-2, y_max)
                    move_3, __ = self.color(bottom_bid, color)
                else:
                    move_1, (bl_bid, br_bid, tr_bid, tl_bid) = self.pcut(x_min, y_max)
                    move_2, (bottom_bid, top_bid) = self.lcut_horizontal(w-2, y_min)
                    move_3, __ = self.color(top_bid, color)
                # move_4, (m1_bid,) = self.merge(bottom_bid, top_bid)
                # move_5, (m2_bid,) = self.merge(m1_bid, bl_bid)
                # move_6, (m3_bid,) = self.merge(tr_bid, tl_bid)
                # move_7, (last_bid,) = self.merge(m2_bid, m3_bid)
                # moves += [move_1, move_2, move_3, move_4, move_5, move_6, move_7]

            elif y_max == h:
                if abs(x_min - w/2) > abs(x_max - w/2):
                    move_1, (bl_bid, br_bid, tr_bid, tl_bid) = self.pcut(x_min, y_min)
                    move_2, (left_bid, right_bid) = self.lcut_veritical(x_max, h-2)
                    move_3, __ = self.color(left_bid, color)
                else:
                    move_1, (bl_bid, br_bid, tr_bid, tl_bid) = self.pcut(x_max, y_min)
                    move_2, (left_bid, right_bid) = self.lcut_veritical(x_min, h-2)
                    move_3, __ = self.color(right_bid, color)
                # move_4, (m1_bid,) = self.merge(left_bid, right_bid)
                # move_5, (m2_bid,) = self.merge(m1_bid, tr_bid)
                # move_6, (m3_bid,) = self.merge(br_bid, bl_bid)
                # move_7, (last_bid,) = self.merge(m2_bid, m3_bid)
                # moves += [move_1, move_2, move_3, move_4, move_5, move_6, move_7]

            else:
                bl_dist = abs(x_min - w/2) + abs(y_min - h/w)
                br_dist = abs(x_max - w/2) + abs(y_min - h/w)
                tr_dist = abs(x_min - w/2) + abs(y_min - h/w)
                tl_dist = abs(x_min - w/2) + abs(y_min - h/w)
                max_dist = max(bl_dist, br_dist, tr_dist, tl_dist)
                if bl_dist == max_dist:
                    move_1, (bl_bid_1, br_bid_1, tr_bid_1, tl_bid_1) = self.pcut(x_min, y_min)
                    move_2, (bl_bid_2, br_bid_2, tr_bid_2, tl_bid_2) = self.pcut(x_max, y_max)
                    move_3, __ = self.color(bl_bid_2, color)
                elif br_dist == max_dist:
                    move_1, (bl_bid_1, br_bid_1, tr_bid_1, tl_bid_1) = self.pcut(x_max, y_min)
                    move_2, (bl_bid_2, br_bid_2, tr_bid_2, tl_bid_2) = self.pcut(x_min, y_max)
                    move_3, __ = self.color(br_bid_2, color)
                elif tr_dist == max_dist:
                    move_1, (bl_bid_1, br_bid_1, tr_bid_1, tl_bid_1) = self.pcut(x_max, y_max)
                    move_2, (bl_bid_2, br_bid_2, tr_bid_2, tl_bid_2) = self.pcut(x_min, y_min)
                    move_3, __ = self.color(tr_bid_2, color)
                elif tl_dist == max_dist:
                    move_1, (bl_bid_1, br_bid_1, tr_bid_1, tl_bid_1) = self.pcut(x_min, y_max)
                    move_2, (bl_bid_2, br_bid_2, tr_bid_2, tl_bid_2) = self.pcut(x_max, y_min)
                    move_3, __ = self.color(tl_bid_2, color)
                # move_4, (m1_bid,) = self.merge(bl_bid_2, br_bid_2)
                # move_5, (m2_bid,) = self.merge(tl_bid_2, tr_bid_2)
                # move_6, (m3_bid,) = self.merge(m1_bid, m2_bid)
                # move_7, (m4_bid,) = self.merge(m3_bid, bl_bid_1)
                # move_8, (m5_bid,) = self.merge(tl_bid_1, tr_bid_1)
                # move_9, (last_bid,) = self.merge(m4_bid, m5_bid)
                # moves += [move_1, move_2, move_3, move_4, move_5, move_6, move_7, move_8, move_9]

        else:
            raise NotImplementedError

        if not no_merge:
            moves, (last_bid,) = self.merge_all()

        return moves, (last_bid,)

    def merge_all(self, block_ids: Optional[List[BlockId]]=None) -> Tuple[List[Move], Tuple[BlockId]]:
        moves = []
        merged_bid = None
        if block_ids is None:
            blocks = self.blocks()
        else:
            blocks = self.get_blocks(block_ids)

        while len(blocks) > 1:
            biggest_mergable_block = self.get_biggest_and_mergable_block(blocks)
            mergable_blocks = self.mergable_blocks(biggest_mergable_block)
            smallest_mergable_block = get_smallest_block(mergable_blocks)

            move, (merged_bid,) = self.merge(biggest_mergable_block.block_id, smallest_mergable_block.block_id)
            moves.append(move)
            blocks.append(self.get_blocks([merged_bid])[0])
            blocks.remove(biggest_mergable_block)
            if smallest_mergable_block in blocks:
                blocks.remove(smallest_mergable_block)

        return moves, (merged_bid,)

    def get_biggest_and_mergable_block(self, blocks: List[Block]) -> Optional[Block]:
        blocks = copy(blocks)
        while len(blocks) > 0:
            biggest_block = get_biggest_block(blocks)
            mergable_blocks = self.mergable_blocks(biggest_block)
            if len(mergable_blocks) > 0:
                return biggest_block
            blocks.remove(biggest_block)
        return None

    def get_blocks(self, block_ids: List[BlockId]) -> List[Block]:
        all_blocks = self.blocks()
        result_blocks = []
        for block_id in block_ids:
            for present_block in all_blocks:
                if present_block.block_id == block_id:
                    result_blocks.append(present_block)
        if len(result_blocks) != len(block_ids):
            raise Exception('Not all blocks are present in state')
        return result_blocks

    def mergable_blocks(self, block: BlockId) -> List[Block]:
        result = []
        for other_block in self.blocks():
            if block.mergable(other_block):
                result.append(other_block)
        return result


if __name__ == '__main__':
    box = BoxMk2([1, 2, 3, 4])
    x_min, y_min, x_max, y_max = box.box

    problems_path = Path('../problems')
    target_image = load_image(problems_path / f'{16}.png', revert=True)
    state = State(target_image)

    # move_1, (bid_l, bid_r) = state.lcut_veritical(200, 1)
    # move_2, __ = state.color(bid_l)
    # move_3, (bid_m,) = state.merge(bid_l, bid_r)
    # move_4, __ = state.color(bid_m)
    # move_5, (bl_bid, br_bid, tr_bid, tl_bid) = state.pcut(30, 150)
    # move_6, __ = state.color(bl_bid)
    # move_7, __ = state.color(br_bid)
    # move_8, __ = state.color(tr_bid)

    moves, (main_bid,) = state.color_rect_and_remerge([150, 150, 250, 250])
    moves, main_bid = state.color_rect_and_remerge([0, 0, 20, 25])
    moves, main_bid = state.color_rect_and_remerge([380, 0, 400, 25])
    moves, main_bid = state.color_rect_and_remerge([380, 375, 400, 400])
    moves, main_bid = state.color_rect_and_remerge([0, 375, 20, 400])
    moves, main_bid = state.color_rect_and_remerge([0, 150, 20, 250])
    moves, main_bid = state.color_rect_and_remerge([150, 0, 250, 25])
    moves, main_bid = state.color_rect_and_remerge([380, 150, 400, 250])
    moves, main_bid = state.color_rect_and_remerge([150, 375, 250, 400])

    moves, main_bid = state.color_rect_and_remerge([150, 150, 170, 175])
    moves, main_bid = state.color_rect_and_remerge([230, 150, 250, 175])
    moves, main_bid = state.color_rect_and_remerge([230, 225, 250, 250])
    moves, main_bid = state.color_rect_and_remerge([150, 225, 170, 250])
    moves, main_bid = state.color_rect_and_remerge([190, 190, 210, 210])

    # moves, (bl_bid, br_bid, tr_bid, tl_bid) = state.pcut(30, 150)
    # moves, (main_bid,) = state.merge_all()
    # state.color(main_bid)

    cv2.imshow('cur', state.cur_image())
    cv2.waitKey(0)
