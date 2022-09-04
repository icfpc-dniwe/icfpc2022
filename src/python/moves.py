import numpy as np
import typing as t
from .types import Orientation, Program


def add_line_cut_move(block_id: str, orientation: Orientation, offset: int) -> str:
    return f'cut [{block_id}] [{orientation}] [{offset}]'


def add_point_cut_move(block_id: str, offset_point: t.Union[np.ndarray, t.Tuple[int, int]]) -> str:
    return f'cut [{block_id}] [{offset_point[0]},{offset_point[1]}]'


def add_color_move(block_id: str, color: t.Union[np.ndarray, t.Tuple[int, int, int, int]]) -> str:
    return f'color [{block_id}] [{int(color[0])},{int(color[1])},{int(color[2])},{int(color[3])}]'


def add_merge_move(left_block_id: str, right_block_id: str) -> str:
    return f'merge [{left_block_id}] [{right_block_id}]'


def add_swap_move(left_block_id: str, right_block_id: str) -> str:
    return f'swap [{left_block_id}] [{right_block_id}]'


class Move:
    @staticmethod
    def cost() -> int:
        raise NotImplementedError

    def program(self) -> str:
        raise NotImplementedError


class PointCut(Move):
    def __init__(self, block_id: str, offset_point: t.Union[np.ndarray, t.Tuple[int, int]]):
        super().__init__()
        self.block_id = block_id
        self.offset_point = offset_point

    @staticmethod
    def cost() -> int:
        return 10

    def program(self) -> str:
        return add_point_cut_move(self.block_id, self.offset_point)


class LineCut(Move):
    def __init__(self, block_id: str, orientation: Orientation, offset: int):
        super().__init__()
        self.block_id = block_id
        self.orientation = orientation
        self.offset = offset

    @staticmethod
    def cost() -> int:
        return 7

    def program(self) -> str:
        return add_line_cut_move(self.block_id, self.orientation, self.offset)


class ColorMove(Move):
    def __init__(self, block_id: str, color: t.Union[np.ndarray, t.Tuple[int, int, int, int]]):
        super().__init__()
        self.block_id = block_id
        self.color = color

    @staticmethod
    def cost() -> int:
        return 5

    def program(self) -> str:
        return add_color_move(self.block_id, self.color)


class Swap(Move):
    def __init__(self, left_block_id: str, right_block_id: str):
        super().__init__()
        self.left_block_id = left_block_id
        self.right_block_id = right_block_id

    @staticmethod
    def cost() -> int:
        return 3

    def program(self) -> str:
        return add_swap_move(self.left_block_id, self.right_block_id)


class Merge(Move):
    def __init__(self, left_block_id: str, right_block_id: str):
        super().__init__()
        self.left_block_id = left_block_id
        self.right_block_id = right_block_id

    @staticmethod
    def cost() -> int:
        return 1

    def program(self) -> str:
        return add_merge_move(self.left_block_id, self.right_block_id)


def get_program(moves: t.Iterable[Move]) -> Program:
    return list(map(lambda m: m.program(), moves))
