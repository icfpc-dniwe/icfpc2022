import numpy as np
from copy import copy
import numpy.typing as npt
from functools import cached_property
from typing import List, Dict, Union
from dataclasses import dataclass, field
import numba as nb
from enum import Enum
from PIL import Image


Surface = npt.NDArray # 2D or 3D (with channels) array with an image
Picture = Surface[np.uint8]
BWPicture = Surface[np.np.uint32]

BlockNum = int
BlockId = tuple[BlockNum, ...]
Point = npt.NDArray[np.int_]
Color = npt.NDArray[np.int_]
Size = tuple[int, int]
Cost = int


@nb.jit(nopython=True)
def surface_to_norm(surface: Surface) -> PictureNorm:
    return np.sum(surface * surface, axis=2)


def read_picture(path: str) -> Picture:
    img = Image.open(path)


def image_to_picture(image: Image) -> Picture:
    arr = np.array(image.convert("RGBA"))
    return np.copy(np.rot90(arr)[::-1, ::-1])


def surface_to_native_coords(surface):
    return np.rot90(surface[::-1, ::-1], 3)


def picture_to_image(pic: Picture, non_transparent=False) -> Image:
    img = surface_to_native_coords(pic)
    if non_transparent:
        img = np.copy(img)
        img[:, :, 3] = 255
    return Image.fromarray(img)


def pack_channels(pic: Picture) -> BWPicture:
    shift = 0
    ret = np.zeros(pic.shape[:2], dtype=np.uint32)
    for c in range(pic.shape[2]):
        ret += pic[:, :, c] << shift
        shift += 8
    return ret


CANVAS_COLOR = np.full((4,), 255, dtype=np.uint8)

def empty_canvas(shape: tuple) -> Picture:
    return np.full(shape, 255, dtype=np.uint8)


@nb.jit(nopython=True)
def similarity_cost(pic1: Picture, pic2: Picture, alpha=0.005) -> int:
    diff = pic1.astype(np.int64) - pic2
    return round(np.sum(np.sqrt(np.sum(diff * diff, axis=2))) * alpha)


class Orientation(Enum):
    X = 0
    Y = 1
    
    def __str__(self):
        if self == Orientation.X:
            return "x"
        elif self == Orientation.Y:
            return "y"
        else:
            raise RuntimeError("Invalid orientation")


def block_id_str(block_id: BlockId) -> str:
    return ".".join(map(str, block_id))


class Move:
    @staticmethod
    def cost() -> int:
        raise NotImplementedError
        
    def command(self) -> str:
        raise NotImplementedError


@dataclass
class LineCutMove(Move):
    block: BlockId
    orientation: Orientation
    offset: int

    @staticmethod
    def cost():
        return 7

    def command(self):
        return f"cut [{block_id_str(self.block)}] [{self.orientation}] [{self.offset}]"
        

@dataclass
class PointCutMove(Move):
    block: BlockId
    point: Point

    @staticmethod
    def cost():
        return 10

    def command(self):
        point_str = ",".join(map(str, self.point))
        return f"cut [{block_id_str(self.block)}] [{point_str}]"


@dataclass
class ColorMove(Move):
    block: BlockId
    color: Color

    @staticmethod
    def cost():
        return 5

    def command(self):
        color_str = ",".join(map(str, self.color))
        return f"color [{block_id_str(self.block)}] [{color_str}]"


@nb.jit(nopython=True)
def move_cost(canvas_size: Size, block_size: Size, move_cost: int) -> int:
    return round(move_cost * canvas_size[0] * canvas_size[1] / block_size[0] / block_size[1])


def mean_color(pic: Picture) -> Color:
    return np.around(np.mean(pic, axis=(0, 1))).astype(np.uint8)


def print_program(moves: List[Move]):
    print("\n".join(map(lambda x: x.command(), moves)))


@dataclass
class State:
    moves: List[Move] = field(default_factory=list)
    next_global_block: BlockNum = 1
    cost: Cost = 0


@dataclass
class Input:
    picture: Picture
    block: Picture
    block_id: BlockId
    offset: Point
    
    @cached_property
    def mean_color(self):
        return mean_color(self.block)
    
    @cached_property
    def noop_similarity_cost(self):
        return similarity_cost(self.block, CANVAS_COLOR)

    @property
    def noop_total_cost(self):
        return self.noop_similarity_cost
    
    @cached_property
    def color_move_cost(self):
        return move_cost(self.picture.shape, self.block.shape, ColorMove.cost())

    @cached_property
    def color_similarity_cost(self):
        return similarity_cost(self.block, self.mean_color)

    @property
    def color_total_cost(self):
        return self.color_move_cost + self.color_similarity_cost
    
    @property
    def min_total_cost(self):
        return min(self.noop_total_cost, self.color_total_cost)
    

def _solve(state: State, input: Input) -> State:
    color_state = copy(state)
    if input.color_total_cost < input.noop_total_cost:
        color_state.cost += input.color_total_cost
        color_similarity_cost = input.color_similarity_cost
        color_state.moves = color_state.moves + [ColorMove(
            block=input.block_id,
            color=input.mean_color,
        )]
    else:
        color_state.cost += input.noop_total_cost
        color_similarity_cost = input.noop_similarity_cost

    if color_similarity_cost < 0.25 * input.block.shape[0] * input.block.shape[1]:
        #print(f"Exiting early, block id {input.block_id}")
        return color_state
    
    cut_point = find_cut_point(input.block)
    
    line_cut_cost = move_cost(input.picture.shape, input.block.shape, LineCutMove.cost())
    
    if input.block.shape[1] <= 1 or cut_point[0] <= 0 or cut_point[0] >= input.block.shape[0] - 1:
        horizontal_cost = None
    else:
        input_horizontal_0 = Input(
            picture=input.picture,
            block=input.block[:cut_point[0], :],
            block_id=input.block_id + (0,),
            offset=input.offset,
        )
        input_horizontal_1 = Input(
            picture=input.picture,
            block=input.block[cut_point[0]:, :],
            block_id=input.block_id + (1,),
            offset=np.array([input.offset[0] + cut_point[0], input.offset[1]]),
        )
        horizontal_cost = \
            line_cut_cost + \
            input_horizontal_0.min_total_cost + \
            input_horizontal_1.min_total_cost

    if input.block.shape[0] <= 1 or cut_point[1] <= 0 or cut_point[1] >= input.block.shape[1] - 1:
        vertical_cost = None
    else:
        input_vertical_0 = Input(
            picture=input.picture,
            block=input.block[:, :cut_point[1]],
            block_id=input.block_id + (0,),
            offset=input.offset,
        )
        input_vertical_1 = Input(
            picture=input.picture,
            block=input.block[:, cut_point[1]:],
            block_id=input.block_id + (1,),
            offset=np.array([input.offset[0], input.offset[1] + cut_point[1]]),
        )
        vertical_cost = \
            line_cut_cost + \
            input_vertical_0.min_total_cost + \
            input_vertical_1.min_total_cost

    if horizontal_cost is None or vertical_cost is None:
        point_cost = None
    else:
        point_cut_cost = move_cost(input.picture.shape, input.block.shape, PointCutMove.cost())
        input_point_0 = Input(
            picture=input.picture,
            block=input.block[:cut_point[0], :cut_point[1]],
            block_id=input.block_id + (0,),
            offset=input.offset,
        )
        input_point_1 = Input(
            picture=input.picture,
            block=input.block[cut_point[0]:, :cut_point[1]],
            block_id=input.block_id + (1,),
            offset=np.array([input.offset[0] + cut_point[0], input.offset[1]]),
        )
        input_point_2 = Input(
            picture=input.picture,
            block=input.block[cut_point[0]:, cut_point[1]:],
            block_id=input.block_id + (2,),
            offset=input.offset + cut_point,
        )
        input_point_3 = Input(
            picture=input.picture,
            block=input.block[:cut_point[0], cut_point[1]:],
            block_id=input.block_id + (3,),
            offset=np.array([input.offset[0], input.offset[1] + cut_point[1]]),
        )
        point_cost = \
            point_cut_cost + \
            input_point_0.min_total_cost + \
            input_point_1.min_total_cost + \
            input_point_2.min_total_cost + \
            input_point_3.min_total_cost
    
    min_cut_cost = None
    if horizontal_cost is not None:
        min_cut_cost = horizontal_cost if min_cut_cost is None else min(min_cut_cost, horizontal_cost)
    if vertical_cost is not None:
        min_cut_cost = vertical_cost if min_cut_cost is None else min(min_cut_cost, vertical_cost)
    if point_cost is not None:
        min_cut_cost = point_cost if min_cut_cost is None else min(min_cut_cost, point_cost)

    if min_cut_cost is None:
        cut_state = None
    elif horizontal_cost == min_cut_cost:
        cut_state = copy(state)
        cut_state.cost += line_cut_cost
        cut_state.moves = cut_state.moves + [LineCutMove(
            block=input.block_id,
            orientation=Orientation.X,
            offset=input.offset[0] + cut_point[0],
        )]
        cut_state = _solve(cut_state, input_horizontal_0)
        cut_state = _solve(cut_state, input_horizontal_1)
    elif vertical_cost == min_cut_cost:
        cut_state = copy(state)
        cut_state.cost += line_cut_cost
        cut_state.moves = cut_state.moves + [LineCutMove(
            block=input.block_id,
            orientation=Orientation.Y,
            offset=input.offset[1] + cut_point[1],
        )]
        cut_state = _solve(cut_state, input_vertical_0)
        cut_state = _solve(cut_state, input_vertical_1)
    elif point_cost == min_cut_cost:
        cut_state = copy(state)
        cut_state.cost += point_cut_cost
        cut_state.moves = cut_state.moves + [PointCutMove(
            block=input.block_id,
            point=input.offset + cut_point,
        )]
        cut_state = _solve(cut_state, input_point_0)
        cut_state = _solve(cut_state, input_point_1)
        cut_state = _solve(cut_state, input_point_2)
        cut_state = _solve(cut_state, input_point_3)

    if cut_state is not None and cut_state.cost < color_state.cost:
        return cut_state
    else:
        return color_state


def solve(picture: Picture) -> State:
    state = State()
    input = Input(
        picture=picture,
        block=picture,
        block_id=(0,),
        offset=np.zeros((2,), np.int_),
    )
    return _solve(state, input)
