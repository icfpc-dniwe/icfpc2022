import numpy as np
from enum import Enum, auto
import typing as t


RGBAImage = np.ndarray  # HxWxC image in RGBA channel format
LabelImage = np.ndarray  # HxW matrix of labels (np.int32)
Box = t.Union[np.ndarray, t.Tuple[int, int, int, int]]  # bounding box in (x_min, y_min, x_max, y_max) format
Program = t.Sequence[str]
Color = t.Union[np.ndarray, t.Tuple[int, int, int, int]]  # RGBA color
BlockId: str


class Orientation(Enum):
    X = auto()
    Y = auto()

    def __str__(self):
        if self is self.X:
            return 'X'
        else:
            return 'Y'
