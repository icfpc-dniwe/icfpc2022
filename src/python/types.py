import numpy as np
import typing as t


RGBAImage = np.ndarray  # HxWxC image in RGBA channel format
LabelImage = np.ndarray  # HxW matrix of labels (np.int32)
Box = np.ndarray  # bounding box in (x_min, y_min, x_max, y_max) format
