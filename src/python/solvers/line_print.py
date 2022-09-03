from pathlib import Path

import cv2
import numpy as np
import typing as t

from ..image_utils import load_image
from ..types import RGBAImage, Box, Program, Orientation
from ..moves import LineCut, PointCut, Merge, Color, Move


def render_by_line(img: RGBAImage) -> t.Tuple[RGBAImage, t.List[Move]]:
    cur_block_prefix = '0'
    global_block_id = 0
    h, w = img.shape[:2]
    new_canvas = np.zeros_like(img)

    def merge_columns():
        cur_global_id = global_block_id
        moves = [Merge(cur_columns[0], cur_columns[1])]
        cur_global_id += 1
        if len(cur_columns) > 2:
            for cur_col in cur_columns[2:]:
                moves.append(Merge(f'{cur_global_id}', cur_col))
                cur_global_id += 1
        return moves, cur_global_id

    def color_needed(x, y):
        return np.any(new_canvas[y, x] != img[y, x])

    program = []
    cur_columns = []
    need_secure = False
    for cur_line in range(h):
        for cur_x in range(w):
            # cv_line = h - cur_line - 1
            cv_line = cur_line
            if color_needed(cur_x, cv_line):
                if need_secure:
                    new_moves, new_id = merge_columns()
                    program += new_moves
                    global_block_id = new_id
                    program.append(LineCut(f'{global_block_id}', Orientation.Y, cur_line))
                    cur_block_prefix = f'{global_block_id}.1'
                    cur_columns = []
                    need_secure = False
                if len(cur_columns) == 0:
                    # a new cut
                    cur_prefix = cur_block_prefix
                else:
                    # continue cutting
                    cur_prefix = cur_columns.pop()
                new_canvas[:cv_line+1, cur_x:] = img[cv_line, cur_x]
                program.append(Color(f'{cur_prefix}', img[cv_line, cur_x]))
                if cur_x + 1 < w:
                    program.append(LineCut(f'{cur_prefix}', Orientation.X, cur_x + 1))
                    cur_columns += [f'{cur_prefix}.0', f'{cur_prefix}.1']
        if cur_line + 1 < h:
            # securing just printed line
            if len(cur_columns) > 1:
                need_secure = True
    return new_canvas, program
