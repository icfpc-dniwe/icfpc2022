import typing as t

import numpy as np

from ..image_utils import get_area, get_average_color
from ..moves import add_point_cut_move, add_color_move
from ..types import RGBAImage, Program, Box


# TODO debug
def solve_by_splitting_evenly_and_coloring_each_block_by_its_average_color(source_img: RGBAImage, block_max_size: int) -> t.Tuple[RGBAImage, Program]:
    canvas = np.zeros_like(source_img)
    whole_prog = []

    def get_max_depth(whole_size: int, block_max_size: int) -> int:
        blocks_max_num = whole_size / block_max_size

        tmp = blocks_max_num
        print(blocks_max_num)
        depth = 0
        while tmp >= 4:
            tmp /= 4
            depth += 1

        return depth

    whole_size = get_area(source_img)
    depth = get_max_depth(whole_size, block_max_size)


    def pcut_recursively_and_color_at_end(cur_box: Box, cur_depth: int, prefix: str):
        cur_prog = []
        x_min, y_min, x_max, y_max = cur_box
        x_new, y_new = (x_min + x_max)/2, (y_min + y_max)/2

        if cur_depth > 0:
            cur_prog += [
                add_point_cut_move(f'{prefix}', (x_new, y_new)),
                pcut_recursively_and_color_at_end(np.array([x_min, y_min, x_new, y_new]), cur_depth-1, prefix=f'{prefix}.0'),
                pcut_recursively_and_color_at_end(np.array([x_new, y_min, x_max, y_new]), cur_depth-1, prefix=f'{prefix}.1'),
                pcut_recursively_and_color_at_end(np.array([x_new, y_new, x_max, y_max]), cur_depth-1, prefix=f'{prefix}.2'),
                pcut_recursively_and_color_at_end(np.array([x_min, y_new, x_new, y_max]), cur_depth-1, prefix=f'{prefix}.3'),
            ]

        else:
            color = get_average_color()
            cur_prog += [
                add_color_move(f'{prefix}', color)
            ]
            canvas[y_min:y_max, x_min:x_max] = color

        return cur_prog

    whole_prog += pcut_recursively_and_color_at_end(np.array([0, 0, source_img.shape[1], source_img.shape[0]]), depth, prefix=f'0')

    return canvas, whole_prog


if __name__ == '__main__':
    pass


def get_area(img: RGBAImage) -> int:
    return img.shape[0] * img.shape[1]


def get_average_color(img: RGBAImage):
    return np.array([np.average(img[:, :, i]) for i in range(4)])
