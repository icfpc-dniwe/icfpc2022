import sys
import typing as t
from pathlib import Path

import cv2
import numpy as np

from ..image_utils import get_area, get_average_color, load_image
from ..moves import add_point_cut_move, add_color_move
from ..types import RGBAImage, Program, Box


def solve_by_splitting_evenly_and_coloring_each_block_by_its_average_color(source_img: RGBAImage, block_max_size: int) -> t.Tuple[RGBAImage, Program]:
    canvas = np.zeros_like(source_img)
    whole_prog = []
    max_y = canvas.shape[0]

    def get_max_depth(whole_size: int, block_max_size: int) -> int:
        blocks_max_num = whole_size / block_max_size

        tmp = blocks_max_num
        depth = 0
        while tmp >= 4:
            tmp /= 4
            depth += 1

        return depth

    whole_size = get_area(source_img)
    depth = get_max_depth(whole_size, block_max_size)

    def pcut_recursively_and_color_at_end(cur_box: Box, cur_depth: int, prefix: str) -> t.List[str]:
        cur_prog = []
        x_min, y_min, x_max, y_max = cur_box
        x_min, y_min, x_max, y_max =  int(x_min), int(y_min), int(x_max), int(y_max)
        x_new, y_new = int((x_min + x_max)/2), int((y_min + y_max)/2)

        if cur_depth > 0:
            cur_prog += [add_point_cut_move(f'{prefix}', (x_new, y_new))]
            cur_prog += pcut_recursively_and_color_at_end(np.array([x_min, y_min, x_new, y_new]), cur_depth-1, prefix=f'{prefix}.0')
            cur_prog += pcut_recursively_and_color_at_end(np.array([x_new, y_min, x_max, y_new]), cur_depth-1, prefix=f'{prefix}.1')
            cur_prog += pcut_recursively_and_color_at_end(np.array([x_new, y_new, x_max, y_max]), cur_depth-1, prefix=f'{prefix}.2')
            cur_prog += pcut_recursively_and_color_at_end(np.array([x_min, y_new, x_new, y_max]), cur_depth-1, prefix=f'{prefix}.3')

        else:
            # cv2.imshow(prefix, source_img[y_min:y_max, x_min:x_max])
            # cv2.waitKey(0)
            # color = get_average_color(source_img[max_y - y_max:max_y-y_min, x_min:x_max])
            color = get_average_color(source_img[y_min:y_max, x_min:x_max])
            cur_prog.append(add_color_move(f'{prefix}', color))
            canvas[y_min:y_max, x_min:x_max] = color
            # cv2.imshow(prefix, canvas[y_min:y_max, x_min:x_max])
            # cv2.waitKey(0)

        return cur_prog

    whole_prog += pcut_recursively_and_color_at_end(np.array([0, 0, source_img.shape[1], source_img.shape[0]]), depth, prefix=f'0')

    return canvas, whole_prog


if __name__ == '__main__':
    problem_num = sys.argv[1]

    problems_path = Path('../problems')
    img = load_image(problems_path / f'{problem_num}.png')

    # cv2.imshow('a', img)
    # cv2.waitKey(0)

    canvas, prog = solve_by_splitting_evenly_and_coloring_each_block_by_its_average_color(img, 1236)

    with open('test_average_min_block.txt', 'w') as f:
        print('\n'.join(prog), file=f)

    # cv2.imshow('a', canvas)
    # cv2.waitKey(0)
