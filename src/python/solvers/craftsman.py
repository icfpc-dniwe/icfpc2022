from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

from python.image_utils import load_image
from python.moves import Move, get_program
from python.state import State

problems_path = Path('../problems')


def problem_1() -> State:
    target_image = load_image(problems_path / f'{1}.png', revert=True)
    state = State(target_image)

    blue = np.array([0, 74, 173, 255])
    white = np.array([255, 255, 255, 255])
    black = np.array([0, 0, 0, 255])

    coords_x = [0, 40, 79, 119, 158, 198, 237, 277, 317, 356]
    coords_y = [0, 40, 80, 120, 160, 199, 238, 278, 318, 357]
    coords_y = [400-y for y in coords_y]
    coords_y = list(reversed(coords_y))

    desk_box = [0, 400-357, 356, 400]
    white_boxes = []
    for i in range(0, 81):
        if i % 2 == 0 and i != 8:
            white_boxes.append([coords_x[i%9], coords_y[i//9], coords_x[i%9 + 1], coords_y[i//9 + 1]])

    blue_box = [coords_x[-2], coords_y[0], coords_x[-1], coords_y[1]]

    state.color('0', blue)
    state.color_rect_and_remerge(desk_box, black)
    for box in white_boxes:
        state.color_rect_and_remerge(box, white)
    state.color_rect_and_remerge(blue_box, blue)

    # __, (bid_m,) = state.merge(bid_l, bid_r)
    # __, __ = state.color(bid_m)
    # __, (bl_bid, br_bid, tr_bid, tl_bid) = state.pcut(30, 150)
    # __, __ = state.color(bl_bid)
    # __, __ = state.color(br_bid)
    # __, __ = state.color(tr_bid)

    return state


if __name__ == '__main__':
    state = problem_1()
    # print(f'State total cost: {state.total_cost()}')

    program = get_program(state.moves())
    with open('test_craftsman.txt', 'w') as f:
        print('\n'.join(program), file=f)

    print(state.similarity())

    cv2.imshow('cur', state.cur_image())
    cv2.waitKey(0)
