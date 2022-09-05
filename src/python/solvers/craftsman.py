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

    return state


def problem_1_2() -> State:
    target_image = load_image(problems_path / f'{1}.png', revert=True)
    state = State(target_image)

    h = 400

    blue = np.array([0, 74, 173, 255])
    white = np.array([255, 255, 255, 255])
    black = np.array([0, 0, 0, 255])

    coords_x = [0, 40, 79, 119, 158, 198, 237, 277, 317, 356]
    coords_y = [0, 40, 80, 120, 160, 199, 238, 278, 318, 357]
    coords_y = [400-y for y in coords_y]
    coords_y = list(reversed(coords_y))

    desk_box = [0, h-357, 356, 400]

    state.color('0', blue)
    __, (__, __, __, desk_bid) = state.pcut(356, h-357)
    state.color(desk_bid, black)
    __, (bl, br, tr, tl) = state.pcut(coords_x[8], coords_y[4])
    state.color(tl, white)

    state.pcut(coords_x[4], coords_y[7])

    state.lcut_horizontal(coords_x[2], coords_y[8])
    state.pcut(coords_x[1], coords_y[8])
    state.pcut(coords_x[3], coords_y[8])
    state.pcut(coords_x[2], coords_y[6])
    state.pcut(coords_x[1], coords_y[5])
    state.pcut(coords_x[3], coords_y[5])
    state.lcut_horizontal(coords_x[1], coords_y[7])
    state.lcut_horizontal(coords_x[3], coords_y[7])

    state.lcut_horizontal(coords_x[2+4], coords_y[8])
    state.pcut(coords_x[1+4], coords_y[8])
    state.pcut(coords_x[3+4], coords_y[8])
    state.pcut(coords_x[2+4], coords_y[6])
    state.pcut(coords_x[1+4], coords_y[5])
    state.pcut(coords_x[3+4], coords_y[5])
    state.lcut_horizontal(coords_x[1+4], coords_y[7])
    state.lcut_horizontal(coords_x[3+4], coords_y[7])

    state.pcut(coords_x[4], coords_y[2])
    state.lcut_horizontal(coords_x[2], coords_y[3])
    state.pcut(coords_x[1], coords_y[3])
    state.pcut(coords_x[3], coords_y[3])
    state.lcut_horizontal(coords_x[2], coords_y[1])
    state.pcut(coords_x[1], coords_y[1])
    state.pcut(coords_x[3], coords_y[1])
    state.lcut_horizontal(coords_x[6], coords_y[3])
    state.pcut(coords_x[5], coords_y[3])
    state.pcut(coords_x[7], coords_y[3])
    state.lcut_horizontal(coords_x[6], coords_y[1])
    state.pcut(coords_x[5], coords_y[1])
    state.pcut(coords_x[7], coords_y[1])

    state.lcut_veritical(coords_x[8], coords_y[1])
    state.lcut_veritical(coords_x[8], coords_y[3])
    state.lcut_veritical(coords_x[8], coords_y[5])
    state.lcut_veritical(coords_x[8], coords_y[6])
    state.lcut_veritical(coords_x[8], coords_y[7])
    state.lcut_veritical(coords_x[8], coords_y[8])

    # TODO swap

    return state


def problem_4() -> State:
    target_image = load_image(problems_path / f'{4}.png', revert=True)
    state = State(target_image)

    h = 400
    white = np.array([255, 255, 255, 255])
    black = np.array([0, 0, 0, 255])

    boxes = {}
    # boxes['I_bar'] = [74, 261, 86, 321]
    # boxes['C_left_bar'] = [99, 271, 110, 311]
    # boxes['C_top_bar'] = [109, 262, 148, 272]
    # boxes['C_bottom_bar'] = [109, 311, 148, 321]
    # boxes['F_left_bar'] = [161, 261, 171,321]
    # boxes['F_top_bar'] = [161, 262, 207, 272]
    # boxes['F_medium_bar'] = [161, 282, 192, 293]
    # boxes['F2_left_bar'] = [215, 262, 226, 321]
    # boxes['F2_top_bar'] = [215, 262, 254, 272]
    # boxes['F2_medium_bar'] = [215, 282, 255, 293]
    # boxes['C2_left_bar'] = [273, 271, 284, 311]
    # boxes['C2_top_bar'] = [283, 262, 324, 272]
    # boxes['C2_bottom_bar'] = [282, 311, 323, 321]
    #
    # boxes['l_left_leg_1'] = [149, 222, 179, 239]
    # boxes['l_left_leg_2'] = [158, 205, 179, 230]
    # boxes['l_left_leg_3'] = [169, 189, 186, 223]
    # boxes['l_left_leg_4'] = [178, 173, 195, 206]
    # boxes['l_left_leg_5'] = [185, 156, 202, 189]
    # boxes['l_center'] = [194, 137, 219, 166]
    # boxes['l_right_leg_1'] = [227, 218, 271, 239]
    # boxes['l_head'] = [162, 69, 199, 85]

    boxes['top_line'] = [70,262, 324, 273]

    for i, (__, box) in enumerate(boxes.items()):
        fixed_box = [box[0], h-box[3], box[2], h-box[1]]
        if i == len(boxes)-1:
            state.color_rect_and_remerge(fixed_box, black, no_merge=True)
        else:
            state.color_rect_and_remerge(fixed_box, black)

    return state


if __name__ == '__main__':
    state = problem_1_2()
    # print(f'State total cost: {state.total_cost()}')

    program = get_program(state.moves())
    with open('test_craftsman.txt', 'w') as f:
        print('\n'.join(program), file=f)

    print(state.similarity())

    cv2.imshow('cur', cv2.flip(state.cur_image(), 0))
    cv2.waitKey(0)
