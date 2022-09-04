import numpy as np
import cv2
from tqdm import tqdm
import typing as t
from itertools import product
from more_itertools import unzip

from .straight import render_straight
from ..mosaic.fill import expand_pixel
from ..scoring import block_similarity, image_similarity, static_cost
from ..box_utils import box_size, get_part
from ..moves import Color as ColorMove
from ..types import RGBAImage, Box, Color, Program


def expand_cost_fn(
        source_img: RGBAImage,
        cur_img: RGBAImage,
        new_color: Color,
        box: Box,
        img_area: int,
        move_lambda: float = 5
) -> t.Optional[float]:
    left_sim = block_similarity(get_part(source_img, box), new_color)
    right_sim = image_similarity(get_part(source_img, box), get_part(cur_img, box))
    if left_sim > right_sim:
        return None
    else:
        return left_sim - right_sim + move_lambda * static_cost(ColorMove, box_size(box), img_area)


def mult_box(cur_box: Box, mult: float) -> Box:
    x1, y1, x2, y2 = cur_box
    bw = x2 - x1
    bh = y2 - y1
    return np.array([x1 * mult, y1 * mult, (x1 * mult) + bw * mult, (y1 * mult) + bh * mult], dtype=np.int64)


def get_boxes(
        img: RGBAImage,
        starting_points: t.List[t.Tuple[int, int]]
) -> t.Tuple[t.List[t.Tuple[Box, Color]], float]:
    h, w = img.shape[:2]
    real_canvas_area = h * w
    move_lambda = 2
    # r_img = cv2.resize(img, (50, 50))
    # r_h, r_w = r_img.shape[:2]
    real_rendered_canvas = np.zeros_like(img) + 255
    boxes = []
    colors = []
    for cur_x, cur_y in tqdm(starting_points):
        cur_color = img[cur_y, cur_x]
        cur_scorring = lambda box: expand_cost_fn(img, real_rendered_canvas, cur_color, box,
                                                  real_canvas_area, move_lambda)
        new_box = expand_pixel((cur_x, cur_y), h, w, cur_scorring)
        if new_box is not None:
            score = cur_scorring(new_box)
            # print(cur_y, cur_x, '|> adding a new box?:', new_box, cur_color, score)
            if score < 0:
                # print(cur_y, cur_x, '|> adding a new box:', new_box, cur_color)
                boxes.append(new_box)
                colors.append(cur_color)
                real_rendered_canvas[new_box[1]:new_box[3], new_box[0]:new_box[2]] = cur_color
    sim_score = image_similarity(img, real_rendered_canvas)
    render_score = sum([5 * static_cost(ColorMove, box_size(cur_box), real_canvas_area) for cur_box in boxes])
    total_score = sim_score + render_score
    print('Sim score:', sim_score)
    print('Rendering score:', render_score)
    print('Total score:', total_score)
    return list(zip(boxes, colors)), total_score


def produce_program(
        img: RGBAImage,
        num_random_starts: int = 0,
        num_random_points: int = 1000
) -> t.Tuple[RGBAImage, Program]:
    h, w = img.shape[:2]
    starting_points = list(product(range(0, w, 8), range(0, h, 8)))
    results, score = get_boxes(img, starting_points)
    top_score = score
    top_results = results
    for _ in range(num_random_starts):
        starting_x = np.random.randint(0, w, size=num_random_points)
        starting_y = np.random.randint(0, h, size=num_random_points)
        starting_points = list(zip(starting_x, starting_y))
        results, score = get_boxes(img, starting_points)
        if score < top_score:
            top_score = score
            top_results = results
    print('Top Score:', top_score)
    print('Num boxes:', len(top_results))
    if len(top_results) < 1:
        return img, ['# nothing to do here']
    boxes, colors = unzip(top_results)
    new_boxes = []
    for cur_box in boxes:
        if cur_box[0] == 0:
            cur_box[0] += 1
        if cur_box[1] == 0:
            cur_box[1] += 1
        if cur_box[2] == w:
            cur_box[2] -= 1
        if cur_box[3] == h:
            cur_box[3] -= 1
        new_boxes.append(cur_box)
    return render_straight(img, new_boxes, list(colors))
