import numpy as np
import cv2
import typing as t
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


def get_boxes(img: RGBAImage) -> t.List[t.Tuple[Box, Color]]:
    h, w = img.shape[:2]
    real_canvas_area = h * w
    r_img = cv2.resize(img, (50, 50))
    r_h, r_w = r_img.shape[:2]
    real_rendered_canvas = np.zeros_like(img) + 255
    boxes = []
    colors = []
    for cur_x in range(r_w):
        for cur_y in range(r_h):
            cur_color = r_img[cur_y, cur_x]
            cur_scorring = lambda box: expand_cost_fn(img, real_rendered_canvas, cur_color, mult_box(box, 8),
                                                      real_canvas_area)
            new_box = expand_pixel((cur_x, cur_y), r_h, r_w, cur_scorring)
            if new_box is not None:
                score = cur_scorring(new_box)
                new_box = mult_box(new_box, 8)
                print(cur_y, cur_x, '|> adding a new box?:', new_box, cur_color, score)
                if score < 0:
                    print(cur_y, cur_x, '|> adding a new box:', new_box, cur_color)
                    boxes.append(new_box)
                    colors.append(cur_color)
                    real_rendered_canvas[new_box[1]:new_box[3], new_box[0]:new_box[2]] = cur_color
    sim_score = image_similarity(img, real_rendered_canvas)
    render_score = sum([static_cost(ColorMove, box_size(cur_box), real_canvas_area) for cur_box in boxes])
    total_score = sim_score + render_score
    print('Sim score:', sim_score)
    print('Rendering score:', render_score)
    print('Total score:', total_score)
    return list(zip(boxes, colors))


def produce_program(img: RGBAImage) -> t.Tuple[RGBAImage, Program]:
    h, w = img.shape[:2]
    results = get_boxes(img)
    boxes, colors = unzip(results)
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