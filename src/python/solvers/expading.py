import numpy as np
import cv2
from tqdm import tqdm
import typing as t
from itertools import product
from more_itertools import unzip

from .straight import render_straight
from ..mosaic.fill import expand_pixel
from ..scoring import block_similarity, image_similarity, static_cost, score_program_agaist_nothing
from ..box_utils import box_size, get_part
from ..moves import ColorMove, Move, EmptyMove
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
    # if left_sim > right_sim:
    #     return None
    # else:
    #     return left_sim - right_sim + move_lambda * static_cost(ColorMove, box_size(box), img_area)
    return left_sim - right_sim + move_lambda * static_cost(ColorMove, box_size(box), img_area)


def mult_box(cur_box: Box, mult: float) -> Box:
    x1, y1, x2, y2 = cur_box
    bw = x2 - x1
    bh = y2 - y1
    return np.array([x1 * mult, y1 * mult, (x1 * mult) + bw * mult, (y1 * mult) + bh * mult], dtype=np.int64)


def get_boxes(
        img: RGBAImage,
        starting_points: t.List[t.Tuple[int, int]],
        tol_iter: int = 3,
        return_best: bool = True,
        choose_thresholds: t.Optional[t.Sequence[float]] = None
) -> t.Tuple[t.List[t.Tuple[Box, Color]], float]:
    if choose_thresholds is None:
        choose_thresholds = [0] * len(starting_points)
    h, w = img.shape[:2]
    real_canvas_area = h * w
    move_lambda = 2
    # r_img = cv2.resize(img, (50, 50))
    # r_h, r_w = r_img.shape[:2]
    real_rendered_canvas = np.zeros_like(img) + 255
    boxes = []
    colors = []
    for point_idx, (cur_x, cur_y) in enumerate(tqdm(starting_points)):
        cur_color = img[cur_y, cur_x]
        cur_scorring = lambda box: expand_cost_fn(img, real_rendered_canvas, cur_color, box,
                                                  real_canvas_area, move_lambda)
        new_box = expand_pixel((cur_x, cur_y), h, w, cur_scorring, tol_iter=tol_iter, return_best=return_best)
        if new_box is not None:
            score = cur_scorring(new_box)
            # print(cur_y, cur_x, '|> adding a new box?:', new_box, cur_color, score)
            if score < choose_thresholds[point_idx]:
                # print(cur_y, cur_x, '|> adding a new box:', new_box, cur_color)
                boxes.append(new_box)
                colors.append(cur_color)
                real_rendered_canvas[new_box[1]:new_box[3], new_box[0]:new_box[2]] = cur_color
    sim_score = image_similarity(img, real_rendered_canvas)
    render_score = sum([move_lambda * static_cost(ColorMove, box_size(cur_box), real_canvas_area) for cur_box in boxes])
    total_score = sim_score + render_score
    print('Sim score:', sim_score)
    print('Rendering score:', render_score)
    print('Total score:', total_score)
    return list(zip(boxes, colors)), total_score


def filter_unneeded_boxes(boxes: t.Sequence[t.Tuple[Box, t.Any]], height: int, width: int) -> t.List[t.Tuple[Box, t.Any]]:
    new_boxes = []
    canvas = np.zeros((height, width), dtype=np.uint8)
    for cur_box, other in reversed(boxes):
        if np.any(canvas[cur_box[1]:cur_box[3], cur_box[0]:cur_box[2]] == 0):
            new_boxes.append((cur_box, other))
            canvas[cur_box[1]:cur_box[3], cur_box[0]:cur_box[2]] = 255
    return list(reversed(new_boxes))


def prepare_boxes(boxes: t.Sequence[t.Tuple[Box, t.Any]], height: int, width: int) -> t.List[t.Tuple[Box, t.Any]]:
    res = filter_unneeded_boxes(boxes, height, width)
    new_boxes = []
    for cur_box, other in res:
        if cur_box[0] == 0:
            cur_box[0] += 1
        if cur_box[1] == 0:
            cur_box[1] += 1
        if cur_box[2] == height:
            cur_box[2] -= 1
        if cur_box[3] == width:
            cur_box[3] -= 1
        new_boxes.append((cur_box, other))
    return new_boxes


def produce_program(
        img: RGBAImage,
        num_random_starts: int = 0,
        num_random_points: int = 1000,
        default_canvas: t.Optional[RGBAImage] = None
) -> t.Tuple[RGBAImage, t.List[Move]]:
    if default_canvas is None:
        default_canvas = np.zeros_like(img) + 255
    h, w = img.shape[:2]
    starting_points = list(product(range(0, w, 100), range(0, h, 100)))
    results, score = get_boxes(img, starting_points)
    top_score = np.inf
    top_results = results
    top_idx = -1
    params = []
    for cur_idx in range(num_random_starts):
        starting_x = np.random.randint(0, w, size=num_random_points)
        starting_y = np.random.randint(0, h, size=num_random_points)
        cur_tolerance = np.random.randint(0, 20)
        cur_return_best = np.random.rand() < 0.8
        if np.random.rand() < 0.5:
            max_threshold = np.exp(np.random.rand() * 7)
            min_threshold = -np.exp(np.random.rand() * 7)
            thresholds = np.linspace(min_threshold, max_threshold, num=num_random_points)[::-1]
        else:
            thresholds = [0] * num_random_points
        starting_points = list(zip(starting_x, starting_y))
        params.append((cur_tolerance, cur_return_best, np.max(thresholds), np.min(thresholds)))
        print('Hyperparameters:')
        print('Tolerance:', cur_tolerance)
        print('Return best:', cur_return_best)
        print('Max threshold:', np.max(thresholds))
        print('Min threshold:', np.min(thresholds))
        results, score = get_boxes(img, starting_points,
                                   tol_iter=cur_tolerance,
                                   return_best=cur_return_best,
                                   choose_thresholds=thresholds)
        if len(results) > 0:
            results = prepare_boxes(results, h, w)
            boxes, colors = unzip(results)
            new_canvas, moves = render_straight(img, list(boxes), list(colors))
            _, score = score_program_agaist_nothing(img, default_canvas, new_canvas, (0, 0, 1, 1), moves)
            print('New points score:', score)
        if score < top_score:
            top_score = score
            top_results = results
            top_idx = cur_idx
    print('Top Score:', top_score)
    print('Num boxes:', len(top_results))
    print('Top hyperparameters:')
    print('Tolerance:', params[top_idx][0])
    print('Return best:', params[top_idx][1])
    print('Max threshold:', params[top_idx][2])
    print('Min threshold:', params[top_idx][3])
    if len(top_results) < 1:
        return img, [EmptyMove()]
    results = filter_unneeded_boxes(top_results, h, w)
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
