import numpy as np
import cv2
import random
from tqdm import tqdm
import typing as t
from itertools import product
from more_itertools import unzip

from .straight import render_straight
from ..mosaic.fill import expand_pixel
from ..scoring import block_similarity, image_similarity, static_cost, score_program_agaist_nothing
from ..box_utils import box_size, get_part
from ..image_utils import allgomerative_image
from ..moves import ColorMove, Move, EmptyMove
from ..types import RGBAImage, Box, Color, Program


def expand_cost_fn(
        source_img: RGBAImage,
        cur_img: RGBAImage,
        new_color: Color,
        box: Box,
        img_area: int,
        move_lambda: float = 5,
        verbose: bool = False
) -> t.Optional[float]:
    left_sim = block_similarity(get_part(source_img, box), new_color)
    right_sim = image_similarity(get_part(source_img, box), get_part(cur_img, box))
    move_cost = static_cost(ColorMove, box_size(box), img_area)
    total = left_sim - right_sim + move_lambda * move_cost
    # if left_sim > right_sim:
    #     return None
    # else:
    #     return left_sim - right_sim + move_lambda * static_cost(ColorMove, box_size(box), img_area)
    if verbose:
        print('left_sim:', left_sim)
        print('right_sim:', right_sim)
        print('move cost:', move_cost)
        print('lambda:', move_lambda)
        print('total:', total)
    return total


def mult_box(cur_box: Box, mult: float) -> Box:
    x1, y1, x2, y2 = cur_box
    bw = x2 - x1
    bh = y2 - y1
    return np.array([x1 * mult, y1 * mult, (x1 * mult) + bw * mult, (y1 * mult) + bh * mult], dtype=np.int64)


def get_boxes(
        img: RGBAImage,
        starting_points: t.List[t.Tuple[int, int]],
        tol_iter: int = 0,
        default_canvas: t.Optional[RGBAImage] = None,
        global_block_id: int = 0,
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
    if default_canvas is None:
        real_rendered_canvas = np.zeros_like(img) + 255
    else:
        real_rendered_canvas = default_canvas.copy()
    # print('Staring sim score:', image_similarity(img, real_rendered_canvas))
    boxes = []
    colors = []
    for point_idx, (cur_x, cur_y) in enumerate(tqdm(starting_points)):
        cur_color = img[cur_y, cur_x]

        def cur_scorring(box, verbose: bool = False):
            return expand_cost_fn(img, real_rendered_canvas, cur_color, box,
                                  real_canvas_area, move_lambda, verbose=verbose)
        new_box = expand_pixel((cur_x, cur_y), h, w, cur_scorring, tol_iter=tol_iter, return_best=return_best)
        if new_box is not None:
            score = cur_scorring(new_box, verbose=False)
            # print(cur_y, cur_x, '|> adding a new box?:', new_box, cur_color, score)
            if score < choose_thresholds[point_idx]:
                # print(cur_y, cur_x, '|> adding a new box:', new_box, cur_color)
                boxes.append(new_box)
                colors.append(cur_color)
                real_rendered_canvas[new_box[1]:new_box[3], new_box[0]:new_box[2]] = cur_color
                # print('New sim score:', image_similarity(img, real_rendered_canvas))
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
        default_canvas: t.Optional[RGBAImage] = None,
        global_block_id: int = 0
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
    all_scores = []
    for start_idx in range(num_random_starts):
        cur_num_points = np.random.randint(num_random_points, 2 * num_random_points)
        if np.random.rand() < 0.8:
            agglomerative_clusters = np.random.randint(10, 40)
            label_img, color_map = allgomerative_image(img, agglomerative_clusters)
            per_cluster = (cur_num_points - 10) // agglomerative_clusters
            num_rest = cur_num_points - per_cluster * agglomerative_clusters
            num_begin = num_rest // 2
            num_end = num_rest - num_begin
            begin_x = np.random.randint(0, w, size=num_begin)
            begin_y = np.random.randint(0, h, size=num_begin)
            end_x = np.random.randint(0, w, size=num_end)
            end_y = np.random.randint(0, h, size=num_end)
            agglomerative_points = []
            # aggl_keys = list(color_map.keys())
            # random.shuffle(aggl_keys)
            for cur_label in color_map.keys():
                points = list(zip(*np.where(label_img == cur_label)))
                agglomerative_points += random.choices(points, k=per_cluster)
            random.shuffle(agglomerative_points)
            starting_points = list(zip(begin_x, begin_y)) + agglomerative_points + list(zip(end_x, end_y))
        else:
            agglomerative_clusters = 0
            starting_x = np.random.randint(0, w, size=cur_num_points)
            starting_y = np.random.randint(0, h, size=cur_num_points)
            starting_points = list(zip(starting_x, starting_y))
        cur_tolerance = np.random.randint(0, 20)
        cur_return_best = np.random.rand() < 0.8
        if np.random.rand() < 0.5:
            max_threshold = np.exp(np.random.rand() * 7)
            min_threshold = -np.exp(np.random.rand() * 7)
            thresholds = np.linspace(min_threshold, max_threshold, num=cur_num_points)[::-1]
        else:
            thresholds = [0] * cur_num_points
        params.append((cur_tolerance, cur_return_best, np.max(thresholds), np.min(thresholds), cur_num_points, agglomerative_clusters))
        print('Hyperparameters:')
        print('Tolerance:', cur_tolerance)
        print('Return best:', cur_return_best)
        print('Max threshold:', np.max(thresholds))
        print('Min threshold:', np.min(thresholds))
        print('Num points:', cur_num_points)
        print('agglomerative_clusters:', agglomerative_clusters)
        results, score = get_boxes(img, starting_points,
                                   tol_iter=cur_tolerance,
                                   return_best=cur_return_best,
                                   choose_thresholds=thresholds,
                                   default_canvas=default_canvas
                                   )
        if len(results) > 0:
            results = prepare_boxes(results, h, w)
            boxes, colors = unzip(results)
            new_canvas, moves = render_straight(img, list(boxes), list(colors),
                                                default_canvas=default_canvas, global_block_id=global_block_id)
            _, score = score_program_agaist_nothing(img, default_canvas, new_canvas, moves)
            print('New points score:', score)
        all_scores.append(score)
        if score < top_score:
            top_score = score
            top_results = results
            top_idx = start_idx
        # with open('cur_top_results.txt', 'w') as f:
        #     indices = np.argsort(all_scores)
        #     for cur_idx in indices:
        #         print('-' * 10, file=f)
        #         print('Score:', all_scores[cur_idx], file=f)
        #         print('Tolerance:', params[cur_idx][0], file=f)
        #         print('Return best:', params[cur_idx][1], file=f)
        #         print('Max threshold:', params[cur_idx][2], file=f)
        #         print('Min threshold:', params[cur_idx][3], file=f)
        #         print('Num points:', params[cur_idx][4], file=f)
        #         print('agglomerative_clusters:', params[cur_idx][5], file=f)
    print('Top Score:', top_score)
    print('Num boxes:', len(top_results))
    print('Top hyperparameters:')
    print('Tolerance:', params[top_idx][0])
    print('Return best:', params[top_idx][1])
    print('Max threshold:', params[top_idx][2])
    print('Min threshold:', params[top_idx][3])
    print('Num points:', params[top_idx][4])
    print('agglomerative_clusters:', params[top_idx][5])
    # with open('top_results.txt', 'w') as f:
    #     indices = np.argsort(all_scores)
    #     for cur_idx in indices:
    #         print('-' * 10, file=f)
    #         print('Score:', all_scores[cur_idx], file=f)
    #         print('Tolerance:', params[cur_idx][0], file=f)
    #         print('Return best:', params[cur_idx][1], file=f)
    #         print('Max threshold:', params[cur_idx][2], file=f)
    #         print('Min threshold:', params[cur_idx][3], file=f)
    #         print('Num points:', params[cur_idx][4], file=f)
    #         print('agglomerative_clusters:', params[cur_idx][5], file=f)
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
    return render_straight(img, list(new_boxes), list(colors),
                           default_canvas=default_canvas, global_block_id=global_block_id)
    # return render_straight(img, new_boxes, list(colors))
