from pathlib import Path

import cv2
import numpy as np
import typing as t

from ..box_utils import box_size
from ..image_utils import load_image
from ..types import RGBAImage, Box, Program, Color, Orientation
from ..moves import Move, PointCut, LineCut, Merge, ColorMove


# TODO box with 0 coordinate does not render
def render_straight(
        source_img: RGBAImage,
        boxes: t.Sequence[Box],
        colors: t.Optional[t.Sequence[Color]] = None,
        canvas: t.Optional[RGBAImage] = None,
        global_block_id: int = 0,
        block_prefix: t.Optional[str] = None
) -> t.Tuple[RGBAImage, t.List[Move]]:
    if canvas is None:
        canvas = np.zeros_like(source_img) + 255
    h, w = canvas.shape[:2]
    img_size = h * w
    # global_block_id = 0
    if block_prefix is None:
        block_prefix = f'{global_block_id}'

    def render_box(cur_box: Box, color: t.Optional[Color] = None) -> np.ndarray:
        x_min, y_min, x_max, y_max = cur_box
        # assert x_min < x_max - 1
        # assert y_min < y_max - 1
        if color is None:
            color = np.mean(source_img[y_min:y_max, x_min:x_max], axis=(0, 1)).astype(np.uint8)
        canvas[y_min:y_max, x_min:x_max] = color
        return color

    def get_program_pcut(cur_box: Box, color: np.ndarray, cur_prefix: str) -> t.List[Move]:
        x_min, y_min, x_max, y_max = cur_box
        return [
            PointCut(f'{cur_prefix}', (x_min, y_min), block_size=img_size),
            PointCut(f'{cur_prefix}.2', (x_max, y_max), block_size=box_size((x_min, y_min, w, h))),
            ColorMove(f'{cur_prefix}.2.0', color, block_size=box_size(cur_box))
        ]

    def merge_back_pcut(cur_prefix: str, cur_box: Box) -> t.List[Move]:
        x_min, y_min, x_max, y_max = cur_box
        return [
            Merge(f'{cur_prefix}.2.2', f'{cur_prefix}.2.3',
                  block_size=max(box_size((x_max, y_max, w, h)), box_size((x_min, y_max, x_max, h)))),
            Merge(f'{cur_prefix}.2.0', f'{cur_prefix}.2.1',
                  block_size=max(box_size(cur_box), box_size((x_max, y_min, w, y_max)))),
            Merge(f'{global_block_id + 1}', f'{global_block_id + 2}',
                  block_size=max(box_size((x_min, y_min, w, y_max)), box_size((x_min, y_max, w, h)))),

            Merge(f'{cur_prefix}.0', f'{cur_prefix}.1',
                  block_size=max(box_size((0, 0, x_min, y_min)), box_size((x_min, 0, w, y_min)))),
            Merge(f'{cur_prefix}.3', f'{global_block_id + 3}',
                  block_size=max(box_size((0, y_min, x_min, h)), box_size((x_min, y_min, w, h)))),
            Merge(f'{global_block_id + 4}', f'{global_block_id + 5}',
                  block_size=max(box_size((0, 0, w, y_min)), box_size((0, y_min, w, y_max)))),
        ]

    def get_program_corner_cut(prefix: str, point: t.Tuple[int, int], block_postfix: int, color: Color) -> t.List[Move]:
        return [
            PointCut(f'{prefix}', point),
            ColorMove(f'{prefix}.{block_postfix}', color)
        ]

    def merge_corner_cut(block_id: int) -> t.List[Move]:
        return [
            Merge(f'{block_id}.0', f'{block_id}.1'),
            Merge(f'{block_id}.2', f'{block_id}.3'),
            Merge(f'{block_id + 1}', f'{block_id + 2}')
        ]

    def get_program_linear_cut(cur_box: Box, color: np.ndarray, last_box: bool = False) -> t.Tuple[Program, int]:
        x_min, y_min, x_max, y_max = cur_box
        moves = []
        global_addition = 0
        if x_min == 0:
            if y_min == 0:
                moves += get_program_corner_cut(f'{global_block_id}', (x_max, y_max), 0, color)
                if not last_box:
                    moves += merge_corner_cut(global_block_id)
                    global_addition = 3
            elif y_max == h:
                moves += get_program_corner_cut(f'{global_block_id}', (x_max, y_min), 3, color)
                if not last_box:
                    moves += merge_corner_cut(global_block_id)
                    global_addition = 3
            else:
                raise NotImplementedError
        elif x_max == w:
            if y_min == 0:
                moves += get_program_corner_cut(f'{global_block_id}', (x_min, y_max), 1, color)
                if not last_box:
                    moves += merge_corner_cut(global_block_id)
                    global_addition = 3
            elif y_max == h:
                moves += get_program_corner_cut(f'{global_block_id}', (x_min, y_min), 2, color)
                if not last_box:
                    moves += merge_corner_cut(global_block_id)
                    global_addition = 3
            else:
                raise NotImplementedError
        elif y_min == 0:
            raise NotImplementedError
        elif y_max == h:
            raise NotImplementedError
        else:
            raise RuntimeError
        return moves, global_addition

    def use_point_cut(cur_box: Box) -> bool:
        x_min, y_min, x_max, y_max = cur_box
        return x_min > 0 and y_min > 0 and x_max < w and y_max < h

    whole_prog = []
    for i, cur_box in enumerate(boxes):
        if colors is None:
            cur_color = render_box(cur_box)
        else:
            cur_color = render_box(cur_box, colors[i])
        if use_point_cut(cur_box):
            cur_prog = get_program_pcut(cur_box, cur_color, block_prefix)
            whole_prog += cur_prog

            if i < len(boxes)-1:
                whole_prog += merge_back_pcut(block_prefix, cur_box)
                global_block_id += 6
                block_prefix = f'{global_block_id}'
        else:
            raise NotImplementedError
            cur_prog, addition = get_program_linear_cut(cur_box, cur_color, last_box=i < len(boxes)-1)
            whole_prog += cur_prog
            global_block_id += addition

    return canvas, whole_prog


if __name__ == '__main__':
    problems_path = Path('../problems')
    img = load_image(problems_path / f'1.png')

    boxes = [
        np.array([1, 20, 30, 60]),
        np.array([70, 120, 80, 150]),
    ]

    # colors = [
    #     np.array([255, 0, 0, 255]),
    #     np.array([0, 255, 0, 255]),
    # ]

    canvas, prog = render_straight(img, boxes)

    with open('test_render_straight.txt', 'w') as f:
        print('\n'.join(prog), file=f)

    cv2.imshow('orig', img)
    cv2.imshow('result', canvas)

    cv2.waitKey(0)
