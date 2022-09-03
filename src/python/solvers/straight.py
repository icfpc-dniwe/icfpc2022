import numpy as np
import typing as t
from ..types import RGBAImage, Box, Program
from ..moves import add_merge_move, add_color_move, add_point_cut_move


def render_straight(source_img: RGBAImage, boxes: t.Sequence[Box]) -> t.Tuple[RGBAImage, Program]:
    canvas = np.zeros_like(source_img)
    global_block_id = 0

    def render_box(cur_box: Box) -> np.ndarray:
        x_min, y_min, x_max, y_max = cur_box
        assert x_min < x_max - 1
        assert y_min < y_max - 1
        color = np.mean(source_img[y_min:y_max, x_min:x_max], axis=(0, 1)).astype(np.uint8)
        canvas[y_min:y_max, x_min:x_max] = color
        return color

    def get_program_pcut(cur_box: Box, color: np.ndarray) -> t.List[str]:
        x_min, y_min, x_max, y_max = cur_box
        cur_prog = [
            add_point_cut_move(f'{global_block_id}', (y_min, x_min)),
            add_point_cut_move(f'{global_block_id}.2', (y_max, x_max)),
        ]
        cur_prog += [
            add_color_move(f'{global_block_id}.2.0', color)
        ]
        return cur_prog

    def merge_back_pcut() -> t.List[str]:
        return [
            add_merge_move(f'{global_block_id}.2.0', f'{global_block_id}.2.1'),
            add_merge_move(f'{global_block_id}.2.2', f'{global_block_id}.2.3'),
            add_merge_move(f'{global_block_id + 1}', f'{global_block_id + 2}'),
            add_merge_move(f'{global_block_id}.0', f'{global_block_id}.1'),
            add_merge_move(f'{global_block_id}.3', f'{global_block_id + 3}'),
            add_merge_move(f'{global_block_id + 4}', f'{global_block_id + 5}'),
        ]

    def use_point_cut(cur_box: Box) -> bool:
        x_min, y_min, x_max, y_max = cur_box
        return x_min > 0 and y_min > 0 and (x_max + 1) < source_img.shape[0]

    whole_prog = []
    for cur_box in boxes:
        if use_point_cut(cur_box):
            cur_color = render_box(cur_box)
            cur_prog = get_program_pcut(cur_box, cur_color)
            whole_prog += cur_prog
            whole_prog += merge_back_pcut()
            global_block_id += 6
        else:
            pass
    return canvas, whole_prog


def render_straight_copy(source_img: RGBAImage, boxes: t.Sequence[Box]) -> t.Tuple[RGBAImage, Program]:
    canvas = np.zeros_like(source_img)
    global_block_id = 0

    def render_box(cur_box: Box) -> np.ndarray:
        x_min, y_min, x_max, y_max = cur_box
        assert x_min < x_max - 1
        assert y_min < y_max - 1
        color = np.mean(source_img[y_min:y_max, x_min:x_max], axis=(0, 1)).astype(np.uint8)
        canvas[y_min:y_max, x_min:x_max] = color
        return color

    def get_program_pcut(cur_box: Box, color: np.ndarray) -> t.List[str]:
        x_min, y_min, x_max, y_max = cur_box
        cur_prog = [
            add_point_cut_move(f'{global_block_id}', (y_min, x_min)),
            add_point_cut_move(f'{global_block_id}.2', (y_max, x_max)),
        ]
        cur_prog += [
            add_color_move(f'{global_block_id}.2.0', color)
        ]
        return cur_prog

    def merge_back_pcut() -> t.List[str]:
        return [
            add_merge_move(f'{global_block_id}.2.0', f'{global_block_id}.2.1'),
            add_merge_move(f'{global_block_id}.2.2', f'{global_block_id}.2.3'),
            add_merge_move(f'{global_block_id + 1}', f'{global_block_id + 2}'),
            add_merge_move(f'{global_block_id}.0', f'{global_block_id}.1'),
            add_merge_move(f'{global_block_id}.3', f'{global_block_id + 3}'),
            add_merge_move(f'{global_block_id + 4}', f'{global_block_id + 5}'),
        ]

    def use_point_cut(cur_box: Box) -> bool:
        x_min, y_min, x_max, y_max = cur_box
        return x_min > 0 and y_min > 0 and (x_max + 1) < source_img.shape[0]

    whole_prog = []
    for cur_box in boxes:
        if use_point_cut(cur_box):
            cur_color = render_box(cur_box)
            cur_prog = get_program_pcut(cur_box, cur_color)
            whole_prog += cur_prog
            whole_prog += merge_back_pcut()
            global_block_id += 6
        else:
            pass
    return canvas, whole_prog
