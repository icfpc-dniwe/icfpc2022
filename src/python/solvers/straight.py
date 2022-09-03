import numpy as np
import typing as t
from ..types import RGBAImage, Box, Program
from ..moves import add_merge_move, add_color_move, add_point_cut_move


def render_straight(source_img: RGBAImage, boxes: t.Sequence[Box]) -> t.Tuple[RGBAImage, Program]:
    canvas = np.zeros_like(source_img)
    global_block_id = 0

    def render_box(cur_box: Box) -> np.ndarray:
        x_min, y_min, x_max, y_max = cur_box
        color = np.mean(source_img[y_min:y_max, x_min:x_max], axis=(0, 1)).astype(np.uint8)
        canvas[y_min:y_max, x_min:x_max] = color
        return color

    def get_program(cur_box: Box, color: np.ndarray) -> t.Tuple[int, t.List[str]]:
        x_min, y_min, x_max, y_max = cur_box
        cur_prog = [
            add_point_cut_move(f'{global_block_id}', (y_min, x_min)),
            add_merge_move(f'{global_block_id}.0', f'{global_block_id}.1'),
            add_merge_move(f'{global_block_id}.2', f'{global_block_id}.3'),
            add_merge_move(f'{global_block_id+1}', f'{global_block_id+2}')
        ]
        cur_global = global_block_id + 2
        cur_prog += [
            add_point_cut_move(f'{cur_global}', (y_max, x_max)),
            add_color_move(f'{cur_global}.3', color),
            add_merge_move(f'{cur_global}.0', f'{cur_global}.1'),
            add_merge_move(f'{cur_global}.2', f'{cur_global}.3'),
            add_merge_move(f'{cur_global + 1}', f'{cur_global + 2}')
        ]
        return 4, cur_prog

    whole_prog = []
    for cur_box in boxes:
        cur_color = render_box(cur_box)
        cur_addition, cur_prog = get_program(cur_box, cur_color)
        whole_prog += cur_prog
        global_block_id += cur_addition
    return canvas, whole_prog
