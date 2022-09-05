import numpy as np
from copy import copy
import typing as t

from ..block import Block, create_canvas
from ..moves import ColorMove, Merge, Move
from ..types import RGBAImage, Box, Color
from ..box_utils import get_part, box_size, can_merge
from ..scoring import image_similarity, block_similarity, static_cost, score_program_agaist_nothing
from . import color_blocks


def blocks_are_neightbors(left_block: Block, right_block: Block) -> bool:
    left_box = left_block.box
    right_box = right_block.box
    return can_merge(left_box, right_box)


def find_neighbors(blocks: t.Sequence[Block]) -> t.Dict[int, t.Tuple[int, int, int, int]]:
    """
    :param blocks: sequence of Block
    :return: a dictionary block_idx -> (top_neighbor_idx, right_neighbor_idx, bottom_neighbor_idx, left_neighbor_idx)
    """
    neighbors_dict = {cur_idx: [-1, -1, -1, -1] for cur_idx in range(len(blocks))}
    # 1 -- top; 2 -- right; 3 -- bottom; 4 -- left
    for left_idx, left_block in enumerate(blocks[:-1]):
        for right_idx, right_block in zip(range(left_idx+1, len(blocks)), blocks[left_idx+1:]):
            # if left_idx == right_idx:
            #     continue
            if blocks_are_neightbors(left_block, right_block):
                left_x_min, left_y_min, left_x_max, left_y_max = left_block.box
                right_x_min, right_y_min, right_x_max, right_y_max = right_block.box
                if left_x_min == right_x_min:
                    if left_y_min < right_y_min:
                        neighbors_dict[left_idx][2] = right_idx
                        neighbors_dict[right_idx][0] = left_idx
                    else:
                        neighbors_dict[left_idx][0] = right_idx
                        neighbors_dict[right_idx][2] = left_idx
                else:
                    if left_x_min < right_x_min:
                        neighbors_dict[left_idx][1] = right_idx
                        neighbors_dict[right_idx][3] = left_idx
                    else:
                        neighbors_dict[left_idx][3] = right_idx
                        neighbors_dict[right_idx][1] = left_idx
    return neighbors_dict


def find_top_left_block(blocks: t.Sequence[Block]) -> int:
    x_min = np.inf
    y_min = np.inf
    found_idx = -1
    for cur_idx, cur_block in enumerate(blocks):
        cur_x_min, cur_y_min, _, _ = cur_block.box
        if cur_x_min < x_min or cur_y_min < y_min:
            x_min = cur_x_min
            y_min = cur_y_min
            found_idx = cur_idx
    return found_idx


def merge_program(
        img: RGBAImage,
        blocks: t.Sequence[Block],
        global_block_id: t.Optional[int] = None
) -> t.Tuple[t.List[Block], t.List[Move], int]:
    neighbors_map = find_neighbors(blocks)
    new_blocks = []
    moves = []
    if global_block_id is None:
        global_block_id = np.max([int(cur_block.block_id) for cur_block in blocks])
    old_blocks = {cur_idx: cur_block for cur_idx, cur_block in enumerate(blocks)}
    while len(old_blocks) > 0:
        keys = list(old_blocks.keys())
        found_idx = find_top_left_block([old_blocks[cur_idx] for cur_idx in keys])
        bottom_left_block = keys[found_idx]
        right_neighbor = neighbors_map[bottom_left_block][1]
        bottom_neighbor = neighbors_map[bottom_left_block][2]
        if right_neighbor >= 0:
            left_box = blocks[bottom_left_block].box
            right_box = blocks[right_neighbor].box
            merged_box = (left_box[0], left_box[1], right_box[2], right_box[3])
            moves.append(Merge(blocks[bottom_left_block].block_id, blocks[right_neighbor].block_id,
                               max(box_size(blocks[bottom_left_block].box), box_size(blocks[right_neighbor].box))))
            global_block_id += 1
            # new_blocks.append(Block(f'{global_block_id}', merged_box, get_part(img, merged_box)))
            del old_blocks[bottom_left_block]
            del old_blocks[right_neighbor]
            if bottom_neighbor >= 0:
                n_right_neighbor = neighbors_map[bottom_neighbor][1]
                left_box = blocks[bottom_neighbor].box
                right_box = blocks[n_right_neighbor].box
                bottom_merged_box = (left_box[0], left_box[1], right_box[2], right_box[3])
                moves.append(Merge(blocks[bottom_neighbor].block_id, blocks[n_right_neighbor].block_id,
                                   max(box_size(blocks[bottom_neighbor].box), box_size(blocks[n_right_neighbor].box))))
                global_block_id += 1
                # new_blocks.append(Block(f'{global_block_id}', bottom_merged_box, get_part(img, bottom_merged_box)))

                n_merged_box = (merged_box[0], merged_box[1], bottom_merged_box[2], bottom_merged_box[3])
                moves.append(Merge(f'{global_block_id-1}', f'{global_block_id}',
                                   max(box_size(merged_box), box_size(bottom_merged_box))))
                global_block_id += 1
                new_blocks.append(Block(f'{global_block_id}', n_merged_box, get_part(img, n_merged_box)))
                del old_blocks[bottom_neighbor]
                del old_blocks[n_right_neighbor]
            else:
                new_blocks.append(Block(f'{global_block_id}', merged_box, get_part(img, merged_box)))
        else:
            if bottom_neighbor >= 0:
                top_box = blocks[bottom_left_block].box
                bottom_box = blocks[bottom_neighbor].box
                merged_box = (top_box[0], top_box[1], bottom_box[2], bottom_box[3])
                moves.append(Merge(blocks[bottom_left_block].block_id, blocks[bottom_neighbor].block_id,
                                   max(box_size(blocks[bottom_left_block].box), box_size(blocks[bottom_neighbor].box))))
                global_block_id += 1
                new_blocks.append(Block(f'{global_block_id}', merged_box, get_part(img, merged_box)))
                del old_blocks[bottom_left_block]
                del old_blocks[bottom_neighbor]
    return new_blocks, moves, global_block_id


def produce_program(
        img: RGBAImage,
        blocks: t.Sequence[Block]
) -> t.Tuple[RGBAImage, t.List[Move]]:
    default_canvas = create_canvas(blocks, *img.shape[:2])
    cur_canvas = default_canvas.copy()
    old_cost = np.inf
    new_cost = image_similarity(img, default_canvas)
    moves = []
    cur_blocks = blocks
    cur_global_id = np.max([int(cur_block.block_id) for cur_block in blocks])
    while len(cur_blocks) > 9:
        # try recoloring
        recolor_canvas, recolor_blocks, recolor_moves = color_blocks.produce_program(img, cur_blocks, cur_canvas)
        _, recolor_cost = score_program_agaist_nothing(img, default_canvas, recolor_canvas, moves + recolor_moves)
        # try merging, then recoloring
        new_blocks, merge_moves, new_global_id = merge_program(img, cur_blocks, cur_global_id)
        # merge_color_canvas, merge_color_moves = color_blocks.produce_program(img, new_blocks, cur_canvas)
        _, merge_cost = score_program_agaist_nothing(img, default_canvas, cur_canvas,
                                                     moves + merge_moves)
        if len(recolor_moves) > 0:
            old_cost = new_cost
            new_cost = recolor_cost
            # if new_cost > old_cost:
            #     break
            cur_canvas = recolor_canvas
            moves += recolor_moves
            cur_blocks = recolor_blocks
            print('Recoloring')
        else:
            # new_cost = merge_cost
            # cur_canvas = merge_color_canvas
            moves += merge_moves
            cur_blocks = new_blocks
            cur_global_id = new_global_id
            print('Merging')
        print('New cost:', new_cost, 'Old cost:', old_cost)

    return cur_canvas, moves
