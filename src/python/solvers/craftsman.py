from pathlib import Path
from typing import List, Tuple

from python.image_utils import load_image
from python.moves import Move, get_program
from python.state import State
from python.types import Cost

problems_path = Path('../problems')


def problem_1() -> State:
    target_image = load_image(problems_path / f'{1}.png', revert=True)
    state = State(target_image)

    # __, __ = state.color(bid_l)
    # __, (bid_m,) = state.merge(bid_l, bid_r)
    # __, __ = state.color(bid_m)
    # __, (bl_bid, br_bid, tr_bid, tl_bid) = state.pcut(30, 150)
    # __, __ = state.color(bl_bid)
    # __, __ = state.color(br_bid)
    # __, __ = state.color(tr_bid)

    return state


if __name__ == '__main__':
    state = problem_1()
    print(f'State total cost: {state.total_cost()}')

    program = get_program(state.moves)
    with open('test_craftsman.txt', 'w') as f:
        print('\n'.join(program), file=f)
