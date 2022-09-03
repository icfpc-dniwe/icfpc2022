from pathlib import Path
from .image_utils import load_image, collect_boxes
from .types import Orientation
from .solvers.straight import render_straight
from .solvers.line_print import render_by_line
from .moves import get_program
from .box_utils import box_wh, box_size


def run():
    problems_path = Path('../problems')
    img = load_image(problems_path / '2.png', revert=True)
    # boxes = collect_boxes(img)
    # boxes = list(filter(lambda b: box_size(b) > 9 and min(box_wh(b)) > 2, boxes))
    # print(boxes)
    canvas, moves = render_by_line(img)
    prog = get_program(moves)
    with open('test_prog.txt', 'w') as f:
        print('\n'.join(prog), file=f)


if __name__ == '__main__':
    run()
    print(Orientation.X, Orientation.Y)
