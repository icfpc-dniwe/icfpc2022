from pathlib import Path
from .image_utils import load_image, collect_boxes
from .types import Orientation
from .solvers.straight import render_straight
from .box_utils import box_wh, box_size


def run():
    problems_path = Path('../problems')
    img = load_image(problems_path / '1.png')
    boxes = collect_boxes(img)
    boxes = list(filter(lambda b: box_size(b) > 9 and min(box_wh(b)) > 2, boxes))
    print(boxes)
    canvas, prog = render_straight(img, boxes)
    with open('test_prog.txt', 'w') as f:
        print('\n'.join(prog), file=f)


if __name__ == '__main__':
    run()
    print(Orientation.X, Orientation.Y)
