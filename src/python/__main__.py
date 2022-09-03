from pathlib import Path
from .image_utils import load_image, collect_boxes
from .types import Orientation
from .solvers.straight import render_straight


def run():
    problems_path = Path('../problems')
    img = load_image(problems_path / '1.png')
    boxes = collect_boxes(img)
    print(boxes)
    canvas, prog = render_straight(img, boxes)
    with open('test_prog.txt', 'w') as f:
        print('\n'.join(prog), file=f)


if __name__ == '__main__':
    run()
    print(Orientation.X, Orientation.Y)
