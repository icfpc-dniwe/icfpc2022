from pathlib import Path
from .image_utils import load_image, collect_boxes


def run():
    problems_path = Path('../problems')
    boxes = collect_boxes(load_image(problems_path / '1.png'))
    print(boxes)


if __name__ == '__main__':
    run()
