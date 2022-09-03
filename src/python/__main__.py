import click
from pathlib import Path
from matplotlib import pyplot as plt
from .image_utils import load_image, collect_boxes, allgomerative_image, get_colored_img
from .types import Orientation
from .solvers.straight import render_straight
from .solvers.expading import produce_program
from .solvers.line_print import render_by_line
from .moves import get_program
from .box_utils import box_wh, box_size


def test_run():
    problems_path = Path('../problems')
    img = load_image(problems_path / '2.png', revert=True)
    # boxes = collect_boxes(img)
    # boxes = list(filter(lambda b: box_size(b) > 9 and min(box_wh(b)) > 2, boxes))
    # print(boxes)
    # label_img, color_map = allgomerative_image(img, 20)
    # print(color_map)
    # plt.imshow(label_img)
    # plt.savefig('label.pdf')
    # img = get_colored_img(label_img, color_map)
    # plt.imshow(img)
    # plt.savefig('recolored.pdf')
    # canvas, moves = render_by_line(img)
    # plt.imshow(canvas)
    # plt.savefig('canvas.pdf')
    # prog = get_program(moves)
    canvas, prog = produce_program(img)
    plt.imshow(canvas)
    plt.savefig('canvas.pdf')
    with open('test_prog.txt', 'w') as f:
        print('\n'.join(prog), file=f)


@click.command()
@click.option('-p', '--problem-path', type=Path)
@click.option('-o', '--output-path', type=Path)
def main_run(problem_path: Path, output_path: Path):
    img = load_image(problem_path, revert=True)
    canvas, prog = produce_program(img)
    with output_path.open('w') as f:
        print('\n'.join(prog), file=f)


if __name__ == '__main__':
    main_run()
