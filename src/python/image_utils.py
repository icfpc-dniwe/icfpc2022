import cv2
import numpy as np
from pathlib import Path
import typing as t
import logging

from .types import RGBAImage, LabelImage, Box


def cluster_colors(img: RGBAImage) -> LabelImage:
    raise NotImplementedError


def collect_boxes(img: RGBAImage) -> t.Sequence[Box]:
    src_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    src_gray = cv2.medianBlur(src_gray, 7)

    threshold = 20

    canny_output = cv2.Canny(src_gray, threshold, threshold * 2)
    # Find contours
    contours, hierarchy = cv2.findContours(canny_output, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    new_cnts = []
    for i in range(len(contours)):
        eps = 0.1 * cv2.arcLength(contours[i], True)
        app_cnt = cv2.approxPolyDP(contours[i], eps, True)
        new_cnts.append(app_cnt)
    boxes = [cv2.boundingRect(cnt) for cnt in new_cnts]
    return [np.array((x, y, x + w, y + h)) for (x, y, w, h) in boxes]


def load_image(img_path: Path) -> t.Optional[RGBAImage]:
    img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
    if img is None:
        logging.error(f'Cannot read image `{img_path}`')
        return None
    if len(img.shape) < 3:
        return np.tile(img[:, :, np.newaxis], (1, 1, 4))
    elif img.shape[2] == 3:
        new_img = np.zeros((*img.shape[:2], 4), dtype=np.uint8)
        new_img[:, :, :4] = img
        return new_img
    else:
        return img


def get_palette(img: RGBAImage) -> t.Tuple[LabelImage, t.Dict[int, Color]]:
    pass
