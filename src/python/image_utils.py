import cv2
import numpy as np
from pathlib import Path
import typing as t
from PIL import Image as PILImage
import logging

from .types import RGBAImage, LabelImage, Box, Color


def cluster_colors(img: RGBAImage) -> LabelImage:
    raise NotImplementedError


def collect_boxes(img: RGBAImage) -> t.Sequence[Box]:
    src_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
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
        new_img[:, :, :-1] = img[:, :, ::-1]
        return new_img
    else:
        rgba_img = np.zeros_like(img)
        rgba_img[:, :, :-1] = img[:, :, 2::-1]
        rgba_img[:, :, -1] = img[:, :, -1]
        return rgba_img


def determine_num_colors(img: RGBAImage) -> int:
    return 10


def get_palette(img: RGBAImage, num_color: t.Optional[int] = None) -> t.Tuple[LabelImage, t.Dict[int, Color]]:
    pil_img = PILImage.fromarray(img)
    if num_color is None:
        num_colors = determine_num_colors(img)
    paletted_img = pil_img.quantize(num_colors, method=PILImage.Quantize.LIBIMAGEQUANT)
    colors = np.array(paletted_img.getpalette(None)).reshape((-1, 4))[:10]
    label_map = np.asarray(paletted_img).astype(np.int32)
    lable_dict = {idx: color for idx, color in enumerate(label_map)}
    return colors, lable_dict


def get_area(img: RGBAImage) -> int:
    return img.shape[0] * img.shape[1]


def get_average_color(img: RGBAImage):
    return np.array([np.average(img[:, :, i]) for i in range(4)])
