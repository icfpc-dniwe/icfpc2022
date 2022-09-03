import cv2
import numpy as np
from pathlib import Path
import typing as t
from PIL import Image as PILImage
from sklearn.cluster import OPTICS
from sklearn.feature_extraction.image import grid_to_graph
from sklearn.cluster import AgglomerativeClustering
import logging

from .types import RGBAImage, LabelImage, Box, Color
from .metrics import config_mosaic_metric


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
    r_img = cv2.resize(img, (50, 50), interpolation=cv2.INTER_AREA)
    flat_img = r_img.reshape((-1, 4)).astype(np.float64) / 255.
    cluster = OPTICS(min_samples=20, n_jobs=12)
    cluster.fit(flat_img)
    n_clusters = len(set(cluster.labels_)) - (1 if -1 in cluster.labels_ else 0)
    return max(1, n_clusters)


def deal_with_noise(
        img: RGBAImage,
        labeled_img: LabelImage,
        color_map: t.Dict[int, Color],
        tol: float = 1e-2
) -> t.Tuple[LabelImage, t.Dict[int, Color]]:
    f_img = img.astype(np.float64) / 255.
    new_map = labeled_img.copy()
    sorted_color_map = [(key, val) for key, val in color_map.items()]
    palette = np.array([el[1] for el in sorted_color_map]) / 255.
    for next_pixel in zip(*np.where(labeled_img == -1)):
        palette_dist = np.mean((f_img[next_pixel[0], next_pixel[1]] - palette) ** 2)
        if np.min(palette_dist) < tol:
            idx = np.argmin(palette_dist)
            new_label = sorted_color_map[idx][1]
            new_map[next_pixel[0], next_pixel[1]] = new_label
        else:
            new_color = 0
    return new_map


def get_color_map(img: RGBAImage, label_img: LabelImage) -> t.Dict[int, Color]:
    color_map = {}
    for cur_label in np.unique(label_img):
        color_map[cur_label] = np.mean(img[label_img == cur_label], axis=(0, 1))
    return color_map


def mosaic_image(img: RGBAImage, eps: float = 1e-6) -> t.Tuple[LabelImage, t.Dict[int, Color]]:
    r_img = cv2.resize(img, (50, 50), interpolation=cv2.INTER_AREA)
    metric = config_mosaic_metric(eps=eps)
    cluster = OPTICS(min_samples=20, metric=metric, n_jobs=12)
    indexed_img = np.array([(y, x, r_img[y, x, 0] / 255, r_img[y, x, 1] / 255, r_img[y, x, 2] / 255, r_img[y, x, 3] / 255)
                            for y in r_img.shape[0] for x in r_img.shape[1]], dtype=np.float64)
    cluster.fit(indexed_img)
    labels = cluster.labels_.reshape((r_img.shape[0], r_img.shape[1]))
    # new_labels = deal_with_noise(r_img, labels)
    color_map = get_color_map(r_img, labels)
    label_img = cv2.resize(labels, (img.shape[0], img.shape[1]), interpolation=cv2.INTER_NEAREST)
    return label_img, color_map


def allgomerative_image(
        img: RGBAImage,
        num_clusters: int,
        rescale_size: t.Optional[int] = None
) -> t.Tuple[LabelImage, t.Dict[int, Color]]:
    if rescale_size is not None:
        orig_shape = img.shape[:2]
        img = cv2.resize(img, (rescale_size, rescale_size), interpolation=cv2.INTER_AREA)
    connectivity = grid_to_graph(*img.shape[:2])
    ward = AgglomerativeClustering(
        n_clusters=num_clusters, linkage="ward", connectivity=connectivity
    )
    ward.fit(img.reshape((-1, 4)))
    label = np.reshape(ward.labels_, img.shape[:2])
    if rescale_size is not None:
        label = cv2.resize(label, orig_shape, interpolation=cv2.INTER_NEAREST)
    color_map = get_color_map(img, label)
    return label, color_map


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
