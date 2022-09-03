import numpy as np
import typing as t


Metric = t.Callable[[np.ndarray, np.ndarray], float]


def config_mosaic_metric(eps: float = 1e-8) -> Metric:

    def calc(left: np.ndarray, right: np.ndarray) -> float:
        left_idx = left[0:2]
        right_idx = right[0:2]
        left_color = left[2:6]
        right_color = right[2:6]
        return float(np.mean((left_color - right_color) ** 2 + eps) * np.max(np.abs(right_idx - left_idx)))

    return calc
