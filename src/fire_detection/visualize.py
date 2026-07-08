"""Render an image with its predicted fire pixels highlighted in red."""

from __future__ import annotations

from pathlib import Path

import imageio.v3 as iio
import numpy as np
import pandas as pd
from PIL import Image

from fire_detection.dataset import COLUMNS, FIRE_LABEL

# A predicted fire pixel is only highlighted if it is bright enough and part of
# a run of at least two consecutive fire pixels, which filters out isolated
# false positives.
RED_THRESHOLD = 160
GREEN_THRESHOLD = 80


def highlight_fire(image_path: str | Path, predicted: pd.DataFrame, output_path: str | Path) -> np.ndarray:
    """Save `image_path` with confident fire pixels recolored red, return the array."""
    pic = iio.imread(image_path)[..., :3]
    height, width = pic.shape[:2]

    pixels = predicted[COLUMNS].to_numpy(dtype=np.uint8).copy()
    is_fire = predicted["label"].to_numpy() == FIRE_LABEL
    is_bright = (predicted["red"].to_numpy() > RED_THRESHOLD) & (predicted["green"].to_numpy() > GREEN_THRESHOLD)
    in_a_run = is_fire & np.roll(is_fire, -1)
    highlight = is_fire & is_bright & in_a_run

    pixels[highlight] = [255, 0, 0]
    rgb_array = pixels.reshape(height, width, 3)

    Image.fromarray(rgb_array, "RGB").save(output_path)
    return rgb_array
