"""Build a labeled RGB pixel dataset from fire / no-fire example images."""

from __future__ import annotations

from pathlib import Path

import imageio.v3 as iio
import numpy as np
import pandas as pd

FIRE_LABEL = 1
NO_FIRE_LABEL = 2

COLUMNS = ["red", "green", "blue"]


def load_pixels(image_path: str | Path) -> np.ndarray:
    """Read an image and return its pixels as an (N, 3) RGB array."""
    pixels = iio.imread(image_path)
    return pixels[..., :3].reshape(-1, 3)


def labeled_pixels(image_path: str | Path, label: int) -> pd.DataFrame:
    """Return a DataFrame of an image's RGB pixels tagged with the given label."""
    pixels = load_pixels(image_path)
    df = pd.DataFrame(pixels, columns=COLUMNS)
    df["label"] = label
    return df


def build_dataset(fire_images: list[str | Path], no_fire_images: list[str | Path]) -> pd.DataFrame:
    """Combine pixels from fire and no-fire images into one labeled dataset."""
    frames = [labeled_pixels(path, FIRE_LABEL) for path in fire_images]
    frames += [labeled_pixels(path, NO_FIRE_LABEL) for path in no_fire_images]
    return pd.concat(frames, ignore_index=True)
