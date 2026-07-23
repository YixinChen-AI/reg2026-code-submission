from __future__ import annotations

from pathlib import Path

import numpy as np


def open_wsi(path: Path):
    import tifffile
    import zarr

    store = tifffile.imread(path, aszarr=True)
    array = zarr.open(store, mode="r")
    if not hasattr(array, "shape"):
        candidates = [
            array[key] for key in array.keys() if hasattr(array[key], "shape")
        ]
        array = max(candidates, key=lambda item: item.shape[0])
    return store, array


def sample_tissue_tiles(
    path: Path,
    tile_size: int,
    grid: int,
    limit: int,
    min_tissue_fraction: float,
) -> list[np.ndarray]:
    store, array = open_wsi(path)
    try:
        height, width = array.shape[:2]
        ys = np.linspace(0, max(0, height - tile_size), grid).astype(int)
        xs = np.linspace(0, max(0, width - tile_size), grid).astype(int)
        scored: list[tuple[float, int, np.ndarray]] = []
        index = 0
        for y in ys:
            for x in xs:
                patch = np.asarray(array[y : y + tile_size, x : x + tile_size])[..., :3]
                if patch.shape[:2] != (tile_size, tile_size):
                    index += 1
                    continue
                tissue = (~np.all(patch > 220, axis=2)) & (~np.all(patch < 25, axis=2))
                fraction = float(tissue.mean())
                if fraction >= min_tissue_fraction:
                    scored.append((fraction, index, patch))
                index += 1
    finally:
        store.close()
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [patch for _, _, patch in scored[:limit]]


def otsu_threshold(gray: np.ndarray) -> int:
    histogram = np.histogram(gray, bins=256, range=(0, 256))[0].astype(np.float64)
    total = histogram.sum()
    if total == 0:
        return 128
    weights = np.cumsum(histogram)
    means = np.cumsum(histogram * np.arange(256))
    foreground_weights = total - weights
    with np.errstate(invalid="ignore", divide="ignore"):
        background = np.where(weights > 0, means / weights, 0.0)
        foreground = np.where(
            foreground_weights > 0, (means[-1] - means) / foreground_weights, 0.0
        )
        between = weights * foreground_weights * (background - foreground) ** 2
    between[(weights == 0) | (foreground_weights == 0)] = 0
    return int(np.argmax(between))


def tissue_coordinates(
    path: Path,
    tile_size: int,
    downsample: int,
    min_tissue_fraction: float,
    max_tiles: int,
) -> np.ndarray:
    store, array = open_wsi(path)
    try:
        height, width = array.shape[:2]
        rows = []
        for start in range(0, height, 2048):
            block = np.asarray(array[start : min(height, start + 2048), :width])
            rows.append(block[::downsample, ::downsample, :3])
        thumbnail = np.concatenate(rows, axis=0)
    finally:
        store.close()
    gray = thumbnail.mean(axis=2).astype(np.uint8)
    threshold = otsu_threshold(gray)
    tissue = (gray < threshold) & (gray > 8)
    thumbnail_tile = max(1, tile_size // downsample)
    coordinates = []
    for y in range(0, height - tile_size + 1, tile_size):
        row = y // downsample
        for x in range(0, width - tile_size + 1, tile_size):
            column = x // downsample
            patch = tissue[
                row : row + thumbnail_tile,
                column : column + thumbnail_tile,
            ]
            if patch.size and float(patch.mean()) >= min_tissue_fraction:
                coordinates.append((x, y))
    result = np.asarray(coordinates, dtype=np.int32).reshape(-1, 2)
    if max_tiles and len(result) > max_tiles:
        indices = np.linspace(0, len(result) - 1, max_tiles).astype(int)
        result = result[indices]
    return result
