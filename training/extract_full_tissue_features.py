from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from training.common import atomic_npy, read_nonempty_lines, verify_sha256
from training.uni2h import FEATURE_DIMENSION, encode_tiles, load_model
from training.wsi import open_wsi


def read_tiles(path: Path, coordinates: np.ndarray, tile_size: int) -> np.ndarray:
    store, array = open_wsi(path)
    tiles = []
    try:
        height, width = array.shape[:2]
        for x, y in coordinates:
            if y + tile_size > height or x + tile_size > width:
                continue
            patch = np.asarray(array[y : y + tile_size, x : x + tile_size])[..., :3]
            if patch.shape[:2] != (tile_size, tile_size):
                continue
            tiles.append(
                np.asarray(
                    Image.fromarray(patch).resize(
                        (224, 224), Image.Resampling.BILINEAR
                    ),
                    dtype=np.uint8,
                )
            )
    finally:
        store.close()
    if not tiles:
        return np.zeros((0, 224, 224, 3), dtype=np.uint8)
    return np.stack(tiles)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-sha256")
    parser.add_argument("--wsi-root", type=Path, required=True)
    parser.add_argument("--coords", type=Path, required=True)
    parser.add_argument("--tile-size", type=int, required=True, choices=(256, 512))
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--weights-sha256")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--wsi-list", type=Path)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    args = parser.parse_args()
    if not 0 <= args.shard_index < args.shard_count:
        parser.error("--shard-index must be in [0, --shard-count)")
    verify_sha256(args.manifest, args.manifest_sha256)
    verify_sha256(args.weights, args.weights_sha256)
    rows = json.loads(args.manifest.read_text(encoding="utf-8"))
    selected = read_nonempty_lines(args.wsi_list)
    if selected is not None:
        rows = [row for row in rows if row["wsi"] in selected]
    rows = rows[args.shard_index :: args.shard_count]
    args.out.mkdir(parents=True, exist_ok=True)
    model = load_model(args.weights, args.weights_sha256, args.device)
    completed = skipped = 0
    for row in rows:
        output = args.out / f"{row['wsi']}.npy"
        if output.exists():
            skipped += 1
            continue
        coordinate_path = args.coords / f"{row['wsi']}.npy"
        if not coordinate_path.exists():
            continue
        coordinates = np.load(coordinate_path, allow_pickle=False)
        tiles = read_tiles(args.wsi_root / row["path"], coordinates, args.tile_size)
        features = encode_tiles(model, tiles, args.batch_size, args.device).astype(
            np.float16
        )
        if features.shape[1:] != (FEATURE_DIMENSION,):
            raise RuntimeError(
                f"invalid feature shape for {row['wsi']}: {features.shape}"
            )
        atomic_npy(output, features)
        completed += 1
    print(f"FINISHED completed={completed} skipped={skipped} selected={len(rows)}")


if __name__ == "__main__":
    main()
