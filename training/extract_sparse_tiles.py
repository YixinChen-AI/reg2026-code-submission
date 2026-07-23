from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

from training.common import atomic_npz, ordered_files, verify_sha256
from training.wsi import sample_tissue_tiles


def _extract(
    job: tuple[dict[str, str], str, str, tuple[int, ...], int, int, int, float],
) -> tuple[str, str]:
    row, root, cache, scales, tiles_per_scale, grid, output_size, min_fraction = job
    output = Path(cache) / f"{row['wsi']}.npz"
    if output.exists():
        return "SKIP", row["wsi"]
    try:
        source = Path(root) / row["path"]
        tiles = []
        tile_scales = []
        for scale in scales:
            selected = sample_tissue_tiles(
                source, scale, grid, tiles_per_scale, min_fraction
            )
            if len(selected) < 4:
                selected = sample_tissue_tiles(
                    source, scale, grid + 10, tiles_per_scale, min_fraction
                )
            for tile in selected:
                tiles.append(
                    np.asarray(
                        Image.fromarray(tile).resize(
                            (output_size, output_size), Image.Resampling.BILINEAR
                        ),
                        dtype=np.uint8,
                    )
                )
                tile_scales.append(scale)
        if not tiles:
            return "EMPTY", row["wsi"]
        atomic_npz(
            output,
            tiles=np.stack(tiles),
            scales=np.asarray(tile_scales, dtype=np.int16),
            organ=np.asarray(row["organ"]),
            wsi=np.asarray(row["wsi"]),
        )
        return "OK", row["wsi"]
    except Exception as error:
        return "ERROR", f"{row['wsi']}: {error}"


def merge(cache: Path, output: Path, skip_existing: bool) -> None:
    if skip_existing and output.exists():
        print(f"SKIP {output}")
        return
    tiles: list[np.ndarray] = []
    scales: list[int] = []
    organs: list[str] = []
    wsis: list[str] = []
    for shard in ordered_files(cache, ".npz"):
        with np.load(shard, allow_pickle=False) as data:
            count = len(data["tiles"])
            tiles.extend(data["tiles"])
            scales.extend(data["scales"].astype(int).tolist())
            organs.extend([str(data["organ"])] * count)
            wsis.extend([str(data["wsi"])] * count)
    if not tiles:
        raise RuntimeError(f"no tile shards in {cache}")
    atomic_npz(
        output,
        tiles=np.stack(tiles),
        scales=np.asarray(scales, dtype=np.int16),
        organ=np.asarray(organs),
        wsi=np.asarray(wsis),
    )
    print(f"WROTE {output}: {len(tiles)} tiles from {len(set(wsis))} WSIs")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--manifest-sha256")
    parser.add_argument("--wsi-root", type=Path)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--merge", action="store_true")
    parser.add_argument("--scales", default="256,512")
    parser.add_argument("--tiles-per-scale", type=int, default=16)
    parser.add_argument("--grid", type=int, default=12)
    parser.add_argument("--output-size", type=int, default=224)
    parser.add_argument("--min-tissue-fraction", type=float, default=0.30)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--skip-existing", action=argparse.BooleanOptionalAction, default=True
    )
    args = parser.parse_args()
    if args.merge:
        if args.out is None:
            parser.error("--out is required with --merge")
        merge(args.cache, args.out, args.skip_existing)
        return
    if args.manifest is None or args.wsi_root is None:
        parser.error("--manifest and --wsi-root are required for extraction")
    verify_sha256(args.manifest, args.manifest_sha256)
    rows = json.loads(args.manifest.read_text(encoding="utf-8"))
    args.cache.mkdir(parents=True, exist_ok=True)
    scales = tuple(int(value) for value in args.scales.split(","))
    jobs = [
        (
            row,
            str(args.wsi_root),
            str(args.cache),
            scales,
            args.tiles_per_scale,
            args.grid,
            args.output_size,
            args.min_tissue_fraction,
        )
        for row in rows
    ]
    counts = {"OK": 0, "SKIP": 0, "EMPTY": 0, "ERROR": 0}
    errors = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for status, detail in executor.map(_extract, jobs):
            counts[status] += 1
            if status == "ERROR":
                errors.append(detail)
    print(f"FINISHED {counts}")
    if errors:
        raise RuntimeError("; ".join(errors[:8]))


if __name__ == "__main__":
    main()
