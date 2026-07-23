from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from training.common import atomic_npy, read_nonempty_lines, verify_sha256
from training.wsi import tissue_coordinates


def _tile(job: tuple[str, str, int, int, float, int]) -> tuple[str, str]:
    source, output, tile_size, downsample, min_fraction, max_tiles = job
    destination = Path(output)
    if destination.exists():
        return "SKIP", destination.stem
    try:
        coordinates = tissue_coordinates(
            Path(source), tile_size, downsample, min_fraction, max_tiles
        )
        atomic_npy(destination, coordinates)
        return "OK", destination.stem
    except Exception as error:
        return "ERROR", f"{destination.stem}: {error}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-sha256")
    parser.add_argument("--wsi-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--tile-size", type=int, required=True, choices=(256, 512))
    parser.add_argument("--downsample", type=int, default=32)
    parser.add_argument("--min-tissue-fraction", type=float, default=0.25)
    parser.add_argument("--max-tiles", type=int, default=2000)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--wsi-list", type=Path)
    args = parser.parse_args()
    verify_sha256(args.manifest, args.manifest_sha256)
    selected = read_nonempty_lines(args.wsi_list)
    rows = json.loads(args.manifest.read_text(encoding="utf-8"))
    if selected is not None:
        rows = [row for row in rows if row["wsi"] in selected]
    args.out.mkdir(parents=True, exist_ok=True)
    jobs = [
        (
            str(args.wsi_root / row["path"]),
            str(args.out / f"{row['wsi']}.npy"),
            args.tile_size,
            args.downsample,
            args.min_tissue_fraction,
            args.max_tiles,
        )
        for row in rows
    ]
    counts = {"OK": 0, "SKIP": 0, "ERROR": 0}
    errors = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for status, detail in executor.map(_tile, jobs):
            counts[status] += 1
            if status == "ERROR":
                errors.append(detail)
    print(f"FINISHED {counts}")
    if errors:
        raise RuntimeError("; ".join(errors[:8]))


if __name__ == "__main__":
    main()
