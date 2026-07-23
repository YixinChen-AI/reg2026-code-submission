from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from training.common import atomic_npz, verify_sha256
from training.uni2h import encode_tiles, load_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tiles", type=Path, required=True)
    parser.add_argument("--tiles-sha256")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--weights-sha256")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--skip-existing", action=argparse.BooleanOptionalAction, default=True
    )
    args = parser.parse_args()
    verify_sha256(args.tiles, args.tiles_sha256)
    verify_sha256(args.weights, args.weights_sha256)
    if args.skip_existing and args.out.exists():
        print(f"SKIP {args.out}")
        return
    with np.load(args.tiles, allow_pickle=False) as data:
        tiles = data["tiles"]
        organs = data["organ"].astype(str)
        wsis = data["wsi"].astype(str)
        scales = data["scales"].astype(np.int16)
    model = load_model(args.weights, args.weights_sha256, args.device)
    features = encode_tiles(model, tiles, args.batch_size, args.device)
    atomic_npz(
        args.out,
        feats=features,
        organ=organs,
        wsi=wsis,
        scales=scales,
    )
    print(f"WROTE {args.out}: features={features.shape}")


if __name__ == "__main__":
    main()
