from __future__ import annotations

import argparse
from pathlib import Path

from training.common import ORGANS, atomic_json


def generate_heads() -> list[dict[str, object]]:
    heads: list[dict[str, object]] = []
    for organ in ORGANS:
        for scale, count in ((256, 20), (512, 9)):
            for seed in range(1, count + 1):
                heads.append(
                    {
                        "id": f"{organ}-{scale}-s{seed:02d}",
                        "organ": organ,
                        "scale": scale,
                        "seed": seed,
                        "features": f"features/full_tissue_{scale}",
                        "output": f"checkpoints/dx/{organ}/{scale}/seed-{seed:02d}.pt",
                    }
                )
    if len(heads) != 203 or len({head["id"] for head in heads}) != 203:
        raise RuntimeError("diagnosis-head contract must contain 203 unique heads")
    return heads


def build_config() -> dict[str, object]:
    heads = generate_heads()
    return {
        "schema_version": 1,
        "release": "0.6.0",
        "model": {
            "architecture": "ACMIL_GA",
            "branches": 1,
            "n_masked": 0,
            "mask_drop": 0.0,
            "epochs": 40,
            "center": "ALL",
            "subsample": 1.0,
            "center_balance": False,
            "selection_set": "overlapping_first_10_percent_training_fallback",
        },
        "head_count": len(heads),
        "heads": heads,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    config = build_config()
    if args.check:
        if not args.out.exists():
            raise FileNotFoundError(args.out)
        import json

        current = json.loads(args.out.read_text(encoding="utf-8"))
        if current != config:
            raise SystemExit(f"{args.out} is not generated from the current contract")
        print(f"OK {args.out}: 203 heads")
        return
    atomic_json(args.out, config)
    print(f"WROTE {args.out}: 203 heads")


if __name__ == "__main__":
    main()
