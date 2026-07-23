from __future__ import annotations

import argparse
import json
from pathlib import Path

from training.common import ORGANS, atomic_torch_save, verify_sha256


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--config-sha256")
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--skip-existing", action=argparse.BooleanOptionalAction, default=True
    )
    args = parser.parse_args()
    verify_sha256(args.config, args.config_sha256)
    if args.skip_existing and args.out.exists():
        print(f"SKIP {args.out}")
        return

    import torch

    config = json.loads(args.config.read_text(encoding="utf-8"))
    heads = config["heads"]
    if config.get("head_count") != 203 or len(heads) != 203:
        raise ValueError("diagnosis config must contain exactly 203 heads")
    bundle = {"organs": {organ: {"e256": [], "e512": []} for organ in ORGANS}}
    seen = set()
    for head in heads:
        checkpoint_path = args.work_root / head["output"]
        if not checkpoint_path.is_file():
            raise FileNotFoundError(checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if int(checkpoint.get("K", -1)) != 1:
            raise ValueError(
                f"{checkpoint_path} has K={checkpoint.get('K')}, expected 1"
            )
        identity = (
            checkpoint.get("organ"),
            checkpoint.get("scale"),
            checkpoint.get("seed"),
        )
        expected = (head["organ"], int(head["scale"]), int(head["seed"]))
        if identity != expected:
            raise ValueError(
                f"{checkpoint_path} identity {identity} does not match {expected}"
            )
        if not checkpoint.get("vocab") or not isinstance(checkpoint.get("state"), dict):
            raise ValueError(f"invalid checkpoint payload: {checkpoint_path}")
        if head["id"] in seen:
            raise ValueError(f"duplicate head: {head['id']}")
        seen.add(head["id"])
        tag = f"e{head['scale']}"
        bundle["organs"][head["organ"]][tag].append(
            {
                "state": checkpoint["state"],
                "vocab": checkpoint["vocab"],
                "K": 1,
            }
        )
    for organ in ORGANS:
        if len(bundle["organs"][organ]["e256"]) != 20:
            raise ValueError(f"{organ} does not have 20 scale-256 heads")
        if len(bundle["organs"][organ]["e512"]) != 9:
            raise ValueError(f"{organ} does not have 9 scale-512 heads")
    bundle["release"] = "0.6.0"
    bundle["head_count"] = 203
    atomic_torch_save(args.out, bundle)
    print(f"WROTE {args.out}: 203 K=1 diagnosis heads")


if __name__ == "__main__":
    main()
