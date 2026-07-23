from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from training.common import verify_sha256


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--config-sha256")
    parser.add_argument("--cot", type=Path, required=True)
    parser.add_argument("--cot-sha256")
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--continue-on-error", action="store_true")
    args = parser.parse_args()
    if not 0 <= args.shard_index < args.shard_count:
        parser.error("--shard-index must be in [0, --shard-count)")
    verify_sha256(args.config, args.config_sha256)
    verify_sha256(args.cot, args.cot_sha256)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    heads = config["heads"][args.shard_index :: args.shard_count]
    failures = []
    for index, head in enumerate(heads, start=1):
        output = args.work_root / head["output"]
        if output.exists():
            print(f"[{index}/{len(heads)}] SKIP {head['id']}", flush=True)
            continue
        command = [
            sys.executable,
            "-m",
            "training.train_dx_acmil",
            "--config",
            str(args.config),
            "--head-id",
            head["id"],
            "--cot",
            str(args.cot),
            "--work-root",
            str(args.work_root),
            "--device",
            args.device,
        ]
        if args.config_sha256:
            command.extend(["--config-sha256", args.config_sha256])
        if args.cot_sha256:
            command.extend(["--cot-sha256", args.cot_sha256])
        print(f"[{index}/{len(heads)}] RUN {head['id']}", flush=True)
        result = subprocess.run(command, check=False)
        if result.returncode:
            failures.append(head["id"])
            if not args.continue_on_error:
                raise SystemExit(result.returncode)
    if failures:
        raise RuntimeError(f"failed diagnosis heads: {', '.join(failures)}")
    print(f"FINISHED {len(heads)} diagnosis heads")


if __name__ == "__main__":
    main()
