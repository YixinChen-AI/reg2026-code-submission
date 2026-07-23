from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path

from training.common import atomic_json, atomic_text, sha256_file, verify_sha256

ARTIFACTS = {
    "backbone": Path("model/uni2h/pytorch_model.bin"),
    "organ": Path("model/organ_uni2h_ms_ensemble.pt"),
    "diagnosis": Path("model/organ_dx_ensemble.pt"),
    "medoids": Path("model/slot_medoids.json"),
    "retrieval_bank": Path("model/exemplar_bank.npz"),
    "retrieval_cots": Path("model/exemplar_cots.json"),
}


def copy_atomic(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
    os.close(fd)
    temporary = Path(name)
    try:
        shutil.copyfile(source, temporary)
        os.replace(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def verify(root: Path) -> dict:
    manifest_path = root / "artifact-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("release") != "0.6.0":
        raise ValueError("artifact manifest release is not 0.6.0")
    entries = manifest.get("files", [])
    expected_paths = sorted(path.as_posix() for path in ARTIFACTS.values())
    actual_paths = sorted(entry.get("path") for entry in entries)
    if actual_paths != expected_paths:
        raise ValueError(
            f"artifact paths differ: expected {expected_paths}, got {actual_paths}"
        )
    for entry in entries:
        relative = Path(entry["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"unsafe artifact path: {relative}")
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.stat().st_size != entry["size"]:
            raise ValueError(f"size mismatch: {path}")
        verify_sha256(path, entry["sha256"])
    return manifest


def assemble(args) -> None:
    sources = {
        "backbone": args.backbone,
        "organ": args.organ,
        "diagnosis": args.diagnosis,
        "medoids": args.medoids,
        "retrieval_bank": args.retrieval_bank,
        "retrieval_cots": args.retrieval_cots,
    }
    lock_data = json.loads(args.lock.read_text(encoding="utf-8"))
    if lock_data.get("release") != "0.6.0":
        raise ValueError(f"invalid release lock: {args.lock}")
    locked_files = lock_data.get("files")
    expected_names = {
        path.relative_to("model").as_posix() for path in ARTIFACTS.values()
    }
    if not isinstance(locked_files, dict) or set(locked_files) != expected_names:
        raise ValueError(f"release lock has an invalid file set: {args.lock}")
    if args.skip_existing and (args.out / "artifact-manifest.json").exists():
        verify(args.out)
        print(f"SKIP verified artifact set {args.out}")
        return
    for name, source in sources.items():
        if not source.is_file():
            raise FileNotFoundError(source)
        relative = ARTIFACTS[name].relative_to("model").as_posix()
        verify_sha256(source, locked_files[relative])
    for name, source in sources.items():
        copy_atomic(source, args.out / ARTIFACTS[name])
    entries = []
    for relative in sorted(ARTIFACTS.values()):
        path = args.out / relative
        entries.append(
            {
                "path": relative.as_posix(),
                "size": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    atomic_json(
        args.out / "artifact-manifest.json",
        {
            "schema_version": 1,
            "release": "0.6.0",
            "files": entries,
        },
    )
    model_entries = [
        entry for entry in entries if Path(entry["path"]).parts[0] == "model"
    ]
    checksum_lines = [
        f"{entry['sha256']}  {Path(entry['path']).relative_to('model').as_posix()}"
        for entry in model_entries
    ]
    atomic_text(
        args.out / "model" / "MANIFEST.sha256",
        "\n".join(checksum_lines) + "\n",
    )
    verify(args.out)
    print(f"WROTE verified artifact set {args.out}")


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    assemble_parser = subparsers.add_parser("assemble")
    for name in ARTIFACTS:
        assemble_parser.add_argument(
            f"--{name.replace('_', '-')}", type=Path, required=True
        )
    assemble_parser.add_argument("--lock", type=Path, required=True)
    assemble_parser.add_argument("--out", type=Path, required=True)
    assemble_parser.add_argument(
        "--skip-existing", action=argparse.BooleanOptionalAction, default=True
    )
    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "assemble":
        assemble(args)
    else:
        manifest = verify(args.root)
        print(f"OK {args.root}: {len(manifest['files'])} files")


if __name__ == "__main__":
    main()
