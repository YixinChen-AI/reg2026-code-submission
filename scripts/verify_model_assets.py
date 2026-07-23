from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_lock(path: Path) -> dict[str, str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("release") != "0.6.0" or not isinstance(data.get("files"), dict):
        raise ValueError(f"invalid release lock: {path}")
    return data["files"]


def load_manifest(path: Path) -> dict[str, str]:
    entries: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        digest, separator, name = line.partition("  ")
        name = name.removeprefix("*")
        if not separator or name in entries:
            raise ValueError(f"invalid or duplicate manifest entry: {line!r}")
        entries[name] = digest.lower()
    return entries


def verify(root: Path, lock_path: Path) -> None:
    expected = load_lock(lock_path)
    manifest_path = root / "MANIFEST.sha256"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = load_manifest(manifest_path)
    if manifest != expected:
        raise ValueError("MANIFEST.sha256 does not match the v0.6.0 release lock")

    allowed = set(expected) | {"MANIFEST.sha256", "README.md"}
    actual = {
        path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()
    }
    unexpected = sorted(actual - allowed)
    missing = sorted(set(expected) - actual)
    if unexpected or missing:
        raise ValueError(
            f"model file set differs: missing={missing}, unexpected={unexpected}"
        )

    for name, expected_hash in expected.items():
        path = root / name
        actual_hash = sha256(path)
        if actual_hash != expected_hash:
            raise ValueError(f"SHA-256 mismatch for {name}: {actual_hash}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    args = parser.parse_args()
    verify(args.root, args.lock)
    print(f"OK {args.root}: 6 locked assets")


if __name__ == "__main__":
    main()
