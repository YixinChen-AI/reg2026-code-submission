from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from training.common import atomic_json, load_cases, verify_sha256, wsi_stem


def build_manifest(
    cot_path: Path, wsi_root: Path
) -> tuple[list[dict[str, str]], list[str]]:
    cases = load_cases(cot_path)
    labels: dict[str, str] = {}
    for case in cases:
        case_id = str(case.get("id", ""))
        organ = str(case.get("organ", "")).lower()
        if case_id and organ:
            labels[case_id] = organ
            labels[wsi_stem(case_id)] = organ

    rows: list[dict[str, str]] = []
    unmatched: list[str] = []
    paths = sorted(
        path
        for path in wsi_root.rglob("*")
        if path.is_file() and path.suffix.lower() in {".tif", ".tiff"}
    )
    for path in paths:
        organ = labels.get(path.name) or labels.get(wsi_stem(path.name))
        if organ is None:
            unmatched.append(path.relative_to(wsi_root).as_posix())
            continue
        rows.append(
            {
                "path": path.relative_to(wsi_root).as_posix(),
                "wsi": wsi_stem(path.name),
                "organ": organ,
            }
        )
    stems = [row["wsi"] for row in rows]
    duplicates = sorted(stem for stem, count in Counter(stems).items() if count > 1)
    if duplicates:
        raise ValueError(f"duplicate WSI stems in manifest: {duplicates[:8]}")
    return rows, unmatched


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cot", type=Path, required=True)
    parser.add_argument("--cot-sha256")
    parser.add_argument("--wsi-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--unmatched-out", type=Path)
    parser.add_argument(
        "--skip-existing", action=argparse.BooleanOptionalAction, default=True
    )
    args = parser.parse_args()

    verify_sha256(args.cot, args.cot_sha256)
    if args.skip_existing and args.out.exists():
        print(f"SKIP {args.out}")
        return
    rows, unmatched = build_manifest(args.cot, args.wsi_root)
    if not rows:
        raise RuntimeError("no labeled WSIs matched the case metadata")
    atomic_json(args.out, rows)
    if args.unmatched_out:
        atomic_json(args.unmatched_out, unmatched)
    counts = Counter(row["organ"] for row in rows)
    print(
        f"WROTE {args.out}: {len(rows)} WSIs, unmatched={len(unmatched)}, organs={dict(sorted(counts.items()))}"
    )


if __name__ == "__main__":
    main()
