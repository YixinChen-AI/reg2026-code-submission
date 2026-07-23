from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

from training.build_medoids import valid_cot, workflow
from training.common import (
    atomic_json,
    atomic_npz,
    diagnosis_of,
    load_cases,
    verify_sha256,
    wsi_stem,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--features-sha256")
    parser.add_argument("--cot", type=Path, required=True)
    parser.add_argument("--cot-sha256")
    parser.add_argument("--out-bank", type=Path, required=True)
    parser.add_argument("--out-cots", type=Path, required=True)
    parser.add_argument(
        "--skip-existing", action=argparse.BooleanOptionalAction, default=True
    )
    args = parser.parse_args()
    verify_sha256(args.features, args.features_sha256)
    verify_sha256(args.cot, args.cot_sha256)
    if args.skip_existing and args.out_bank.exists() and args.out_cots.exists():
        print(f"SKIP {args.out_bank} and {args.out_cots}")
        return
    with np.load(args.features, allow_pickle=False) as data:
        features = data["feats"].astype(np.float32)
        wsis = data["wsi"].astype(str)
    indices: dict[str, list[int]] = defaultdict(list)
    for index, wsi in enumerate(wsis):
        indices[wsi_stem(wsi)].append(index)
    embeddings = {}
    for stem, selected in sorted(indices.items()):
        embedding = features[selected].mean(axis=0)
        norm = float(np.linalg.norm(embedding))
        embeddings[stem] = embedding / norm if norm else embedding

    rows = []
    cots = {}
    for case in load_cases(args.cot):
        stem = wsi_stem(case.get("id"))
        diagnosis = diagnosis_of(case)
        cot = workflow(case)
        if stem not in embeddings or not diagnosis or not valid_cot(cot):
            continue
        case_id = str(case.get("id"))
        rows.append(
            (case_id, str(case.get("organ", "")).lower(), diagnosis, embeddings[stem])
        )
        cots[case_id] = cot
    rows.sort(key=lambda row: row[0])
    if not rows:
        raise RuntimeError("no valid retrieval exemplars matched the features")
    atomic_npz(
        args.out_bank,
        embs=np.stack([row[3] for row in rows]).astype(np.float16),
        wids=np.asarray([row[0] for row in rows]),
        organs=np.asarray([row[1] for row in rows]),
        dxs=np.asarray([row[2] for row in rows]),
    )
    atomic_json(args.out_cots, cots)
    print(f"WROTE {args.out_bank} and {args.out_cots}: {len(rows)} exemplars")


if __name__ == "__main__":
    main()
